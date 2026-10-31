#!/usr/bin/env python3
"""Exclusive guarded correctness admission for explicitly listed requests.

Passing per-candidate checks is provisional until an independent raw-evidence
audit verifies the complete guard, ledger, source and execution records.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

import candidate
import identity
import validation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FRONTIER = HERE.parent / 'frontier_execution' / 'runner.py'
LEGACY_MODULES = ('run_suite', 'validation', 'search_controller', 'original_compare_controller',
                  'original_compare_guard', 'audit_admission', 'budget', 'identity', 'manifest',
                  'enumerate_candidates', 'selection')


def load_frontier_components():
    """Load legacy helpers without inheriting or leaking same-name modules.

    The legacy runner and search controller modify sys.path and use bare imports.
    Keep their own mutually consistent modules while restoring caller state even
    if an import or source-location check fails.
    """
    saved_path = sys.path[:]
    saved = {name: sys.modules[name] for name in LEGACY_MODULES if name in sys.modules}
    try:
        for name in LEGACY_MODULES:
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location('nonuniform_frontier_budget_guard', FRONTIER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expanded = HERE.parent / 'expanded_search'
        original = HERE.parent / 'original_compare'
        expected = {
            'run_suite': expanded / 'run_suite.py',
            'validation': expanded / 'validation.py',
            'search_controller': expanded / 'search_controller.py',
            'original_compare_controller': original / 'original_compare_controller.py',
            'original_compare_guard': original / 'original_compare_guard.py',
            'audit_admission': expanded / 'audit_admission.py',
            'budget': expanded / 'budget.py',
            'identity': expanded / 'identity.py',
            'manifest': expanded / 'manifest.py',
            'enumerate_candidates': expanded / 'enumerate_candidates.py',
            'selection': expanded / 'selection.py',
        }
        for name, path in expected.items():
            loaded = sys.modules.get(name)
            if loaded is None or Path(getattr(loaded, '__file__', '')).resolve() != path.resolve():
                raise ImportError('unexpected legacy helper source: ' + name)
        if (module.validation is not sys.modules['validation'] or
                module.search_controller.validation is not sys.modules['validation'] or
                module.search_controller.identity is not sys.modules['identity'] or
                module.search_controller.budget is not sys.modules['budget'] or
                module.search_controller.manifest is not sys.modules['manifest'] or
                module.search_controller.suite is not sys.modules['run_suite'] or
                module.search_controller.original_comparison is not sys.modules['original_compare_controller'] or
                sys.modules['manifest'].enumerate_candidates is not sys.modules['enumerate_candidates'] or
                sys.modules['manifest'].selection is not sys.modules['selection'] or
                module.suite is not sys.modules['run_suite'] or
                module.comparison is not sys.modules['original_compare_controller'] or
                module.comparison.suite is not sys.modules['run_suite'] or
                module.guard_common is not sys.modules['original_compare_guard']):
            raise ImportError('legacy helper module bindings differ')
        return module
    finally:
        sys.path[:] = saved_path
        for name in LEGACY_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


v2 = load_frontier_components()
guard_common = v2.guard_common
PATTERNS = validation.PATTERNS
SEEDS = (42, 53, 71)
MAX_CANDIDATES = 32
PREFIXES = ('VKFFT_NONUNIFORM_', 'VKFFT_EXPANDED_', 'VKFFT_DIRECT_')
GPU_RE = re.compile(r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return v2.sha(path)


def load_manifest(path):
    doc = json.loads(Path(path).read_text(encoding='utf8'))
    need(type(doc) is dict and set(doc) == {'schema', 'candidates'} and
         doc['schema'] == 'nonuniform-admission-manifest/v1', 'invalid manifest schema')
    items = doc['candidates']
    need(type(items) is list and 1 <= len(items) <= MAX_CANDIDATES, 'invalid candidate count')
    seen = set()
    for item in items:
        need(type(item) is dict and set(item) == {'candidate_id', 'request'}, 'invalid manifest candidate')
        cid = candidate.candidate_id(item['request'])
        # The mandatory quarter-tone input is only defined by the harness for N divisible by 4.
        need(item['request']['workload']['n'] >= 4, 'mandatory quarter-tone validation requires n>=4')
        need(item['candidate_id'] == cid and cid not in seen, 'candidate ID mismatch or duplicate')
        seen.add(cid)
    return items


def load_protocol(path):
    doc = json.loads(Path(path).read_text(encoding='utf8'))
    fields = {'schema', 'gpu_uuid', 'hostname', 'boot_id', 'machine_id', 'seconds',
              'idle_seconds', 'timeout', 'graph_repeats', 'patterns', 'seeds'}
    need(type(doc) is dict and set(doc) == fields and doc['schema'] == 'nonuniform-admission-protocol/v1',
         'invalid protocol schema')
    need(type(doc['gpu_uuid']) is str and GPU_RE.fullmatch(doc['gpu_uuid']) is not None,
         'invalid GPU UUID')
    need(type(doc['hostname']) is str and bool(doc['hostname']) and
         type(doc['boot_id']) is str and str(uuid.UUID(doc['boot_id'])) == doc['boot_id'] and
         type(doc['machine_id']) is str and re.fullmatch(r'[0-9a-f]{32}', doc['machine_id']) is not None,
         'invalid host identity')
    need(type(doc['seconds']) in (int, float) and math.isfinite(doc['seconds']) and
         type(doc['idle_seconds']) in (int, float) and math.isfinite(doc['idle_seconds']) and
         0 <= doc['idle_seconds'] <= 3600 and doc['seconds'] > doc['idle_seconds'] and
         type(doc['timeout']) is int and 1 <= doc['timeout'] <= 3600 and
         type(doc['graph_repeats']) is int and 1 <= doc['graph_repeats'] <= 2048,
         'invalid timing or budget configuration')
    need(doc['patterns'] == list(PATTERNS) and doc['seeds'] == list(SEEDS), 'validation matrix differs from contract')
    return doc


def reserve_outputs(paths):
    resolved = [Path(p).resolve() for p in paths]
    need(len(set(resolved)) == len(resolved), 'output paths must be distinct')
    if any(p.exists() for p in resolved):
        raise FileExistsError('evidence output already exists')
    # O_EXCL handles races. A partial reservation is deliberately retained and
    # requires manual inspection; never delete a potentially replaced file.
    for path in resolved:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf8'):
            pass


def tree_inventory(root):
    """Hash every regular file and reject symlinks or special files."""
    root = Path(root).resolve(strict=True)
    need(root.is_dir(), 'tree root is not a directory')
    files = {}
    for path in sorted(root.rglob('*'), key=lambda p: p.relative_to(root).as_posix()):
        need(not path.is_symlink(), 'symlink in source tree')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha(path)
        else:
            need(path.is_dir(), 'unsupported tree entry')
    need(bool(files), 'empty source tree')
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf8')
    return {'files': files, 'sha256': hashlib.sha256(encoded).hexdigest()}


def tracked_instrumentation(root):
    """Hash committed instrumentation files, not ignored build/cache outputs."""
    root = Path(root).resolve(strict=True)
    prefix = 'test/plan_selection_study/instrumentation/'
    paths = v2.git(root, 'ls-files', '--', prefix).splitlines()
    need(paths and all(p.startswith(prefix) for p in paths), 'missing tracked instrumentation')
    files = {}
    for name in paths:
        path = root / name
        need(not path.is_symlink() and path.is_file(), 'invalid instrumentation source')
        files[name[len(prefix):]] = sha(path)
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf8')
    return {'files': files, 'sha256': hashlib.sha256(encoded).hexdigest()}


def verify_generated(root, path, vkfft_commit):
    path = Path(path).resolve(strict=True)
    doc = json.loads(path.read_text(encoding='utf8'))
    details = doc.get('nonuniform_v1', {})
    source = Path(root) / 'test/plan_selection_study/nonuniform_search'
    need(doc.get('upstream_commit') == vkfft_commit and type(doc.get('expanded')) is dict and
         type(doc.get('files')) is dict, 'generated provenance missing or wrong upstream')
    for field, target in (
            ('generator_sha256', source / 'prepare_headers.py'),
            ('candidate_sha256', source / 'candidate.hpp'),
            ('expanded_generator_sha256', Path(root) / 'test/plan_selection_study/expanded_search/prepare_headers.py'),
            ('research_contract_sha256', Path(root) / 'experiments/research/nonuniform_register_admission_contract.json')):
        need(details.get(field) == sha(target), 'generator or contract source hash mismatch')
    required = {'include/candidate.hpp', 'include/trace.hpp', 'expanded_benchmark.cpp',
                'include/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h',
                'include/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h',
                'include/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h'}
    need(required <= set(doc['expanded']), 'generated provenance lacks required files')
    bound = {}
    for rel, expected in doc['expanded'].items():
        target = (path.parent / rel).resolve(strict=True)
        need(target.is_relative_to(path.parent) and sha(target) == expected, 'generated file hash mismatch')
        bound[str(target)] = expected
    upstream = (Path(root) / 'VkFFT/vkFFT').resolve(strict=True)
    names = {'vkFFT_InitializeApp.h', 'vkFFT_Scheduler.h', 'vkFFT_Plan_FFT.h',
             'vkFFT_CompileKernel.h', 'vkFFT_ManageLUT.h', 'vkFFT_DispatchPlan.h'}
    need(len(doc['files']) == len(names) and {Path(x).name for x in doc['files']} == names,
         'incomplete instrumented upstream provenance')
    for rel, hashes in doc['files'].items():
        original = (upstream / rel).resolve(strict=True)
        generated = (path.parent / 'include' / rel).resolve(strict=True)
        need(original.is_relative_to(upstream) and generated.is_relative_to(path.parent / 'include') and
             sha(original) == hashes['original_sha256'], 'upstream instrumented source mismatch')
        # Scheduler and plan were changed again and are bound by expanded.
        generated_rel = generated.relative_to(path.parent).as_posix()
        expected = doc['expanded'].get(generated_rel, hashes['generated_sha256'])
        need(sha(generated) == expected, 'instrumented source hash mismatch')
        bound[str(generated)] = expected
    original_tree = tree_inventory(upstream)
    generated_tree = tree_inventory(path.parent / 'include')
    extras = {'candidate.hpp', 'trace.hpp', 'vkfft_trace_source_metadata.hpp'}
    need(not (extras & set(original_tree['files'])) and
         set(generated_tree['files']) == set(original_tree['files']) | extras,
         'generated tree file inventory differs from copied upstream plus expected extras')
    metadata = path.parent / 'include/vkfft_trace_source_metadata.hpp'
    original_compile = next(v['original_sha256'] for k, v in doc['files'].items()
                            if Path(k).name == 'vkFFT_CompileKernel.h')
    expected_metadata = ('#pragma once\n#define VKFFT_ORIGINAL_COMPILE_HEADER_SHA "' + original_compile + '"\n').encode()
    need(metadata.read_bytes() == expected_metadata, 'source metadata header mismatch')
    for rel, expected in original_tree['files'].items():
        generated_rel = 'include/' + rel
        if generated_rel in doc['expanded']:
            wanted = doc['expanded'][generated_rel]
        elif rel in doc['files']:
            wanted = doc['files'][rel]['generated_sha256']
        else:
            wanted = expected
        need(generated_tree['files'][rel] == wanted, 'untracked generated tree change: ' + rel)
    need(generated_tree['files']['candidate.hpp'] == sha(source / 'candidate.hpp') and
         generated_tree['files']['trace.hpp'] == doc['expanded']['include/trace.hpp'], 'generated support header mismatch')
    return {'path': str(path), 'sha256': sha(path), 'generated_files': bound,
            'original_tree': original_tree, 'generated_tree': generated_tree,
            'instrumentation_tree': tracked_instrumentation(root)}


def verify_build_dependencies(compile_commands, original_dependencies, generated, original, instrumentation):
    """Check actual recorded include precedence and original-target dependency list."""
    generated, original, instrumentation = (Path(x).resolve(strict=True) for x in (generated, original, instrumentation))
    commands = json.loads(Path(compile_commands).read_text(encoding='utf8'))
    need(type(commands) is list, 'invalid compile command database')
    selected = {}
    for entry in commands:
        need(type(entry) is dict and type(entry.get('directory')) is str and type(entry.get('file')) is str,
             'invalid compile command entry')
        words = entry.get('arguments')
        if words is None:
            need(type(entry.get('command')) is str, 'missing compile command')
            words = shlex.split(entry['command'])
        need(type(words) is list and all(type(x) is str for x in words), 'invalid compile arguments')
        for role in ('original', 'candidate'):
            marker = 'vkfft_nonuniform_' + role + '.dir'
            if not any(marker in word for word in words):
                continue
            need(role not in selected, 'duplicate compile target')
            directory = Path(entry['directory']).resolve(strict=True)
            source = Path(entry['file'])
            source = (directory / source).resolve(strict=True) if not source.is_absolute() else source.resolve(strict=True)
            need(source == generated / 'expanded_benchmark.cpp', 'unexpected benchmark source')
            include = []
            for i, word in enumerate(words):
                if word == '-I':
                    need(i + 1 < len(words), 'missing include argument')
                    include.append((directory / words[i + 1]).resolve())
                elif word.startswith('-I') and len(word) > 2:
                    include.append((directory / word[2:]).resolve())
                need(not word.startswith(('-include', '-imacros', '-iquote', '-idirafter', '-Wp,')),
                     'unsupported forced include or preprocessor option')
            expected = ([original, generated / 'include', instrumentation] if role == 'original'
                        else [generated / 'include', instrumentation])
            need(include == expected, 'compile include paths/order differ from isolated targets')
            marker_flag = '-DVKFFT_STUDY_UNINSTRUMENTED=1'
            need((marker_flag in words) == (role == 'original') and '-DVKFFT_BACKEND=1' in words,
                 'original instrumentation or backend flag mismatch')
            need(words.count('-o') == 1 and words.index('-o') + 1 < len(words), 'missing object output')
            obj = (directory / words[words.index('-o') + 1]).resolve()
            selected[role] = obj
    need(set(selected) == {'original', 'candidate'}, 'both target compile commands required')
    need(Path(original_dependencies).resolve(strict=True) == Path(str(selected['original']) + '.d').resolve(strict=True),
         'dependency file is not the original compile object dependency')
    dep = Path(original_dependencies).read_text(encoding='utf8').replace('\\\r\n', ' ').replace('\\\n', ' ')
    need(':' in dep, 'invalid compiler dependency file')
    paths = shlex.split(dep.split(':', 1)[1])
    need(paths, 'empty compiler dependencies')
    # GNU dependency paths are relative to the build directory unless absolute.
    command_dir = next(Path(e['directory']).resolve() for e in commands
                       if 'vkfft_nonuniform_original.dir' in (e.get('command') or ' '.join(e.get('arguments', []))))
    dependencies = {(command_dir / p).resolve() for p in paths}
    required = {original / 'vkFFT.h', generated / 'include/candidate.hpp', generated / 'include/trace.hpp',
                generated / 'include/vkfft_trace_source_metadata.hpp', instrumentation / 'cpu_reference.hpp',
                generated / 'expanded_benchmark.cpp'}
    need(required <= dependencies, 'missing original or harness dependency')
    need(not any(p == generated / 'include/vkFFT.h' or p.is_relative_to(generated / 'include/vkFFT')
                 for p in dependencies), 'original target used generated VkFFT headers')
    return {'compile_commands_sha256': sha(compile_commands), 'original_dependencies_sha256': sha(original_dependencies),
            'limitation': 'recorded command and dependencies bind include order, not independent binary reproduction'}


def verify_attestation(path, code_commit, vkfft_commit, generated, binaries, compile_commands, original_dependencies,
                       original, instrumentation):
    doc = json.loads(Path(path).read_text(encoding='utf8'))
    need(doc.get('schema') == 'nonuniform-admission-build/v1' and doc.get('code_commit') == code_commit and
         doc.get('vkfft_commit') == vkfft_commit and doc.get('generated_provenance_sha256') == generated['sha256'] and
         doc.get('binaries') == binaries and type(doc.get('build_commands')) is list and
         doc['build_commands'] and all(type(x) is str and x for x in doc['build_commands']),
         'build attestation mismatch')
    expected_trees = {name: generated[name + '_tree']['sha256'] for name in ('original', 'generated', 'instrumentation')}
    need(doc.get('tree_sha256') == expected_trees and
         doc.get('compile_commands_sha256') == sha(compile_commands) and
         doc.get('original_dependencies_sha256') == sha(original_dependencies), 'build tree or dependency hash mismatch')
    dependencies = verify_build_dependencies(compile_commands, original_dependencies,
                                             Path(generated['path']).parent, original, instrumentation)
    return {'path': str(Path(path).resolve()), 'sha256': sha(path), 'document': doc,
            'dependencies': dependencies,
            'limitation': 'attested input/binary association, not independent reproduction'}


def verify_declaration(root, execution_commit, experiment_id, protocol_path):
    need(type(experiment_id) is str and re.fullmatch(r'[A-Za-z0-9_.-]+', experiment_id) is not None,
         'invalid experiment ID')
    parent = v2.git(root, 'rev-parse', execution_commit + '^')
    record = f'experiments/records/{experiment_id}.json'
    need(v2.git(root, 'diff', '--name-only', parent, execution_commit).splitlines() == [record],
         'execution commit must be a single experiment declaration')
    doc = json.loads(v2.git(root, 'show', execution_commit + ':' + record))
    need(doc.get('schema') == 'git-experiment/v1' and doc.get('id') == experiment_id and
         doc.get('parent_code_commit') == parent and doc.get('status') == 'declared',
         'experiment declaration mismatch')
    rel = Path(protocol_path).resolve(strict=True).relative_to(Path(root).resolve(strict=True)).as_posix()
    need((doc.get('protocol') or {}).get('path') == rel and
         (doc.get('protocol') or {}).get('sha256') == sha(protocol_path), 'declared protocol mismatch')
    return parent


def verify_frozen(contract):
    for path, expected in contract['frozen_files'].items():
        need(sha(path) == expected, 'frozen evidence or source changed: ' + path)
    roots = contract['tree_roots']
    generated = contract['generated_provenance']
    need(tree_inventory(roots['original']) == generated['original_tree'] and
         tree_inventory(roots['generated']) == generated['generated_tree'] and
         tracked_instrumentation(ROOT) == generated['instrumentation_tree'], 'frozen source tree changed')


class DeadlineLedger:
    """Enforce a stage deadline and the caller-supplied same-boot campaign deadline."""
    def __init__(self, ledger, deadline, clock=time.monotonic):
        self.ledger, self.deadline, self.clock = ledger, deadline, clock

    def start(self, identity, expected):
        if self.clock() + expected >= self.deadline:
            self.ledger.event('denied', identity=identity, reason='absolute_stage_or_campaign_deadline')
            return False
        return self.ledger.start(identity, expected)

    def finish(self, status):
        return self.ledger.finish(status)


def run_one(args, workload, item, phase, round_no, order):
    """One exact child invocation; must be called through Guard.wrapped."""
    name, override = item
    mode = workload['study_mode']
    need(mode in ('describe', 'execute'), 'unsupported study mode')
    allowed = set(candidate.request_environment({
        'schema': candidate.SCHEMA, 'workload': {'n': 2, 'batch': 1}, 'radix': [2],
        'register_table': [0, 0, 2] + [0] * 30, 'profile': 'nonuniform_active_only', 'grouped_batch': 1}))
    need(type(override) is dict and (not override or set(override) == allowed), 'invalid override environment')
    env = {key: value for key, value in os.environ.items() if not key.startswith(PREFIXES)}
    env.update(override)
    command = [str(args.exe), '--n', str(workload['n']), '--batch', str(workload['batch']),
               '--seed', str(workload['seed']), '--timing-mode', 'graph', '--placement', 'outofplace',
               '--graph-repeats', str(workload['graph_repeats']), '--warmup', '0', '--warmup-ms', '0',
               '--repeats', '1', '--pattern', workload['pattern'], '--study-mode', mode,
               '--candidate-id', name]
    start = time.monotonic()
    row = {'record_type': 'execution', 'session_id': args.session_id, 'invocation_id': uuid.uuid4().hex,
           'phase': phase, 'workload_id': f"n{workload['n']}_b{workload['batch']}", 'workload': workload,
           'candidate_id': name, 'round': round_no, 'order': order, 'command': command,
           'direct_environment': override, 'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
           'candidate_environment_policy': 'nonuniform, expanded and direct prefixes cleared before explicit override',
           'executable_sha256': args.exe_sha256, 'validation_errors': [], 'started_monotonic': start,
           'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        need(sha(args.exe) == args.exe_sha256, 'frozen binary changed')
        # The guard has just sampled. Recheck immediately before dispatch, with
        # a margin covering the guard's maximum accepted sample age and overhead.
        if time.monotonic() + args.timeout + 3 >= args.absolute_deadline:
            row.update(record_type='launch_denied', status='budget_denied_before_launch',
                       reason='absolute_stage_or_campaign_deadline')
            row['ended_monotonic'] = time.monotonic()
            row['wall_seconds'] = row['ended_monotonic'] - start
            row['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return row
        proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=args.timeout,
                              check=False, pass_fds=(args.lock_fd,))
        row.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        parsed = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                  for line in proc.stdout.splitlines() if line.strip()]
        need(len(parsed) == 1 and type(parsed[0]) is dict, 'expected exactly one JSON result')
        row['result'] = parsed[0]
        row['status'] = 'ok' if proc.returncode == 0 else 'failed'
        if proc.returncode:
            row['validation_errors'].append(f'harness_exit:{proc.returncode}')
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        row['validation_errors'].append(row['error'])
        if isinstance(error, subprocess.TimeoutExpired):
            def printable(value):
                return value.decode('utf8', errors='replace') if isinstance(value, bytes) else (value or '')
            row.update(stdout=printable(error.stdout), stderr=printable(error.stderr))
    row['ended_monotonic'] = time.monotonic()
    row['wall_seconds'] = row['ended_monotonic'] - start
    row['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return row


def plan_signature(result, original=False):
    snapshot = result['postinit_source_kernels'][0] if original else result['precompile_kernels'][0]
    compiler = snapshot['expected_compiler_input'] if original else snapshot['compiler_input']
    return {'structural_features': result['structural_features'], 'resolved_config': result['resolved_config'],
            'compiled_diagnostics': result['compiled_diagnostics'], 'canonical_descriptor': snapshot['canonical_descriptor'],
            'source_sha256': snapshot['source_sha256'], 'source_bytes': snapshot['source_bytes'],
            'compiler_input': compiler, 'actual_compiler_options': snapshot['actual_compiler_options']}


def compare_plan(described, executed):
    a, b = described['precompile_kernels'][0], executed['precompile_kernels'][0]
    need(all(a[k] == b[k] for k in ('canonical_descriptor', 'source_sha256', 'source_bytes', 'compiler_input')),
         'describe and execute plan/source differ')


def run_admission(args, ledger, contract, items):
    """Run fixed cases under an already-started guard; no performance selection."""
    need(os.environ.get('VKFFT_STUDY_GUARD_UUID') == args.gpu and
         bool(os.environ.get('VKFFT_STUDY_GUARD_ID')), 'GPU guard required')
    args.session_id = uuid.uuid4().hex
    args.contract = contract
    guard_id = os.environ['VKFFT_STUDY_GUARD_ID']
    starts = {}

    def emit(row):
        v2.append(args.output, dict(row, session_id=args.session_id, guard_id=guard_id, contract=contract))

    def one(n, batch, name, req=None, *, phase, mode='execute', pattern='random', seed=42):
        work = {'n': n, 'batch': batch, 'seed': seed, 'pattern': pattern, 'config': {},
                'graph_repeats': args.graph_repeats, 'warmup': 0, 'warmup_ms': 0,
                'repeats': 1, 'study_mode': mode}
        key = {'n': n, 'batch': batch, 'candidate_id': name, 'phase': phase, 'round': 0}
        # Whole-tree hashing can be slow: do it before ledger admission and the
        # guard's fresh before-sample, then recheck the executable at launch.
        verify_frozen(contract)
        if not ledger.start(key, args.timeout + 30):
            raise v2.BudgetStop()
        args.exe = args.original if name == 'original' else args.expanded
        args.exe_sha256 = contract['binaries']['original' if name == 'original' else 'expanded']
        row = run_one(args, work, (name, candidate.request_environment(req) if req else {}), phase, 0, 0)
        emit(row)
        ledger.finish(row['status'])
        if row.get('status') == 'budget_denied_before_launch':
            raise v2.BudgetStop()
        starts['executions'] += 1
        need(row.get('result', {}).get('candidate_id') == name, 'child candidate ID mismatch')
        return row

    emit({'record_type': 'session_start', 'gpu': args.gpu, 'candidate_count': len(items),
          'scope': 'correctness admission only; no candidate timing comparison or performance claim'})
    previous = {}
    for index, item in enumerate(items):
        req, cid = item['request'], item['candidate_id']
        n, batch = req['workload']['n'], req['workload']['batch']
        starts['executions'] = 0
        emit({'record_type': 'candidate_start', 'n': n, 'batch': batch, 'candidate_id': cid, 'request': req})
        try:
            described = validation.check_describe(one(n, batch, cid, req, phase='describe', mode='describe'), args.gpu, req)
            original = validation.check(one(n, batch, 'original', phase='identity'), args.gpu, original=True)
            default = validation.check(one(n, batch, 'expanded_default', phase='identity'), args.gpu)
            v2.comparison.compare(default, original)
            baseline, reason = identity.representable(default)
            if baseline is None:
                emit({'record_type': 'candidate_disposition', 'n': n, 'batch': batch, 'candidate_id': cid,
                      'request': req, 'status': 'unsupported_identity', 'reason': reason})
                continue
            replay = validation.check(one(n, batch, 'identity', baseline, phase='identity'), args.gpu, request=baseline)
            proof = identity.compare_three(original, default, replay)
            bases = (plan_signature(original, original=True), plan_signature(default))
            if (n, batch) in previous:
                need(previous[(n, batch)] == bases, 'default plan drift within session')
            previous[(n, batch)] = bases
            emit({'record_type': 'identity_passed', 'n': n, 'batch': batch, 'candidate_id': cid, 'proof': proof})
            signature = None
            for pattern in PATTERNS:
                for seed in SEEDS:
                    row = one(n, batch, cid, req, phase='validation', pattern=pattern, seed=seed)
                    result = validation.check(row, args.gpu, request=req)
                    compare_plan(described, result)
                    current = plan_signature(result)
                    if signature is not None:
                        need(current == signature, 'candidate plan or compiled resource drift')
                    signature = current
            emit({'record_type': 'candidate_disposition', 'n': n, 'batch': batch, 'candidate_id': cid,
                  'request': req, 'status': 'checks_passed_pending_audit', 'validation_count': 15,
                  'interpretation': 'single-session checks only; independent guard/ledger/source audit still required'})
        except v2.BudgetStop:
            emit({'record_type': 'candidate_disposition', 'n': n, 'batch': batch, 'candidate_id': cid,
                  'request': req, 'status': 'partial' if starts['executions'] else 'not_started',
                  'reason': 'budget_denied'})
            for remaining in items[index + 1:]:
                rw = remaining['request']['workload']
                emit({'record_type': 'candidate_disposition', 'n': rw['n'], 'batch': rw['batch'],
                      'candidate_id': remaining['candidate_id'], 'request': remaining['request'],
                      'status': 'not_started', 'reason': 'budget_denied'})
            raise
        except Exception as error:
            emit({'record_type': 'candidate_disposition', 'n': n, 'batch': batch, 'candidate_id': cid,
                  'request': req, 'status': 'candidate_failed',
                  'error': f'{type(error).__name__}: {error}', 'interpretation': 'session stops; no local failure classification'})
            for remaining in items[index + 1:]:
                rw = remaining['request']['workload']
                emit({'record_type': 'candidate_disposition', 'n': rw['n'], 'batch': rw['batch'],
                      'candidate_id': remaining['candidate_id'], 'request': remaining['request'],
                      'status': 'not_started', 'reason': 'prior_candidate_failure'})
            raise
    emit({'record_type': 'session_checks_complete', 'status': 'pending_independent_audit'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'manifest', 'original', 'expanded', 'generated-provenance', 'build-attestation',
                 'compile-commands', 'original-dependencies',
                 'output', 'budget-ledger', 'log', 'incidents', 'lock'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--campaign-deadline-monotonic', required=True, type=float,
                        help='absolute same-boot campaign deadline from an external total budget')
    args = parser.parse_args(argv)
    if os.name != 'posix' or not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        parser.error('Linux pidfd and pidfd_send_signal required')
    for field in ('protocol', 'manifest', 'original', 'expanded', 'generated_provenance', 'build_attestation',
                  'compile_commands', 'original_dependencies'):
        setattr(args, field, getattr(args, field).resolve(strict=True))
    protocol, items = load_protocol(args.protocol), load_manifest(args.manifest)
    args.gpu = protocol['gpu_uuid']
    args.timeout, args.graph_repeats = protocol['timeout'], protocol['graph_repeats']
    boot = Path('/proc/sys/kernel/random/boot_id').read_text(encoding='utf8').strip()
    machine = Path('/etc/machine-id').read_text(encoding='utf8').strip()
    need((socket.gethostname(), boot, machine) == (protocol['hostname'], protocol['boot_id'], protocol['machine_id']),
         'hostname, boot ID or machine ID differs from protocol')
    need(math.isfinite(args.campaign_deadline_monotonic) and args.campaign_deadline_monotonic > time.monotonic(),
         'expired or invalid campaign deadline')
    need(args.original != args.expanded, 'original and generated planner binaries must differ')
    expected_lock = Path('/run/lock') / ('vkfft-study-' + args.gpu + '.lock')
    need(args.lock.resolve() == expected_lock.resolve(), 'lock must be the shared device UUID lock')
    outputs = [args.output, args.budget_ledger, args.log, args.incidents]
    inputs = [args.protocol, args.manifest, args.original, args.expanded, args.generated_provenance, args.build_attestation,
              args.compile_commands, args.original_dependencies]
    need(len({p.resolve() for p in [*outputs, *inputs, args.lock]}) == len(outputs) + len(inputs) + 1,
         'input, output and lock paths must be distinct')
    root = Path(v2.git(HERE, 'rev-parse', '--show-toplevel')).resolve(strict=True)
    need(root == ROOT, 'unexpected repository root')
    vkfft_root = root / 'VkFFT/vkFFT'
    for repo in (root, vkfft_root):
        v2.git(repo, 'diff', '--quiet')
        v2.git(repo, 'diff', '--cached', '--quiet')
    execution_commit = v2.git(root, 'rev-parse', 'HEAD')
    need(os.environ.get('EXPERIMENT_CODE_COMMIT') == execution_commit,
         'run must be bound to experiment execution commit')
    code_commit = verify_declaration(root, execution_commit, os.environ.get('EXPERIMENT_ID'), args.protocol)
    vkfft_commit = v2.git(vkfft_root, 'rev-parse', 'HEAD')
    source_files = [HERE / name for name in ('admission.py', 'candidate.py', 'candidate.hpp', 'identity.py',
                                               'validation.py', 'prepare_headers.py', 'CMakeLists.txt')]
    source_files += [FRONTIER, HERE.parent / 'expanded_search/prepare_headers.py',
                     HERE.parent / 'expanded_search/device_lock.py',
                     HERE.parent / 'original_compare/original_compare_guard.py',
                     HERE.parent / 'original_compare/original_compare_controller.py',
                     HERE.parent / 'instrumentation/prepare_headers.py',
                     HERE.parent / 'instrumentation/trace.hpp',
                     HERE.parent / 'instrumentation/vkfft_instrumented_benchmark.cpp',
                     HERE.parent / 'instrumentation/cpu_reference.hpp',
                     root / 'experiments/research/nonuniform_register_admission_contract.json',
                     args.protocol, args.manifest]
    for path in source_files:
        need(path.resolve(strict=True).is_relative_to(root), 'source outside repository')
        v2.git(root, 'ls-files', '--error-unmatch', str(path))
    generated = verify_generated(root, args.generated_provenance, vkfft_commit)
    binaries = {'original': sha(args.original), 'expanded': sha(args.expanded)}
    attestation = verify_attestation(args.build_attestation, code_commit, vkfft_commit, generated, binaries,
                                     args.compile_commands, args.original_dependencies, vkfft_root,
                                     root / 'test/plan_selection_study/instrumentation')
    frozen = {str(p.resolve()): sha(p) for p in [*source_files, *inputs]}
    frozen.update(generated['generated_files'])
    contract = {'schema': 'nonuniform-admission-execution/v1', 'execution_commit': execution_commit,
                'code_commit': code_commit, 'vkfft_commit': vkfft_commit, 'gpu': args.gpu,
                'hostname': protocol['hostname'], 'boot_id': boot, 'machine_id': machine,
                'campaign_deadline_monotonic': args.campaign_deadline_monotonic,
                'binaries': binaries, 'protocol': {'path': str(args.protocol), 'sha256': sha(args.protocol)},
                'manifest': {'path': str(args.manifest), 'sha256': sha(args.manifest)},
                'generated_provenance': generated, 'build_attestation': attestation, 'frozen_files': frozen,
                'tree_roots': {'original': str(vkfft_root.resolve()),
                               'generated': str((args.generated_provenance.parent / 'include').resolve())},
                'scope': 'single-session correctness checks; not independent audit or timing performance'}
    verify_frozen(contract)
    # Load the Linux-only flock helper without leaving its directory on sys.path
    # or publishing a bare-name module in the caller's import cache.
    lock_spec = importlib.util.spec_from_file_location('nonuniform_device_lock',
                                                       HERE.parent / 'expanded_search/device_lock.py')
    device_lock = importlib.util.module_from_spec(lock_spec)
    lock_spec.loader.exec_module(device_lock)
    global run_one
    with device_lock.DeviceLock(args.lock) as lock:
        reserve_outputs(outputs)
        args.lock_fd = lock.fd
        base_ledger = v2.Ledger(args.budget_ledger, contract, protocol['seconds'], boot=boot)
        args.absolute_deadline = min(base_ledger.state['start'] + protocol['seconds'], args.campaign_deadline_monotonic)
        ledger = DeadlineLedger(base_ledger, args.absolute_deadline)
        guard_args = argparse.Namespace(gpu=args.gpu, log=args.log, incidents=args.incidents,
                                        idle_seconds=protocol['idle_seconds'])
        guard = guard_common.Guard(guard_args)
        original_popen, original_run = subprocess.Popen, run_one
        keys = ('CUDA_VISIBLE_DEVICES', 'VKFFT_STUDY_GUARD_ID', 'VKFFT_STUDY_GUARD_UUID', 'VKFFT_STUDY_GUARD_LOG')
        saved = {key: os.environ.get(key) for key in keys}
        try:
            os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, VKFFT_STUDY_GUARD_ID=guard.guard_id,
                              VKFFT_STUDY_GUARD_UUID=args.gpu, VKFFT_STUDY_GUARD_LOG=str(args.log.resolve()))
            guard.emit('guard_start', script=str(HERE / 'admission.py'), idle_seconds=protocol['idle_seconds'],
                       memory_limit_mib=guard_common.MEMORY_LIMIT_MIB, monitor_interval_seconds=1,
                       parent_pid=os.getpid(), own_signal_method='pidfd_with_rechecked_PPID_exe_starttime')
            guard.idle_check()
            subprocess.Popen = guard.popen(original_popen)
            run_one = guard.wrapped(original_run)
            guard.start()
            try:
                run_admission(args, ledger, contract, items)
            except v2.BudgetStop:
                v2.append(args.output, {'record_type': 'budget_exhausted', 'contract': contract,
                                        'guard_id': guard.guard_id, 'session_id': args.session_id})
            verify_frozen(contract)
            guard.sample('final'); guard.check(); guard.close(); guard.check()
            if guard.thread is not None and guard.thread.is_alive():
                raise guard_common.GuardAbort('monitor failed to stop')
            guard.emit('guard_complete', success=True, clean=True, script_exit_code=0, sample_count=guard.sample_count)
        except BaseException as error:
            if not guard.abort.is_set():
                guard.trip([f'guard_or_target_exception:{type(error).__name__}:{error}'])
            guard.close()
            guard.emit('guard_abort', incident=True, success=False, clean=False,
                       error=f'{type(error).__name__}: {error}', sample_count=guard.sample_count)
            raise
        finally:
            guard.close()
            subprocess.Popen, run_one = original_popen, original_run
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == '__main__':
    main()
