"""CPU/mock checks for the admission controller; never GPU admission evidence."""
import copy
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import admission
import candidate


GPU = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
BOOT = '12345678-1234-1234-1234-123456789abc'
MACHINE = '0123456789abcdef0123456789abcdef'


def request():
    table = [0] * 33
    table[2], table[8] = 4, 8
    return {'schema': candidate.SCHEMA, 'workload': {'n': 128, 'batch': 16},
            'radix': [8, 8, 2], 'register_table': table,
            'profile': 'nonuniform_active_only', 'grouped_batch': 2}


def item(req=None):
    req = request() if req is None else req
    return {'candidate_id': candidate.candidate_id(req), 'request': req}


def protocol():
    return {'schema': 'nonuniform-admission-protocol/v1', 'gpu_uuid': GPU,
            'hostname': 'gpu-test', 'boot_id': BOOT, 'machine_id': MACHINE,
            'seconds': 1200, 'idle_seconds': 1, 'timeout': 10,
            'graph_repeats': 64, 'patterns': list(admission.PATTERNS),
            'seeds': list(admission.SEEDS)}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


class AdmissionInputsTests(unittest.TestCase):
    def test_deadline_ledger_rejects_equal_boundary_and_delegates_earlier_work(self):
        backing = mock.Mock()
        backing.start.return_value = True
        now = [100.0]
        ledger = admission.DeadlineLedger(backing, 140.0, clock=lambda: now[0])
        identity = {'candidate_id': 'fixture', 'phase': 'validation'}
        self.assertFalse(ledger.start(identity, 40))
        backing.start.assert_not_called()
        backing.event.assert_called_once_with('denied', identity=identity,
                                              reason='absolute_stage_or_campaign_deadline')
        now[0] = 99.0
        self.assertTrue(ledger.start(identity, 40))
        backing.start.assert_called_once_with(identity, 40)
        ledger.finish('ok')
        backing.finish.assert_called_once_with('ok')

    def test_manifest_recomputes_id_validates_complete_request_and_preserves_order(self):
        first = item(); req = request(); req['radix'] = [2, 8, 8]; second = item(req)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / 'manifest.json'
            manifest = {'schema': 'nonuniform-admission-manifest/v1', 'candidates': [first, second]}
            write_json(path, manifest)
            self.assertEqual(admission.load_manifest(path), [first, second])
            invalid = []
            bad = copy.deepcopy(manifest); bad['candidates'].append(first); invalid.append(bad)
            bad = copy.deepcopy(manifest); bad['candidates'][0]['candidate_id'] = 'a' * 64; invalid.append(bad)
            bad = copy.deepcopy(manifest); bad['candidates'][0]['request']['register_table'][8] = 4; invalid.append(bad)
            bad = copy.deepcopy(manifest); bad['candidates'][0]['unexpected'] = 1; invalid.append(bad)
            bad = copy.deepcopy(manifest); bad['schema'] = 'other/v1'; invalid.append(bad)
            bad = copy.deepcopy(manifest); bad['candidates'] = []; invalid.append(bad)
            table = [0] * 33; table[2] = 2
            small = {'schema': candidate.SCHEMA, 'workload': {'n': 2, 'batch': 1}, 'radix': [2],
                     'register_table': table, 'profile': 'nonuniform_active_only', 'grouped_batch': 1}
            bad = copy.deepcopy(manifest); bad['candidates'] = [item(small)]; invalid.append(bad)
            bad = copy.deepcopy(manifest)
            bad['candidates'] = [{'candidate_id': row['candidate_id'], 'request': row['request']}
                                 for row in itertools.islice(candidate.enumerate_candidates(128, 16), 33)]
            invalid.append(bad)
            bad = copy.deepcopy(manifest); bad['unexpected'] = 1; invalid.append(bad)
            for doc in invalid:
                write_json(path, doc)
                with self.subTest(doc=doc), self.assertRaises((ValueError, KeyError, TypeError)):
                    admission.load_manifest(path)

    def test_protocol_is_explicit_and_cannot_shrink_validation_matrix(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / 'protocol.json'; good = protocol(); write_json(path, good)
            self.assertEqual(admission.load_protocol(path), good)
            bad = []
            for key, value in [('schema', 'other/v1'), ('gpu_uuid', '0'), ('seconds', 0),
                               ('timeout', 0), ('graph_repeats', 0), ('graph_repeats', 2049),
                               ('patterns', ['random']), ('seeds', [42]), ('machine_id', '')]:
                doc = copy.deepcopy(good); doc[key] = value; bad.append(doc)
            doc = copy.deepcopy(good); doc.pop('boot_id'); bad.append(doc)
            doc = copy.deepcopy(good); doc['extra'] = 'unsafe'; bad.append(doc)
            for doc in bad:
                write_json(path, doc)
                with self.subTest(doc=doc), self.assertRaises((ValueError, KeyError, TypeError)):
                    admission.load_protocol(path)

    def test_reserve_outputs_is_exclusive_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); first, second = root / 'rows', root / 'guard'
            first.write_text('preserve', encoding='utf8')
            with self.assertRaises((FileExistsError, ValueError)):
                admission.reserve_outputs([first, second])
            self.assertEqual(first.read_text(encoding='utf8'), 'preserve')
            self.assertFalse(second.exists())
            with self.assertRaises((FileExistsError, ValueError)):
                admission.reserve_outputs([second, second])
            self.assertFalse(second.exists())
            third = root / 'budget'
            admission.reserve_outputs([second, third])
            self.assertEqual(second.read_bytes(), b'')
            self.assertEqual(third.read_bytes(), b'')
            with self.assertRaises(FileExistsError):
                admission.reserve_outputs([second, third])

    def test_build_attestation_binds_commits_provenance_and_binaries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); path = root / 'attestation.json'
            bins = {'original': 'a' * 64, 'expanded': 'b' * 64}
            commands = root / 'compile_commands.json'; deps = root / 'original.o.d'
            commands.write_text('[]', encoding='utf8'); deps.write_text('obj: x', encoding='utf8')
            generated = {'path': str(root / 'expanded_provenance.json'), 'sha256': 'e' * 64,
                         'original_tree': {'sha256': '1' * 64}, 'generated_tree': {'sha256': '2' * 64},
                         'instrumentation_tree': {'sha256': '3' * 64}}
            doc = {'schema': 'nonuniform-admission-build/v1', 'code_commit': 'c' * 40,
                   'vkfft_commit': 'd' * 40, 'generated_provenance_sha256': 'e' * 64,
                   'binaries': bins, 'build_commands': ['cmake --build frozen-tree'],
                   'compile_commands_sha256': admission.sha(commands), 'original_dependencies_sha256': admission.sha(deps),
                   'tree_sha256': {'original': '1' * 64, 'generated': '2' * 64, 'instrumentation': '3' * 64}}
            arguments = (path, 'c' * 40, 'd' * 40, generated, bins, commands, deps, root, root)
            with mock.patch.object(admission, 'verify_build_dependencies', return_value={'status': 'fixture'}):
                write_json(path, doc)
                proof = admission.verify_attestation(*arguments)
                self.assertEqual(proof['sha256'], admission.sha(path))
                for key, value in [('code_commit', 'f' * 40), ('vkfft_commit', 'f' * 40),
                                   ('generated_provenance_sha256', 'f' * 64), ('binaries', {'original': 'a' * 64}),
                                   ('build_commands', []), ('tree_sha256', {}), ('compile_commands_sha256', 'a' * 64),
                                   ('original_dependencies_sha256', 'a' * 64)]:
                    changed = copy.deepcopy(doc); changed[key] = value; write_json(path, changed)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        admission.verify_attestation(*arguments)

    def test_generated_provenance_detects_generated_and_upstream_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); generated_root = root / 'generated'; include = generated_root / 'include'
            originals = root / 'VkFFT/vkFFT'
            source_files = {
                'generator_sha256': root / 'test/plan_selection_study/nonuniform_search/prepare_headers.py',
                'candidate_sha256': root / 'test/plan_selection_study/nonuniform_search/candidate.hpp',
                'expanded_generator_sha256': root / 'test/plan_selection_study/expanded_search/prepare_headers.py',
                'research_contract_sha256': root / 'experiments/research/nonuniform_register_admission_contract.json',
            }
            for path in source_files.values():
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(str(path.name), encoding='utf8')
            header_paths = [
                'vkFFT_AppManagement/vkFFT_InitializeApp.h',
                'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h',
                'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h',
                'vkFFT_AppManagement/vkFFT_CompileKernel.h',
                'vkFFT_AppManagement/vkFFT_ManageLUT.h',
                'vkFFT_AppManagement/vkFFT_DispatchPlan.h',
            ]
            files, expanded = {}, {}
            for rel in header_paths:
                old, new = originals / rel, include / rel
                old.parent.mkdir(parents=True, exist_ok=True); new.parent.mkdir(parents=True, exist_ok=True)
                old.write_text('upstream:' + rel, encoding='utf8')
                new.write_text('generated:' + rel, encoding='utf8')
                files[rel] = {'original_sha256': admission.sha(old), 'generated_sha256': admission.sha(new)}
            for rel in ('include/candidate.hpp', 'include/trace.hpp', 'expanded_benchmark.cpp',
                        'include/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h'):
                target = generated_root / rel; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('generated:' + rel, encoding='utf8')
                expanded[rel] = admission.sha(target)
            split_rel = 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h'
            (originals / split_rel).write_text('upstream:' + split_rel, encoding='utf8')
            (include / 'candidate.hpp').write_bytes(source_files['candidate_sha256'].read_bytes())
            expanded['include/candidate.hpp'] = admission.sha(include / 'candidate.hpp')
            compile_sha = files[header_paths[3]]['original_sha256']
            (include / 'vkfft_trace_source_metadata.hpp').write_bytes(
                ('#pragma once\n#define VKFFT_ORIGINAL_COMPILE_HEADER_SHA "' + compile_sha + '"\n').encode('utf8'))
            for rel in (header_paths[1], header_paths[2]):
                expanded['include/' + rel] = admission.sha(include / rel)
            doc = {'upstream_commit': 'upstream-fixture', 'files': files, 'expanded': expanded,
                   'nonuniform_v1': {field: admission.sha(path) for field, path in source_files.items()}}
            provenance = generated_root / 'expanded_provenance.json'; write_json(provenance, doc)
            with mock.patch.object(admission, 'tracked_instrumentation',
                                   return_value={'files': {'trace.hpp': 'fixture'}, 'sha256': 'f' * 64}):
                self.assertEqual(admission.verify_generated(root, provenance, 'upstream-fixture')['sha256'],
                                 admission.sha(provenance))
                target = include / 'candidate.hpp'; target.write_text('tampered', encoding='utf8')
                with self.assertRaises(ValueError):
                    admission.verify_generated(root, provenance, 'upstream-fixture')
                target.write_bytes(source_files['candidate_sha256'].read_bytes())
                (originals / header_paths[0]).write_text('tampered', encoding='utf8')
                with self.assertRaises(ValueError):
                    admission.verify_generated(root, provenance, 'upstream-fixture')

    def test_declaration_binds_exact_committed_protocol_and_parent(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); policy = root / 'protocol.json'; policy.write_text('{}', encoding='utf8')
            record = {'schema': 'git-experiment/v1', 'id': 'fixture-run', 'status': 'declared',
                      'parent_code_commit': 'parent',
                      'protocol': {'path': 'protocol.json', 'sha256': admission.sha(policy)}}
            def git_fixture(_root, *args):
                if args[0] == 'rev-parse': return 'parent'
                if args[0] == 'diff': return 'experiments/records/fixture-run.json'
                if args[0] == 'show': return json.dumps(record)
                raise AssertionError('unexpected git query')
            with mock.patch.object(admission.v2, 'git', side_effect=git_fixture):
                self.assertEqual(admission.verify_declaration(root, 'commit', 'fixture-run', policy), 'parent')
                policy.write_text('{"changed": true}', encoding='utf8')
                with self.assertRaises(ValueError):
                    admission.verify_declaration(root, 'commit', 'fixture-run', policy)
                policy.write_text('{}', encoding='utf8')
                record['parent_code_commit'] = 'different'
                with self.assertRaises(ValueError):
                    admission.verify_declaration(root, 'commit', 'fixture-run', policy)

    def test_compiler_include_order_and_original_dependencies_are_checked(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); build = root / 'build'; generated = root / 'generated'
            original = root / 'original'; instrumentation = root / 'instrumentation'
            for path in (build, generated / 'include', original, instrumentation):
                path.mkdir(parents=True, exist_ok=True)
            source = generated / 'expanded_benchmark.cpp'; source.write_text('source', encoding='utf8')
            (original / 'vkFFT.h').write_text('original', encoding='utf8')
            (generated / 'include/candidate.hpp').write_text('candidate', encoding='utf8')
            (generated / 'include/trace.hpp').write_text('trace', encoding='utf8')
            (generated / 'include/vkfft_trace_source_metadata.hpp').write_text('metadata', encoding='utf8')
            (instrumentation / 'cpu_reference.hpp').write_text('reference', encoding='utf8')
            objdir = build / 'CMakeFiles/vkfft_nonuniform_original.dir'; objdir.mkdir(parents=True)
            obj = objdir / 'expanded_benchmark.cpp.o'; deps = Path(str(obj) + '.d')
            dep_values = [source, original / 'vkFFT.h', generated / 'include/candidate.hpp',
                          generated / 'include/trace.hpp', generated / 'include/vkfft_trace_source_metadata.hpp',
                          instrumentation / 'cpu_reference.hpp']
            deps.write_text('obj: ' + ' '.join(p.as_posix() for p in dep_values), encoding='utf8')
            def command(role):
                include = [original, generated / 'include', instrumentation] if role == 'original' else [generated / 'include', instrumentation]
                flags = ['-DVKFFT_BACKEND=1'] + (['-DVKFFT_STUDY_UNINSTRUMENTED=1'] if role == 'original' else [])
                return {'directory': build.as_posix(), 'file': source.as_posix(),
                        'arguments': ['c++', *flags, *['-I' + p.as_posix() for p in include],
                                      '-o', f'CMakeFiles/vkfft_nonuniform_{role}.dir/expanded_benchmark.cpp.o']}
            commands = [command('original'), command('candidate')]
            database = root / 'compile_commands.json'; write_json(database, commands)
            proof = admission.verify_build_dependencies(database, deps, generated, original, instrumentation)
            self.assertEqual(proof['compile_commands_sha256'], admission.sha(database))
            bad = copy.deepcopy(commands); bad[0]['arguments'][3], bad[0]['arguments'][4] = (
                bad[0]['arguments'][4], bad[0]['arguments'][3])
            write_json(database, bad)
            with self.assertRaises(ValueError):
                admission.verify_build_dependencies(database, deps, generated, original, instrumentation)
            write_json(database, commands)
            deps.write_text('obj: ' + ' '.join(p.as_posix() for p in dep_values) + ' ' +
                            (generated / 'include/vkFFT.h').as_posix(), encoding='utf8')
            with self.assertRaises(ValueError):
                admission.verify_build_dependencies(database, deps, generated, original, instrumentation)


class MockLedger:
    def __init__(self, deny_phase=None):
        self.deny_phase = deny_phase
        self.started = []
        self.finished = []

    def start(self, identity, expected):
        self.started.append((copy.deepcopy(identity), expected))
        return identity['phase'] != self.deny_phase

    def finish(self, status):
        self.finished.append(status)


class AdmissionFlowTests(unittest.TestCase):
    def test_unguarded_controller_never_starts_work(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            args = SimpleNamespace(gpu=GPU, output=Path(tmp) / 'rows.jsonl', timeout=10, graph_repeats=64)
            ledger = MockLedger()
            with mock.patch.dict(os.environ, {'VKFFT_STUDY_GUARD_UUID': GPU, 'VKFFT_STUDY_GUARD_ID': ''}):
                with self.assertRaises(ValueError):
                    admission.run_admission(args, ledger, {}, [item()])
            self.assertEqual(ledger.started, [])
            self.assertFalse(args.output.exists())

    def test_child_launch_scrubs_overrides_binds_binary_and_inherits_lock(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            executable = Path(tmp) / 'fake-binary'; executable.write_bytes(b'fixture')
            digest = admission.sha(executable)
            args = SimpleNamespace(exe=executable, exe_sha256=digest, session_id='session-fixture',
                                   timeout=10, lock_fd=42, absolute_deadline=10 ** 12,
                                   contract={'frozen_files': {str(executable): digest}})
            work = {'n': 128, 'batch': 16, 'seed': 42, 'pattern': 'random', 'graph_repeats': 64,
                    'warmup': 0, 'warmup_ms': 0, 'repeats': 1, 'study_mode': 'execute', 'config': {}}
            captured = []
            def child(command, **kwargs):
                captured.append((command, kwargs))
                name = command[command.index('--candidate-id') + 1]
                return SimpleNamespace(returncode=0, stdout=json.dumps({'candidate_id': name}), stderr='')
            dirty = {'CUDA_VISIBLE_DEVICES': GPU, 'VKFFT_EXPANDED_REGS': '16',
                     'VKFFT_DIRECT_BATCH': '4', 'VKFFT_NONUNIFORM_TABLE': 'invalid',
                     'VKFFT_NONUNIFORM_UNEXPECTED': 'must-clear'}
            with mock.patch.dict(os.environ, dirty), mock.patch.object(admission.subprocess, 'run', side_effect=child):
                row = admission.run_one(args, work, ('original', {}), 'identity', 0, 0)
                self.assertEqual(row['status'], 'ok')
                command, kwargs = captured[0]
                self.assertEqual(kwargs['pass_fds'], (42,))
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], GPU)
                self.assertFalse(any(key.startswith(admission.PREFIXES) for key in kwargs['env']))
                self.assertEqual(command[command.index('--candidate-id') + 1], 'original')
                self.assertEqual(json.loads(row['stdout']), row['result'])
                row = admission.run_one(args, work, ('candidate', candidate.request_environment(request())),
                                        'validation', 0, 0)
                _, kwargs = captured[-1]
                self.assertEqual({k: v for k, v in kwargs['env'].items() if k.startswith(admission.PREFIXES)},
                                 candidate.request_environment(request()))
                self.assertEqual(row['direct_environment'], candidate.request_environment(request()))
                with mock.patch.object(admission.subprocess, 'run', return_value=SimpleNamespace(
                        returncode=0, stdout='{"x": NaN}', stderr='')):
                    malformed = admission.run_one(args, work, ('original', {}), 'identity', 0, 0)
                    self.assertEqual(malformed['status'], 'failed')
                executable.write_bytes(b'tampered')
                row = admission.run_one(args, work, ('original', {}), 'identity', 0, 0)
                self.assertEqual(row['status'], 'failed')
                self.assertEqual(len(captured), 2)

    def test_near_absolute_deadline_records_denial_without_launch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            executable = Path(tmp) / 'fake-binary'; executable.write_bytes(b'fixture')
            digest = admission.sha(executable)
            args = SimpleNamespace(exe=executable, exe_sha256=digest, session_id='session-fixture',
                                   timeout=10, lock_fd=42, absolute_deadline=113.0,
                                   contract={'frozen_files': {str(executable): digest}})
            work = {'n': 128, 'batch': 16, 'seed': 42, 'pattern': 'random', 'graph_repeats': 64,
                    'warmup': 0, 'warmup_ms': 0, 'repeats': 1, 'study_mode': 'execute', 'config': {}}
            with mock.patch.object(admission.time, 'monotonic', return_value=100.0), \
                 mock.patch.object(admission.subprocess, 'run') as process:
                row = admission.run_one(args, work, ('original', {}), 'identity', 0, 0)
            process.assert_not_called()
            self.assertEqual(row['record_type'], 'launch_denied')
            self.assertEqual(row['status'], 'budget_denied_before_launch')

    def run_case(self, *, deny_phase=None, deny_launch_phase=None, failed_phase=None, status_error=False, num_items=1,
                 unsupported_identity=False):
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name); output = root / 'rows.jsonl'; output.touch(exist_ok=False)
        requests = [item()]
        if num_items == 2:
            req = request(); req['radix'] = [2, 8, 8]; requests.append(item(req))
        args = SimpleNamespace(gpu=GPU, original=root / 'original', expanded=root / 'expanded',
                               output=output, timeout=10, graph_repeats=64, absolute_deadline=10 ** 12)
        contract = {'gpu': GPU, 'binaries': {'original': 'a' * 64, 'expanded': 'b' * 64}}
        ledger = MockLedger(deny_phase)
        calls = []
        snapshot = {'canonical_descriptor': {'same': 1}, 'source_sha256': 'c' * 64,
                    'source_bytes': 50, 'compiler_input': {'options': ['option']}}

        def one(args_, workload, candidate_, phase, round_no, order):
            name, environment = candidate_
            calls.append({'phase': phase, 'name': name, 'env': copy.deepcopy(environment),
                          'pattern': workload['pattern'], 'seed': workload['seed']})
            if phase == deny_launch_phase:
                return {'record_type': 'launch_denied', 'phase': phase, 'candidate_id': name,
                        'status': 'budget_denied_before_launch', 'workload': workload}
            result = {'candidate_id': name, 'status': 'error' if phase == failed_phase and status_error else 'ok',
                      'precompile_kernels': [copy.deepcopy(snapshot)]}
            if phase == failed_phase and status_error:
                result.update(failure_phase='initialize', error='initializeVkFFT failed with code 3002')
            return {'record_type': 'execution', 'phase': phase, 'candidate_id': name,
                    'workload': workload, 'direct_environment': environment,
                    'result': result, 'status': 'failed' if result['status'] == 'error' else 'ok',
                    'returncode': 1 if result['status'] == 'error' else 0,
                    'stdout': json.dumps(result), 'stderr': '', 'guard_clean': True, 'guard_gpu_uuid': GPU}

        def checked(row, *unused, **kwargs):
            if row['phase'] == failed_phase:
                raise ValueError('synthetic validation failure')
            return row['result']

        env = {'VKFFT_STUDY_GUARD_UUID': GPU, 'VKFFT_STUDY_GUARD_ID': 'guard-fixture',
               'CUDA_VISIBLE_DEVICES': GPU}
        with mock.patch.dict(os.environ, env), mock.patch.object(admission, 'run_one', side_effect=one), \
             mock.patch.object(admission, 'verify_frozen', return_value=None), \
             mock.patch.object(admission.validation, 'check', side_effect=checked), \
             mock.patch.object(admission.validation, 'check_describe', side_effect=checked), \
             mock.patch.object(admission.identity, 'representable',
                               return_value=(None, 'unsupported_table') if unsupported_identity else (request(), None)), \
             mock.patch.object(admission.identity, 'compare_three', return_value={'status': 'match'}), \
             mock.patch.object(admission.v2.comparison, 'compare', return_value=None), \
             mock.patch.object(admission, 'compare_plan', return_value=None), \
             mock.patch.object(admission, 'plan_signature', return_value='stable'):
            if deny_phase or deny_launch_phase:
                with self.assertRaises(admission.v2.BudgetStop):
                    admission.run_admission(args, ledger, contract, requests)
            elif failed_phase:
                with self.assertRaises((ValueError, RuntimeError)):
                    admission.run_admission(args, ledger, contract, requests)
            else:
                admission.run_admission(args, ledger, contract, requests)
        rows = [json.loads(line) for line in output.read_text(encoding='utf8').splitlines() if line]
        return requests, ledger, calls, rows

    def test_exact_matrix_completes_before_pending_audit_disposition(self):
        requests, ledger, calls, rows = self.run_case()
        phases = [call['phase'] for call in calls]
        self.assertEqual(phases[0], 'describe')
        validation = [c for c in calls if c['phase'] == 'validation']
        self.assertEqual(len(validation), 15)
        self.assertEqual({(c['pattern'], c['seed']) for c in validation},
                         {(p, s) for p in admission.PATTERNS for s in admission.SEEDS})
        self.assertTrue(all(c['name'] == requests[0]['candidate_id'] and
                            c['env'] == candidate.request_environment(requests[0]['request'])
                            for c in validation))
        self.assertFalse(any('timing' in phase for phase in phases))
        dispositions = [r for r in rows if r.get('record_type') == 'candidate_disposition']
        self.assertEqual([r['status'] for r in dispositions], ['checks_passed_pending_audit'])
        self.assertEqual(dispositions[0]['candidate_id'], requests[0]['candidate_id'])
        self.assertEqual(len(ledger.finished), len(calls))
        self.assertTrue(all(expected == 40 for _, expected in ledger.started))

    def test_unrepresentable_default_is_recorded_without_fabricated_replay(self):
        _, _, calls, rows = self.run_case(unsupported_identity=True)
        self.assertEqual(sum(c['name'] == 'original' for c in calls), 1)
        self.assertEqual(sum(c['name'] == 'expanded_default' for c in calls), 1)
        self.assertFalse(any(c['name'] == 'identity' or c['phase'] == 'validation' for c in calls))
        dispositions = [r for r in rows if r.get('record_type') == 'candidate_disposition']
        self.assertEqual([r['status'] for r in dispositions], ['unsupported_identity'])

    def test_describe_and_execute_fingerprints_must_match(self):
        snapshot = {'canonical_descriptor': {'a': 1}, 'source_sha256': 'a' * 64,
                    'source_bytes': 10, 'compiler_input': {'options': ['option']}}
        desc = {'precompile_kernels': [copy.deepcopy(snapshot)]}
        executed = copy.deepcopy(desc)
        admission.compare_plan(desc, executed)
        for field, value in [('canonical_descriptor', {'a': 2}), ('source_sha256', 'b' * 64),
                             ('source_bytes', 11), ('compiler_input', {'options': ['other']})]:
            bad = copy.deepcopy(executed); bad['precompile_kernels'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                admission.compare_plan(desc, bad)

    def test_failure_is_recorded_then_session_stops_not_counted_admitted(self):
        for phase, scheduler_failure in (('describe', True), ('validation', False)):
            with self.subTest(phase=phase):
                requests, _, calls, rows = self.run_case(failed_phase=phase, status_error=scheduler_failure,
                                                         num_items=2)
                dispositions = [r for r in rows if r.get('record_type') == 'candidate_disposition']
                self.assertTrue(any(r['candidate_id'] == requests[0]['candidate_id'] and
                                    r['status'] == 'candidate_failed' for r in dispositions))
                self.assertFalse(any(r['status'] == 'checks_passed_pending_audit' for r in dispositions))
                self.assertFalse(any(c['name'] == requests[1]['candidate_id'] for c in calls))
                self.assertTrue(any(r.get('record_type') == 'execution' and r.get('phase') == phase for r in rows))

    def test_budget_denial_marks_partial_and_unstarted_without_pass(self):
        for phase, first_status in (('describe', 'not_started'), ('validation', 'partial')):
            with self.subTest(phase=phase):
                requests, ledger, calls, rows = self.run_case(deny_phase=phase, num_items=2)
                dispositions = [r for r in rows if r.get('record_type') == 'candidate_disposition']
                statuses = {r['candidate_id']: r['status'] for r in dispositions}
                self.assertEqual(statuses[requests[0]['candidate_id']], first_status)
                self.assertEqual(statuses[requests[1]['candidate_id']], 'not_started')
                self.assertNotIn('checks_passed_pending_audit', statuses.values())
                self.assertFalse(any(c['name'] == requests[1]['candidate_id'] for c in calls))
                self.assertEqual(len(ledger.finished), len(calls))

    def test_last_moment_launch_denial_finishes_ledger_and_preserves_coverage(self):
        for phase, first_status in (('describe', 'not_started'), ('validation', 'partial')):
            with self.subTest(phase=phase):
                requests, ledger, _, rows = self.run_case(deny_launch_phase=phase, num_items=2)
                self.assertEqual(ledger.finished[-1], 'budget_denied_before_launch')
                self.assertTrue(any(r.get('record_type') == 'launch_denied' and r.get('phase') == phase for r in rows))
                statuses = {r['candidate_id']: r['status'] for r in rows
                            if r.get('record_type') == 'candidate_disposition'}
                self.assertEqual(statuses[requests[0]['candidate_id']], first_status)
                self.assertEqual(statuses[requests[1]['candidate_id']], 'not_started')


if __name__ == '__main__':
    unittest.main()
