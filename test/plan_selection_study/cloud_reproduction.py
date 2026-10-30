#!/usr/bin/env python3
"""Independent cloud cohort. All GPU calls must be wrapped by gpu_guard.py."""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys

from cost_audit import audit_pair
from guard_audit import audit_row, completed_session, read_guard, resolve_log
from run_suite import run_one
from study_common import (TUNING_FIELDS, append_json, block_candidates, block_quality,
                          candidate_config, canonical, input_seed, read_jsonl, sha, utc,
                          workload_order, write_json)


class Stop(RuntimeError):
    pass


def cost_classification(audit):
    """An observed describe mismatch is evidence; protocol corruption is fatal."""
    errors = audit.get('errors', [])
    if not errors:
        return 'equivalent' if audit.get('ok') is True else 'fatal'
    if audit.get('ok') is not False:
        return 'fatal'
    if all(e.startswith('describe_full_inequivalent:') or
           e == 'selected_materialization_changed_plan_or_source' for e in errors):
        return 'negative'
    return 'fatal'


def build_schedule(design):
    """Return predeclared execution blocks in seeded order, without reselection."""
    specs = {x['workload_id']: x for x in design['execution']['workloads']}
    blocks = []
    for round_no in range(max(x['rounds'] for x in specs.values())):
        for w in workload_order(design, 'execution', round_no):
            wid = w['id']
            if wid not in specs or round_no >= specs[wid]['rounds']:
                continue
            ids = specs[wid]['candidates']
            blocks.append(dict(workload_id=wid, round=round_no,
                               candidate_ids=block_candidates(design, wid, 'execution', round_no, ids)))
    return blocks


def validate_design(d):
    ids = [c['id'] for c in d['candidates']]
    ws = {w['id']: w for w in d['workloads']}
    if len(ids) != 26 or len(set(ids)) != 26 or len(ws) != len(d['workloads']):
        raise Stop('invalid_candidate_or_workload_catalog')
    specs = {x['workload_id']: x for x in d['execution']['workloads']}
    selection = d['execution']['frozen_winners']
    costs = d['cost_profile']['workloads']
    if len(specs) != 10 or len(costs) != 8:
        raise Stop('invalid_workload_budget')
    if set(selection) != set(specs) or not set(costs) <= set(specs):
        raise Stop('invalid_selection_or_cost_domain')
    if not set(specs) <= set(ws) or not set(selection.values()) <= set(ids):
        raise Stop('unknown_workload_or_candidate')
    if any(not set(s['candidates']) <= set(ids) or len(s['candidates']) != len(set(s['candidates'])) or
           not {'default','default_replica',selection[wid]} <= set(s['candidates']) or
           s['winner'] != selection[wid] or s['rounds'] not in (3,10) for wid,s in specs.items()):
        raise Stop('unknown_extra_candidate')
    if d['cost_profile']['selected_candidates'] != {wid:selection[wid] for wid in costs}:
        raise Stop('cost_selection_mismatch')
    for wid, w in ws.items():
        if wid != f"n{w['n']}_b{w['batch']}" or w['input_bytes'] != 8*w['n']*w['batch'] or w['input_bytes'] > 268435456:
            raise Stop('invalid_workload:' + wid)
    for cid in ('default', 'default_replica'):
        c = next(x for x in d['candidates'] if x['id'] == cid)
        if any(c.get(k, 0) != 0 for k in TUNING_FIELDS):
            raise Stop('default_control_request_changed')
    if any(c.get('disable_reorder_four_step', 0) for c in d['candidates']):
        raise Stop('natural_order_required')
    if sum(len(b['candidate_ids']) for b in build_schedule(d)) != 199:
        raise Stop('execution_budget_mismatch')
    if len(costs)*3*2 != 48:
        raise Stop('cost_budget_mismatch')
    semantic = d['preflight']['semantic_probes']
    if sum(len(g['candidates']) for g in semantic['groups'])*len(semantic['patterns']) != 28:
        raise Stop('semantic_budget_mismatch')
    if sum(len(c['candidates'])*2 for c in d['preflight']['equivalence']['cases']) != 8:
        raise Stop('equivalence_budget_mismatch')
    if len(d['health']['workloads'])*5*2 != 30:
        raise Stop('health_budget_mismatch')
    if len(d['calibration']['workloads']) != 11 or {w['id'] for w in d['calibration']['workloads']} != set(specs) | {w['id'] for w in d['health']['workloads']}:
        raise Stop('calibration_budget_mismatch')
    return True


def json_file(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise Stop('immutable_file_changed:' + str(path))
    else:
        path.write_bytes(data)


def runtime_manifest(args, design):
    here = Path(__file__).parent
    provenance = json_file(args.build_provenance)
    artifacts = provenance.get('artifacts',{})
    if (provenance.get('state')!='complete' or provenance.get('upstream_commit')!=design['source']['vkfft_commit'] or
        provenance.get('cuda_root')!='/usr/local/cuda-12.8' or provenance.get('actual_cuda_versions',{}).get('nvrtc') != [12,8] or
        provenance.get('actual_cuda_versions',{}).get('runtime') != 12080 or
        not provenance.get('original_header_files') or not provenance.get('generated_header_provenance',{}).get('sha256') or
        not provenance.get('harness_source_sha256')):
        raise Stop('build_provenance_incomplete_or_wrong_toolchain')
    for name,path in [('vkfft_instrumented_benchmark',args.exe),('vkfft_uninstrumented_control',args.plain_exe)]:
        if artifacts.get(name,{}).get('sha256') != sha(Path(path).read_bytes()):
            raise Stop('build_provenance_binary_mismatch:' + name)
    if not provenance.get('commands') or any(c.get('returncode') != 0 for c in provenance['commands']):
        raise Stop('build_or_reference_check_failed')
    if not any(design['scope']['device_uuid'] in c.get('output','') and '610.57.04' in c.get('output','') for c in provenance['commands']):
        raise Stop('build_device_driver_provenance_missing')
    names = ('cloud_reproduction.py', 'run_suite.py', 'study_common.py', 'gpu_guard.py',
             'guard_audit.py', 'trace_audit.py', 'cost_audit.py', 'selection_analysis.py',
             'candidate_policy.py', 'cost_recovery.py')
    return dict(schema_version='cloud-runtime/v1', study_id=design['study_id'],
                design_sha256=sha(args.design.read_bytes()), gpu_uuid=design['scope']['device_uuid'],
                binary_sha256={role: sha(Path(path).read_bytes()) for role, path in
                               [('instrumented', args.exe), ('plain', args.plain_exe)]},
                binary_paths={role: str(Path(path).resolve()) for role, path in
                              [('instrumented', args.exe), ('plain', args.plain_exe)]},
                controller_files_sha256={name: sha((here/name).read_bytes()) for name in names},
                source=design['source'], seed_namespace=design['measurement']['seed_namespace'],
                python_version=platform.python_version(), host=platform.node(), platform=platform.platform(),
                build_provenance_sha256=sha(args.build_provenance.read_bytes()))


class CloudController:
    def __init__(self, args):
        self.args = args
        self.design = json_file(args.design)
        validate_design(self.design)
        freeze = json_file(args.freeze)
        if freeze.get('study_id') != self.design['study_id'] or freeze.get('design_sha256') != sha(args.design.read_bytes()):
            raise Stop('design_freeze_mismatch')
        args.design_data = self.design
        self.output = Path(args.output)
        self.stage = args.stage
        self.stage_path = self.output/self.stage
        self.workloads = {w['id']: w for w in self.design['calibration']['workloads']}
        self.workloads.update({w['id']: w for w in self.design['workloads']})
        self.candidates = {c['id']: c for c in self.design['candidates']}
        self.manifest = runtime_manifest(args, self.design)
        self.args.expected_executable_sha256 = self.manifest['binary_sha256']['instrumented']
        self.guard_id = os.environ.get('VKFFT_STUDY_GUARD_ID')
        if not self.guard_id or os.environ.get('VKFFT_STUDY_GUARD_UUID') != self.design['scope']['device_uuid'] or not os.environ.get('VKFFT_STUDY_GUARD_LOG'):
            raise Stop('GPU_guard_required')
        starts = [e for e in read_guard(Path(os.environ['VKFFT_STUDY_GUARD_LOG']))
                  if e.get('guard_id') == self.guard_id and e.get('event') == 'guard_start']
        if len(starts) != 1 or starts[0].get('idle_seconds', 0) < 30:
            raise Stop('guard_quiet_period_below_30_seconds')
        guard_dir = (self.output/'guards').resolve()
        log_path = Path(os.environ['VKFFT_STUDY_GUARD_LOG']).resolve()
        if log_path.parent != guard_dir:
            raise Stop('guard_log_must_be_in_cohort_guards_directory')
        for path in guard_dir.iterdir():
            if not path.is_file(): continue
            for event in read_guard(path):
                if event.get('event') != 'guard_start' or Path(event.get('script','')).name != 'cloud_reproduction.py': continue
                argv = event.get('script_args',[])
                stages = [argv[i+1] for i,x in enumerate(argv[:-1]) if x=='--stage'] + [x.split('=',1)[1] for x in argv if x.startswith('--stage=')]
                if len(stages)!=1: raise Stop('guard_history_stage_unidentifiable')
                if stages[0]==self.stage and event.get('guard_id')!=self.guard_id:
                    raise Stop('prior_guard_stage_attempt_forbidden:' + self.stage)
        immutable(self.output/'design.json', args.design.read_bytes())
        immutable(self.output/'freeze.json', args.freeze.read_bytes())
        immutable(self.output/'build_provenance.json', args.build_provenance.read_bytes())
        immutable(self.output/'manifest.json', json.dumps(self.manifest, indent=2, sort_keys=True).encode()+b'\n')
        self.stage_path.mkdir(parents=True, exist_ok=True)
        self.measurements = self.stage_path/'measurements.jsonl'
        self.rows = read_jsonl(self.measurements)
        # An incomplete or failed stage is never silently restarted or prefix-spliced.
        if self.rows or (self.stage_path/'status.json').exists():
            raise Stop('stage_already_attempted_no_automatic_retry')
        self.active = None

    def status(self, state, **extra):
        write_json(self.stage_path/'status.json', dict(study_id=self.design['study_id'], stage=self.stage,
                   state=state, updated_utc=utc(), guard_id=self.guard_id, active=self.active,
                   recorded_processes=len(self.rows), **extra))

    def require(self, stage):
        gate = json_file(self.output/stage/'gate.json')
        if gate.get('passed') is not True:
            raise Stop('prerequisite_failed:' + stage)
        rows = read_jsonl(self.output/stage/'measurements.jsonl')
        if not gate.get('guard_sessions') or gate.get('record_count') != len(rows) or not rows:
            raise Stop('prerequisite_count_or_guard_missing:' + stage)
        if {(r['guard_id'],r['guard_log']) for r in rows} != {(s['guard_id'],s['guard_log']) for s in gate['guard_sessions']}:
            raise Stop('prerequisite_guard_set_mismatch:' + stage)
        for session in gate['guard_sessions']:
            if not completed_session(resolve_log(session, self.output), session['guard_id']):
                raise Stop('prerequisite_guard_incomplete:' + stage)
        for row in rows:
            if (row.get('study_id') != self.design['study_id'] or row.get('manifest_sha256') != sha((self.output/'manifest.json').read_bytes()) or row.get('status') != 'ok' or
                sha(canonical({k:v for k,v in row.items() if k != 'record_sha256'})) != row.get('record_sha256') or not audit_row(row, self.design['scope']['device_uuid'], self.output)['ok']):
                raise Stop('prerequisite_row_provenance_failed:' + stage)
            w = row['workload']
            seed = w.get('input_seed_override',input_seed(self.design,w.get('seed_phase',row['phase']),row['workload_id'],row['round']))
            if row.get('input_seed') != seed:
                raise Stop('prerequisite_seed_mismatch:' + stage)
        self.check_coverage(stage, rows)

    def check_coverage(self, stage, rows):
        observed = collections.Counter((r['phase'],r['workload_id'],r['candidate_id'],r['round']) for r in rows)
        expected = []
        if stage == 'equivalence':
            expected = [('equivalence/'+role,c['id'],cid,0) for c in self.design['preflight']['equivalence']['cases'] for cid in c['candidates'] for role in ('plain','instrumented')]
        elif stage == 'semantics':
            probes = self.design['preflight']['semantic_probes']
            expected = [('semantics/'+pattern,g['id'],cid,0) for g in probes['groups'] for pattern in probes['patterns'] for cid in g['candidates']]
        elif stage == 'calibration':
            frozen = json_file(self.output/'graph_repeats.json')
            if set(frozen) != {w['id'] for w in self.design['calibration']['workloads']}:
                raise Stop('calibration_workload_coverage_invalid')
            for wid,item in frozen.items():
                probes = item['probes']
                if probes not in (1,2,3): raise Stop('calibration_probe_count_invalid')
                expected.extend(('calibration',wid,'default',k) for k in range(probes))
                final = [r for r in rows if r['phase']=='calibration' and r['workload_id']==wid and r['round']==probes-1]
                if len(final)!=1 or final[0]['record_sha256']!=item['source_record_sha256'] or final[0]['workload']['graph_repeats']!=item['graph_repeats'] or final[0]['result']['raw_interval_ms_median']!=item['raw_ms']:
                    raise Stop('calibration_source_invalid')
        elif stage in ('health','execution'):
            blocks = ([dict(workload_id=w['id'], round=k, candidate_ids=block_candidates(self.design,w['id'],'health',k,['default','default_replica']))
                       for k in range(5) for w in self.design['health']['workloads']] if stage=='health' else build_schedule(self.design))
            byhash = {r['record_sha256']:r for r in rows}
            health_ratios = collections.defaultdict(list)
            for b in blocks:
                expected.extend((stage,b['workload_id'],cid,b['round']) for cid in b['candidate_ids'])
                name = f"{stage}__{b['workload_id']}__r{b['round']}.json"
                meta = json_file(self.output/stage/'blocks'/name)
                hashes = meta.get('record_hashes',[])
                if len(set(hashes))!=len(hashes) or not all(h in byhash for h in hashes): raise Stop('block_hash_invalid')
                group = [byhash[h] for h in hashes]
                quality = block_quality(group)
                if (meta.get('candidate_ids')!=b['candidate_ids'] or [r['candidate_id'] for r in group]!=b['candidate_ids'] or
                    any(r['phase']!=stage or r['workload_id']!=b['workload_id'] or r['round']!=b['round'] or r.get('block_attempt',0)!=0 for r in group) or
                    len({r['input_seed'] for r in group})!=1 or not quality['ok']):
                    raise Stop('block_quality_or_membership_invalid')
                if stage == 'health': health_ratios[b['workload_id']].append(quality['default_over_replica'])
            if stage == 'health' and any(not .98 <= math.exp(statistics.mean(map(math.log, values))) <= 1.02 for values in health_ratios.values()):
                raise Stop('health_geometric_mean_outside_2pct')
        elif stage == 'cost':
            expected = [('cost',wid,mode,k) for k in range(3) for wid in self.design['cost_profile']['workloads'] for mode in ('full-search-cost','describe-select-cost')]
        if observed != collections.Counter(expected):
            raise Stop('prerequisite_coverage_invalid:' + stage)

    def attempt(self, w, cid, phase, round_no):
        w = dict(w, diagnostic_capture=True)
        self.active = dict(workload_id=w['id'], candidate_id=cid, phase=phase, round=round_no)
        self.status('running')
        original_exe, original_sha = self.args.exe, self.args.expected_executable_sha256
        if w.get('binary_role') == 'plain':
            self.args.exe = self.args.plain_exe
            self.args.expected_executable_sha256 = self.manifest['binary_sha256']['plain']
        try:
            row = run_one(self.args, w, (cid, candidate_config(self.candidates[cid]) if cid in self.candidates else {}), phase, round_no, len(self.rows))
        finally:
            self.args.exe, self.args.expected_executable_sha256 = original_exe, original_sha
        audit = audit_row(row, self.design['scope']['device_uuid'], self.output)
        if not audit['ok']:
            append_json(self.stage_path/'unaccepted_records.jsonl', dict(row=row, guard_audit=audit))
            raise Stop('guard_provenance_rejected:' + ';'.join(audit['errors']))
        row['guard_audit'] = audit
        row['manifest_sha256'] = sha((self.output/'manifest.json').read_bytes())
        row['record_sha256'] = sha(canonical(row))
        append_json(self.measurements, row)
        self.rows.append(row)
        self.active = None
        self.status('running')
        print(f"{self.stage} {len(self.rows)} {phase} {w['id']} {cid} r{round_no}: {row['status']}", flush=True)
        if row['status'] != 'ok':
            raise Stop('process_failed:' + w['id'] + ':' + cid + ':' + ';'.join(row.get('validation_errors', [])))
        return row

    def gate(self, **extra):
        sessions = sorted({(r['guard_id'], r['guard_log']) for r in self.rows})
        write_json(self.stage_path/'gate.json', dict(passed=True, stage=self.stage, created_utc=utc(),
                   record_count=len(self.rows), guard_sessions=[dict(guard_id=a, guard_log=b) for a,b in sessions], **extra))

    def calibrate(self, workloads):
        path = self.output/'graph_repeats.json'
        frozen = json_file(path) if path.exists() else {}
        cfg = self.design['measurement']['repeats']
        for original in workloads:
            wid = original['id']
            if wid in frozen:
                continue
            repeats = 1
            for probe in range(cfg['max_probes']):
                w = dict(original, graph_repeats=repeats, samples=self.design['measurement']['screen_samples'], block_id=f'calibration/{wid}/{probe}')
                row = self.attempt(w, 'default', 'calibration', probe)
                raw = row['result'].get('raw_interval_ms_median')
                if not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw <= 0:
                    raise Stop('invalid_calibration_raw_interval:' + wid)
                reached = raw >= cfg['min_raw_ms']
                if reached or repeats == cfg['cap'] or probe == cfg['max_probes']-1:
                    frozen[wid] = dict(graph_repeats=repeats, raw_ms=raw, target_met=reached,
                                       capped=repeats == cfg['cap'], probes=probe+1, source_record_sha256=row['record_sha256'])
                    write_json(path, frozen)
                    break
                repeats = min(cfg['cap'], max(repeats+1, math.ceil(1.1*cfg['min_raw_ms']*repeats/raw)))
        return frozen

    def block(self, w, phase, round_no, ids, repeats, samples):
        block_id = f"{phase}/{w['id']}/r{round_no}"
        rows = []
        for pos, cid in enumerate(ids):
            item = dict(w, graph_repeats=repeats, samples=samples, block_id=block_id, position_in_block=pos, block_attempt=0)
            rows.append(self.attempt(item, cid, phase, round_no))
        quality = block_quality(rows)
        record = dict(block_id=block_id, candidate_ids=ids, record_hashes=[r['record_sha256'] for r in rows], quality=quality)
        write_json(self.stage_path/'blocks'/(block_id.replace('/', '__')+'.json'), record)
        if not quality['ok']:
            raise Stop('whole_block_quality_failed:' + block_id + ':' + ';'.join(quality['errors']))
        return quality

    def calibration(self):
        if (self.output/'graph_repeats.json').exists():
            raise Stop('preexisting_calibration_forbidden')
        self.calibrate(self.design['calibration']['workloads'])
        self.gate(graph_repeats_sha256=sha((self.output/'graph_repeats.json').read_bytes()))

    def health(self):
        workloads = self.design['health']['workloads']
        frozen = json_file(self.output/'graph_repeats.json')
        ratios = {w['id']: [] for w in workloads}
        for k in range(5):
            for w in workloads:
                ids = block_candidates(self.design, w['id'], 'health', k, ['default', 'default_replica'])
                quality = self.block(w, 'health', k, ids, frozen[w['id']]['graph_repeats'], self.design['measurement']['confirmation_samples'])
                ratios[w['id']].append(quality['default_over_replica'])
        means = {wid: math.exp(statistics.mean(map(math.log, xs))) for wid, xs in ratios.items()}
        if any(not .98 <= v <= 1.02 for v in means.values()):
            raise Stop('health_geometric_mean_outside_2pct')
        self.gate(ratios=ratios, geometric_means=means, graph_repeats_sha256=sha((self.output/'graph_repeats.json').read_bytes()))

    def equivalence(self):
        comparisons = []
        for case in self.design['preflight']['equivalence']['cases']:
            for cid in case['candidates']:
                rows = {}
                for role in self.design['preflight']['equivalence']['binary_roles']:
                    w = dict(case, graph_repeats=16, samples=1, warmup=0, warmup_ms=0,
                             binary_role=role, seed_phase='equivalence', pattern='random')
                    rows[role] = self.attempt(w, cid, 'equivalence/'+role, 0)
                plain = sorted(rows['plain']['result'].get('postinit_source_kernels', []), key=lambda x:(x['axis'],x['upload']))
                inst = sorted(rows['instrumented']['result'].get('precompile_kernels', []), key=lambda x:(x['axis'],x['upload']))
                errors = []
                if not plain or len(plain) != len(inst):
                    errors.append('kernel_coverage_mismatch')
                for a,b in zip(plain, inst):
                    if a['canonical_descriptor'] != b['canonical_descriptor'] or a['source_sha256'] != b['source_sha256']:
                        errors.append('canonical_or_source_mismatch')
                    options = a['expected_compiler_input'].get('options')
                    if options != b['compiler_input'].get('options') or options != b.get('actual_compiler_options') or a.get('actual_compiler_options') is not None:
                        errors.append('compiler_options_mismatch')
                    h = a['expected_compiler_input'].get('original_compile_header_sha256')
                    if not isinstance(h,str) or len(h) != 64 or h != b['compiler_input'].get('original_compile_header_sha256'):
                        errors.append('original_compile_header_mismatch')
                    if a.get('snapshot_stage') != 'post_initialize_retained_source':
                        errors.append('plain_snapshot_stage_mismatch')
                item = dict(workload_id=case['id'], candidate_id=cid, errors=errors,
                            record_hashes=[rows[x]['record_sha256'] for x in ('plain','instrumented')])
                comparisons.append(item)
                write_json(self.stage_path/'comparisons.json', comparisons)
                if errors:
                    raise Stop('instrumentation_equivalence_failed:' + ';'.join(errors))
        self.gate(comparisons=comparisons, excluded_from_performance_analysis=True)

    def semantics(self):
        for group in self.design['preflight']['semantic_probes']['groups']:
            for pattern in self.design['preflight']['semantic_probes']['patterns']:
                for cid in group['candidates']:
                    w = dict(group, graph_repeats=16, samples=1, warmup=0, warmup_ms=0,
                             pattern=pattern, block_id=f"semantics/{group['id']}/{pattern}")
                    self.attempt(w, cid, 'semantics/'+pattern, 0)
        self.gate(excluded_from_performance_analysis=True)

    def execution(self):
        frozen = json_file(self.output/'graph_repeats.json')
        blocks = build_schedule(self.design)
        for b in blocks:
            w = self.workloads[b['workload_id']]
            self.block(w, 'execution', b['round'], b['candidate_ids'], frozen[w['id']]['graph_repeats'], self.design['measurement']['confirmation_samples'])
        self.gate(block_count=len(blocks), graph_repeats_sha256=sha((self.output/'graph_repeats.json').read_bytes()))

    def candidate_file(self, wid, k):
        order = sorted(self.candidates)
        random.Random(input_seed(self.design, 'cost/candidate_order', wid, k)).shuffle(order)
        columns = ('candidate_id',)+TUNING_FIELDS+('disable_reorder_four_step',)
        lines = ['\t'.join(columns)]
        lines += ['\t'.join([cid]+[str(self.candidates[cid].get(field,0)) for field in columns[1:]]) for cid in order]
        path = self.stage_path/'candidate_orders'/f'{wid}_r{k}.tsv'
        immutable(path, ('\n'.join(lines)+'\n').encode())
        return path, order

    def cost(self):
        pairs = []
        for k in range(3):
            for wid in self.design['cost_profile']['workloads']:
                path, order = self.candidate_file(wid, k)
                selected = self.design['cost_profile']['selected_candidates'][wid]
                modes = ['full-search-cost','describe-select-cost']
                if k % 2:
                    modes.reverse()
                rows = {}
                for mode in modes:
                    w = dict(self.workloads[wid], graph_repeats=1, samples=1, warmup=0, warmup_ms=0,
                             study_mode=mode, candidate_file=str(path.resolve()), selected_id=selected,
                             block_id=f'cost/{wid}/r{k}', block_attempt=0)
                    rows[mode] = self.attempt(w, mode, 'cost', k)
                audit = audit_pair(rows['full-search-cost']['result'], rows['describe-select-cost']['result'], order, selected)
                classification = cost_classification(audit)
                pair = dict(workload_id=wid, round=k, candidate_ids=modes, selected_id=selected,
                            record_hashes=[rows[x]['record_sha256'] for x in modes], quality=audit, classification=classification)
                pairs.append(pair)
                write_json(self.stage_path/'pairs'/f'{wid}_r{k}.json', pair)
                if classification == 'fatal':
                    raise Stop('cost_protocol_failed:' + ';'.join(audit['errors']))
        all_equivalent = all(x['classification']=='equivalent' for x in pairs)
        self.gate(pairs=pairs, protocol_valid=True, all_pairs_equivalent=all_equivalent,
                  describe_feasibility='equivalent' if all_equivalent else 'negative',
                  interpretation='Known frozen choice construction; not a deployed online selector.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exe', required=True)
    p.add_argument('--plain-exe', required=True)
    p.add_argument('--build-provenance', required=True, type=Path)
    p.add_argument('--stage', required=True, choices=['equivalence','semantics','calibration','health','execution','cost'])
    p.add_argument('--timeout', type=int, default=180)
    args = p.parse_args()
    if not 0 < args.timeout <= 180:
        p.error('--timeout must be between 1 and 180 seconds')
    controller = CloudController(args)
    requirements = dict(equivalence=[], semantics=['equivalence'], calibration=['equivalence','semantics'],
                        health=['equivalence','semantics','calibration'], execution=['equivalence','semantics','calibration','health'],
                        cost=['equivalence','semantics','calibration','health','execution'])
    try:
        for stage in requirements[args.stage]:
            controller.require(stage)
        if args.stage in ('health','execution','cost'):
            if sha((controller.output/'graph_repeats.json').read_bytes()) != json_file(controller.output/'calibration'/'gate.json')['graph_repeats_sha256']:
                raise Stop('calibration_freeze_changed')
        controller.status('running')
        getattr(controller, args.stage)()
        controller.status('complete')
    except BaseException as exc:
        controller.status('stopped', reason=f'{type(exc).__name__}:{exc}')
        raise


if __name__ == '__main__':
    main()
