#!/usr/bin/env python3
"""One explicit same-device recovery after the frozen monitoring interruption."""
import argparse
import collections
import json
import math
import os
from pathlib import Path

import cloud_reproduction as cloud
from cloud_reproduction import CloudController, Stop, build_schedule, immutable, json_file
from guard_audit import audit_row, completed_session, read_guard, resolve_log
from run_suite import run_one
from study_common import (append_json, block_quality, candidate_config, canonical, execution_errors,
                          input_seed, read_jsonl, sha, utc, write_json)


def requested_config(config):
    names = dict(aim_threads='aimThreads',register_boost='registerBoost',use_lut='useLUT',
                 register_boost_nonpow2='registerBoostNonPow2',coalesced_memory='coalescedMemory',
                 use_lut_4step='useLUT_4step',swap_to_2stage='swapTo2Stage4Step',swap_to_3stage='swapTo3Stage4Step')
    value = {'makeForwardPlanOnly':1}
    value.update({names[k]:v for k,v in config.items()})
    return value


def validate_root_layout(here,old_root,new_root,admission):
    here,old_root,new_root = Path(here).resolve(),Path(old_root).resolve(),Path(new_root).resolve()
    bases = [base for base in (here,here.parents[1])
             if old_root==(base/admission['source']['root']).resolve() and new_root==(base/admission['output']['root']).resolve()]
    if (len(bases)!=1 or old_root==new_root or old_root.is_relative_to(new_root) or new_root.is_relative_to(old_root)):
        raise Stop('recovery_root_mismatch')
    return bases[0]


def check_recovery_history(output,stage,current_guard_id):
    guard_dir = Path(output)/'guards'
    if not guard_dir.exists(): return
    for path in guard_dir.iterdir():
        if not path.is_file(): continue
        for event in read_guard(path):
            if event.get('event')!='guard_start' or Path(event.get('script','')).name!='cloud_recovery.py': continue
            argv = event.get('script_args',[])
            stages = [argv[i+1] for i,x in enumerate(argv[:-1]) if x=='--stage'] + [x.split('=',1)[1] for x in argv if x.startswith('--stage=')]
            if len(stages)!=1: raise Stop('recovery_guard_history_stage_unidentifiable')
            if stages[0]==stage and event.get('guard_id')!=current_guard_id:
                raise Stop('prior_recovery_stage_attempt_forbidden:'+stage)


def remaining_schedule(design, admission):
    all_blocks = build_schedule(design)
    old_keys = {(b['workload_id'],b['round']) for b in admission['source']['preserved_blocks']}
    remaining = [b for b in all_blocks if (b['workload_id'],b['round']) not in old_keys]
    if len(old_keys)!=34 or len(remaining)!=31 or sum(len(b['candidate_ids']) for b in remaining)!=100:
        raise Stop('recovery_frozen_schedule_budget_mismatch')
    if remaining[0]['workload_id']!='n16384_b2048' or remaining[0]['round']!=3:
        raise Stop('recovery_interrupted_block_not_first')
    expected = [dict(**b,block_attempt=1 if (b['workload_id'],b['round'])==('n16384_b2048',3) else 0) for b in remaining]
    if expected!=admission['execution']['new_blocks']:
        raise Stop('recovery_schedule_differs_from_admission')
    return remaining


def audit_interrupted(root,design,admission):
    """Permit exactly 99 frozen rows from 34 closed blocks before first bad sample."""
    root = Path(root)
    spec = admission['source']
    rows = read_jsonl(root/'execution'/'measurements.jsonl')
    if len(rows)!=101:
        raise Stop('interrupted_raw_record_count_mismatch')
    byhash = {r.get('record_sha256'):r for r in rows}
    if len(byhash)!=101 or None in byhash:
        raise Stop('interrupted_duplicate_record_hash')
    events = read_guard(root/'guards'/'execution.jsonl')
    failed = [(i,e) for i,e in enumerate(events) if e.get('event')=='sample' and e.get('ok') is not True]
    incidents = [(i,e) for i,e in enumerate(events) if e.get('event')=='incident']
    aborts = [e for e in events if e.get('event')=='guard_abort']
    quarantines = [e for e in events if e.get('event')=='quarantined_run']
    if (len(failed)!=1 or len(incidents)!=1 or len(aborts)!=1 or len(quarantines)!=1 or
        failed[0][0]>=incidents[0][0] or failed[0][1].get('monitor_error','').find('TimeoutExpired')<0 or
        incidents[0][1].get('reasons')!=['monitor_query_failed'] or
        failed[0][0]!=spec['incident_sample_event_index'] or sha(canonical(failed[0][1]))!=spec['incident_sample_canonical_sha256'] or
        sha(canonical(quarantines[0].get('quarantined_row'))) != spec['quarantined_row_canonical_sha256'] or
        quarantines[0].get('quarantined_row',{}).get('returncode')!=spec['quarantined_returncode']):
        raise Stop('interrupted_event_classification_mismatch')
    cutoff = failed[0][0]
    guard_id = spec['guard_id']
    starts = [(i,e) for i,e in enumerate(events) if e.get('event')=='guard_start' and e.get('guard_id')==guard_id]
    idles = [(i,e) for i,e in enumerate(events) if e.get('event')=='idle_complete' and e.get('guard_id')==guard_id]
    if (len(starts)!=1 or len(idles)!=1 or starts[0][0]>=idles[0][0] or idles[0][0]>=cutoff or
        starts[0][1].get('idle_seconds',0)<30 or idles[0][1].get('observed_idle_seconds',0)<30 or
        starts[0][1].get('gpu_uuid')!=design['scope']['device_uuid']):
        raise Stop('interrupted_guard_quiet_gate_invalid')
    parent_pid = starts[0][1].get('parent_pid')
    if (failed[0][1].get('guard_id')!=guard_id or
        (quarantines[0].get('active_run') or {}).get('guard_run_id')!=spec['quarantined_guard_run_id']):
        raise Stop('interrupted_guard_or_quarantine_mismatch')
    if any(e.get('event')=='guard_complete' for e in events if e.get('guard_id')==guard_id):
        raise Stop('interrupted_guard_incorrectly_completed')
    if (sum(e.get('event')=='run_one_start' for e in events)!=102 or sum(e.get('event')=='run_one_end' and e.get('clean') is True for e in events)!=101):
        raise Stop('interrupted_physical_process_count_mismatch')
    before = events[:cutoff]
    if any(e.get('event') in ('incident','guard_abort') or (e.get('event')=='sample' and e.get('ok') is not True) for e in before):
        raise Stop('interrupted_pre_cutoff_monitor_failure')
    expected_blocks = build_schedule(design)[:34]
    frozen_blocks = spec['preserved_blocks']
    if len(frozen_blocks)!=34:
        raise Stop('interrupted_frozen_block_count_mismatch')
    repeats = json_file(root/'graph_repeats.json')
    candidates = {c['id']:c for c in design['candidates']}
    workloads = {w['id']:w for w in design['workloads']}
    selected = []
    for expected, block in zip(expected_blocks,frozen_blocks):
        wid,k = expected['workload_id'],expected['round']
        if (block['workload_id'],block['round'],block['candidate_ids'])!=(wid,k,expected['candidate_ids']):
            raise Stop('interrupted_block_order_or_membership_mismatch')
        path = root/'execution'/'blocks'/f'execution__{wid}__r{k}.json'
        meta = json_file(path)
        expected_relative = Path(spec['root'])/'execution'/'blocks'/f'execution__{wid}__r{k}.json'
        if (Path(block['block_path'])!=expected_relative or sha(path.read_bytes())!=block['block_sha256'] or
            meta.get('record_hashes')!=block['record_hashes'] or meta.get('candidate_ids')!=expected['candidate_ids']):
            raise Stop('interrupted_block_metadata_mismatch')
        hashes = block['record_hashes']
        if len(set(hashes))!=len(hashes) or not all(h in byhash for h in hashes):
            raise Stop('interrupted_block_hashes_invalid')
        group = [byhash[h] for h in hashes]
        for pos,row in enumerate(group):
            cid = expected['candidate_ids'][pos]
            run_id = row.get('guard_run_id')
            starts = [(i,e) for i,e in enumerate(before) if e.get('event')=='run_one_start' and (e.get('active_run') or {}).get('guard_run_id')==run_id]
            ends = [(i,e) for i,e in enumerate(before) if e.get('event')=='run_one_end' and e.get('guard_run_id')==run_id and e.get('clean') is True]
            samples = [(i,e) for i,e in enumerate(before) if e.get('event')=='sample' and (e.get('active_run') or {}).get('guard_run_id')==run_id]
            original = workloads[wid]
            if (not run_id or len(starts)!=1 or len(ends)!=1 or starts[0][0]>=ends[0][0] or
                not {'before','after'} <= {s.get('boundary') for i,s in samples if starts[0][0]<i<ends[0][0]} or
                (starts[0][1].get('active_run') or {}).get('candidate_id')!=cid or
                (starts[0][1].get('active_run') or {}).get('workload_id')!=wid or
                (starts[0][1].get('active_run') or {}).get('round')!=k or
                ends[0][1].get('candidate_id')!=cid or ends[0][1].get('workload_id')!=wid or
                any(s.get('ok') is not True or s.get('foreign_compute_pids') or s.get('violation_reasons') or s.get('monitor_error') is not None or
                    s.get('gpu_uuid')!=design['scope']['device_uuid'] or not isinstance((s.get('gpu') or {}).get('memory_used_mib'),(int,float)) or
                    not math.isfinite((s.get('gpu') or {})['memory_used_mib']) or (s.get('gpu') or {})['memory_used_mib']>2048 or
                    any(app.get('own_harness') is not True or app.get('ppid')!=parent_pid or app.get('start_ticks') is None or
                        app.get('exe')!=(starts[0][1].get('active_run') or {}).get('expected_exe') for app in s.get('compute_apps',[])) or
                    not isinstance(s.get('query_wall_seconds'),(int,float)) or not math.isfinite(s['query_wall_seconds']) or s['query_wall_seconds']>2 for i,s in samples) or
                row.get('guard_id')!=guard_id or row.get('guard_gpu_uuid')!=design['scope']['device_uuid'] or row.get('guard_clean') is not True or
                row.get('study_id')!=design['study_id'] or row.get('phase')!='execution' or row.get('workload_id')!=wid or
                row.get('manifest_sha256')!=spec['manifest_sha256'] or
                row.get('round')!=k or row.get('candidate_id')!=cid or row.get('position_in_block')!=pos or row.get('block_attempt',0)!=0 or
                row.get('input_seed')!=input_seed(design,'execution',wid,k) or row.get('config')!=candidate_config(candidates[cid]) or
                row.get('result',{}).get('requested_config')!=requested_config(candidate_config(candidates[cid])) or
                row.get('status')!='ok' or sha(canonical({x:y for x,y in row.items() if x!='record_sha256'}))!=row.get('record_sha256') or
                any(row['workload'].get(x)!=original[x] for x in ('id','n','batch','input_bytes')) or
                row['workload'].get('graph_repeats')!=repeats[wid]['graph_repeats'] or
                row.get('result',{}).get('seed')!=row['input_seed'] or row.get('result',{}).get('gpu',{}).get('uuid')!=design['scope']['device_uuid'] or
                execution_errors(row.get('result',{}),row['workload'],101)):
                raise Stop('interrupted_closed_row_invalid:'+wid+':'+cid)
        if not block_quality(group)['ok']:
            raise Stop('interrupted_whole_block_quality_failed:'+wid)
        selected.extend(group)
    selected_hashes = [r['record_sha256'] for r in selected]
    expected_hashes = [h for b in frozen_blocks for h in b['record_hashes']]
    if len(selected)!=99 or len(set(selected_hashes))!=99 or selected_hashes!=expected_hashes or expected_hashes!=spec['record_hashes']:
        raise Stop('interrupted_selected_hashes_mismatch')
    orphan = [r['record_sha256'] for r in rows if r['record_sha256'] not in set(selected_hashes)]
    if orphan!=spec['orphan_record_hashes']:
        raise Stop('interrupted_orphan_hashes_mismatch')
    return dict(rows=selected,blocks=frozen_blocks,guard_id=guard_id,
                cutoff_event_index=cutoff,orphan_record_hashes=orphan,
                quarantined_guard_run_id=spec['quarantined_guard_run_id'])


def preflight_validate(args):
    """Read-only admission check; never starts a guard or writes a result file."""
    here = Path(__file__).parent.resolve()
    admission,freeze = json_file(args.admission),json_file(args.recovery_freeze)
    if (sha(args.recovery_freeze.read_bytes())!='e5f4f9013889db853f6a64ad0d6ce89d5fdab964f8e2bc4fcefa913fb04071e1' or
        freeze.get('recovery_id')!=admission.get('recovery_id') or freeze.get('admission_sha256')!=sha(args.admission.read_bytes())):
        raise Stop('recovery_admission_freeze_mismatch')
    old_root,new_root = Path(args.original).resolve(),Path(args.output).resolve()
    base = validate_root_layout(here,old_root,new_root,admission)
    for item in admission['source']['files']:
        source = (base/item['path']).resolve()
        if not source.is_relative_to(old_root) or not source.is_file() or sha(source.read_bytes())!=item['sha256']:
            raise Stop('recovery_original_source_hash_mismatch:'+item['path'])
    old_guard = (here/'gpu_guard.py').read_bytes()
    new_guard = (here/'gpu_guard_cloud_recovery.py').read_bytes()
    expected = old_guard.replace(
        b'if script.name not in ("run_suite.py", "controlled_recheck.py", "cloud_reproduction.py"):',
        b'if script.name not in ("run_suite.py", "controlled_recheck.py", "cloud_reproduction.py", "cloud_recovery.py"):').replace(
        b'parser.error("--script must be run_suite.py, controlled_recheck.py or cloud_reproduction.py")',
        b'parser.error("--script must be run_suite.py, controlled_recheck.py, cloud_reproduction.py or cloud_recovery.py")')
    if new_guard!=expected or sha(old_guard)!=admission['guards']['original_guard_sha256'] or sha(new_guard)!=admission['guards']['recovery_guard_sha256']:
        raise Stop('recovery_guard_contains_non_allowlist_changes')
    design = json_file(args.design)
    cloud.validate_design(design)
    original_manifest = json_file(old_root/'manifest.json')
    if (sha(args.design.read_bytes())!=admission['primary_design_sha256'] or design['study_id']!=admission['study_id'] or
        sha(args.freeze.read_bytes())!=sha((old_root/'freeze.json').read_bytes()) or
        cloud.runtime_manifest(args,design)!=original_manifest or sha((old_root/'manifest.json').read_bytes())!=admission['source']['manifest_sha256'] or
        sha((old_root/'graph_repeats.json').read_bytes())!=admission['source']['graph_repeats_sha256']):
        raise Stop('original_runtime_identity_changed')
    reviewer = object.__new__(CloudController)
    reviewer.design,reviewer.output = design,old_root
    for stage in ('equivalence','semantics','calibration','health'):
        reviewer.require(stage)
    preserved = audit_interrupted(old_root,design,admission)
    remaining = remaining_schedule(design,admission)
    for stage in ('health','execution','cost'):
        if (new_root/stage/'status.json').exists(): raise Stop('recovery_stage_already_attempted:'+stage)
    guard_dir = new_root/'guards'
    if guard_dir.exists():
        for path in guard_dir.iterdir():
            if path.is_file() and any(e.get('event')=='guard_start' and Path(e.get('script','')).name=='cloud_recovery.py' for e in read_guard(path)):
                raise Stop('recovery_guard_attempt_already_recorded')
    return dict(status='preflight_passed_no_guard_no_gpu',recovery_id=admission['recovery_id'],
                preserved_records=len(preserved['rows']),remaining_blocks=len(remaining),
                remaining_execution_processes=sum(len(b['candidate_ids']) for b in remaining),
                new_health=30,cost=48,total_physical_max=369,
                controller_sha256=sha(Path(__file__).read_bytes()),guard_sha256=sha(new_guard))


class RecoveryController(CloudController):
    def __init__(self,args):
        here = Path(__file__).parent.resolve()
        self.admission = json_file(args.admission)
        freeze = json_file(args.recovery_freeze)
        if (sha(args.recovery_freeze.read_bytes())!='e5f4f9013889db853f6a64ad0d6ce89d5fdab964f8e2bc4fcefa913fb04071e1' or
            freeze.get('recovery_id')!=self.admission.get('recovery_id') or freeze.get('admission_sha256')!=sha(args.admission.read_bytes())):
            raise Stop('recovery_admission_freeze_mismatch')
        self.original = Path(args.original)
        old_root,new_root = self.original.resolve(),Path(args.output).resolve()
        # Original deployment uses BASE/results; local audit checkout uses SCRIPT_DIR/results.
        base = validate_root_layout(here,old_root,new_root,self.admission)
        for item in self.admission['source']['files']:
            source = (base/item['path']).resolve()
            if not source.is_relative_to(old_root) or not source.is_file() or sha(source.read_bytes())!=item['sha256']:
                raise Stop('recovery_original_source_hash_mismatch:'+item['path'])
        original_guard = Path(__file__).with_name('gpu_guard.py').read_bytes()
        recovery_guard = Path(__file__).with_name('gpu_guard_cloud_recovery.py').read_bytes()
        expected_guard = original_guard.replace(
            b'if script.name not in ("run_suite.py", "controlled_recheck.py", "cloud_reproduction.py"):',
            b'if script.name not in ("run_suite.py", "controlled_recheck.py", "cloud_reproduction.py", "cloud_recovery.py"):').replace(
            b'parser.error("--script must be run_suite.py, controlled_recheck.py or cloud_reproduction.py")',
            b'parser.error("--script must be run_suite.py, controlled_recheck.py, cloud_reproduction.py or cloud_recovery.py")')
        if recovery_guard!=expected_guard or sha(recovery_guard)!=self.admission['guards']['recovery_guard_sha256'] or sha(original_guard)!=self.admission['guards']['original_guard_sha256']:
            raise Stop('recovery_guard_contains_non_allowlist_changes')
        cmdline = Path('/proc/self/cmdline').read_bytes().split(b'\0')
        if not any(Path(os.fsdecode(x)).name=='gpu_guard_cloud_recovery.py' for x in cmdline if x):
            raise Stop('recovery_guard_entrypoint_required')
        self.original_manifest = json_file(self.original/'manifest.json')
        if sha((self.original/'manifest.json').read_bytes())!=self.admission['source']['manifest_sha256']:
            raise Stop('original_manifest_hash_changed')
        if sha((self.original/'graph_repeats.json').read_bytes())!=self.admission['source']['graph_repeats_sha256']:
            raise Stop('original_calibration_hash_changed')
        super().__init__(args)
        if self.manifest!=self.original_manifest or self.design['study_id']!=self.admission['study_id'] or sha(args.design.read_bytes())!=self.admission['primary_design_sha256']:
            raise Stop('original_runtime_identity_changed')
        # Require the original completed preflight gates without rewriting their files.
        reviewer = object.__new__(CloudController)
        reviewer.design,reviewer.output = self.design,self.original
        for stage in ('equivalence','semantics','calibration','health'):
            reviewer.require(stage)
        self.preserved = audit_interrupted(self.original,self.design,self.admission)
        immutable(self.output/'graph_repeats.json',(self.original/'graph_repeats.json').read_bytes())
        immutable(self.output/'recovery_admission.json',args.admission.read_bytes())
        immutable(self.output/'recovery_freeze.json',args.recovery_freeze.read_bytes())
        self.recovery_manifest = dict(schema_version='cloud-recovery-runtime/v1',recovery_id=self.admission['recovery_id'],
            original_manifest_sha256=sha((self.original/'manifest.json').read_bytes()),
            original_execution_ledger_sha256=sha((self.original/'execution'/'measurements.jsonl').read_bytes()),
            original_guard_sha256=sha((self.original/'guards'/'execution.jsonl').read_bytes()),
            graph_repeats_sha256=sha((self.original/'graph_repeats.json').read_bytes()),
            admission_sha256=sha(args.admission.read_bytes()),freeze_sha256=sha(args.recovery_freeze.read_bytes()),
            recovery_controller_sha256=sha(Path(__file__).read_bytes()),
            recovery_guard_sha256=sha(Path(__file__).with_name('gpu_guard_cloud_recovery.py').read_bytes()),
            device_uuid=self.design['scope']['device_uuid'],new_budget=dict(health=30,execution=100,cost=48))
        immutable(self.output/'recovery_manifest.json',json.dumps(self.recovery_manifest,sort_keys=True,indent=2).encode()+b'\n')
        check_recovery_history(self.output,self.stage,self.guard_id)

    def attempt(self,w,cid,phase,round_no):
        cap = {'health':30,'execution':100,'cost':48}[self.stage]
        if len(self.rows)>=cap:
            raise Stop('recovery_stage_process_budget_exhausted:'+self.stage)
        return super().attempt(w,cid,phase,round_no)

    def require_recovery_execution(self):
        stage = 'execution'
        gate = json_file(self.output/stage/'gate.json')
        rows = read_jsonl(self.output/stage/'measurements.jsonl')
        if gate.get('passed') is not True or len(rows)!=100 or gate.get('record_count')!=100:
            raise Stop('recovery_execution_gate_or_count_invalid')
        sessions = {(r['guard_id'],r['guard_log']) for r in rows}
        if sessions != {(s['guard_id'],s['guard_log']) for s in gate.get('guard_sessions',[])} or not sessions:
            raise Stop('recovery_execution_guard_set_invalid')
        for gid,log in sessions:
            if not completed_session(resolve_log({'guard_log':log},self.output),gid):
                raise Stop('recovery_execution_guard_incomplete')
        byhash = {r['record_sha256']:r for r in rows}
        if len(byhash)!=100: raise Stop('recovery_execution_duplicate_hash')
        expected_rows = []
        repeats = json_file(self.output/'graph_repeats.json')
        for b in remaining_schedule(self.design,self.admission):
            wid,k = b['workload_id'],b['round']
            path = self.output/stage/'blocks'/f'execution__{wid}__r{k}.json'
            meta = json_file(path)
            hashes = meta.get('record_hashes',[])
            if len(set(hashes))!=len(hashes) or not all(h in byhash for h in hashes): raise Stop('recovery_block_hash_invalid')
            group = [byhash[h] for h in hashes]
            attempt = 1 if (wid,k)==('n16384_b2048',3) else 0
            if (meta.get('candidate_ids')!=b['candidate_ids'] or meta.get('block_attempt')!=attempt or
                [r['candidate_id'] for r in group]!=b['candidate_ids'] or not block_quality(group)['ok']):
                raise Stop('recovery_block_quality_invalid')
            for pos,row in enumerate(group):
                w = row['workload']
                if (row.get('study_id')!=self.design['study_id'] or row.get('manifest_sha256')!=sha((self.output/'manifest.json').read_bytes()) or
                    row.get('phase')!='execution' or row.get('workload_id')!=wid or row.get('round')!=k or row.get('position_in_block')!=pos or
                    row.get('recovery_manifest_sha256')!=sha((self.output/'recovery_manifest.json').read_bytes()) or
                    row.get('block_attempt')!=attempt or row.get('input_seed')!=input_seed(self.design,'execution',wid,k) or
                    any(w.get(x)!=self.workloads[wid][x] for x in ('id','n','batch','input_bytes')) or
                    w.get('samples')!=101 or w.get('graph_repeats')!=repeats[wid]['graph_repeats'] or row.get('status')!='ok' or
                    row.get('config')!=candidate_config(self.candidates[row['candidate_id']]) or
                    row.get('result',{}).get('requested_config')!=requested_config(candidate_config(self.candidates[row['candidate_id']])) or
                    row.get('result',{}).get('seed')!=row['input_seed'] or row.get('result',{}).get('gpu',{}).get('uuid')!=self.design['scope']['device_uuid'] or
                    execution_errors(row.get('result',{}),w,101) or
                    sha(canonical({x:y for x,y in row.items() if x!='record_sha256'}))!=row['record_sha256'] or
                    not audit_row(row,self.design['scope']['device_uuid'],self.output)['ok']):
                    raise Stop('recovery_execution_row_invalid:'+wid)
            expected_rows.extend(group)
        if len(expected_rows)!=100 or {r['record_sha256'] for r in expected_rows}!=set(byhash):
            raise Stop('recovery_execution_coverage_mismatch')
        path = self.output/'execution_selection.json'
        if gate.get('selection_sha256')!=sha(path.read_bytes()) or json_file(path)!=self.make_selection():
            raise Stop('recovery_selection_manifest_mismatch')

    def block(self,w,phase,round_no,ids,repeats,samples):
        block_id = f"{phase}/{w['id']}/r{round_no}"
        attempt = 1 if phase=='execution' and w['id']=='n16384_b2048' and round_no==3 else 0
        rows = []
        for pos,cid in enumerate(ids):
            item = dict(w,graph_repeats=repeats,samples=samples,block_id=block_id,position_in_block=pos,block_attempt=attempt)
            rows.append(self.attempt(item,cid,phase,round_no))
        quality = block_quality(rows)
        meta = dict(block_id=block_id,candidate_ids=ids,block_attempt=attempt,
                    record_hashes=[r['record_sha256'] for r in rows],quality=quality)
        write_json(self.stage_path/'blocks'/(block_id.replace('/','__')+'.json'),meta)
        if not quality['ok']:
            raise Stop('whole_block_quality_failed:'+block_id+':'+ ';'.join(quality['errors']))
        return quality

    def execution(self):
        frozen = json_file(self.output/'graph_repeats.json')
        blocks = remaining_schedule(self.design,self.admission)
        for b in blocks:
            w = self.workloads[b['workload_id']]
            self.block(w,'execution',b['round'],b['candidate_ids'],frozen[w['id']]['graph_repeats'],101)
        selection = self.make_selection()
        write_json(self.output/'execution_selection.json',selection)
        self.gate(block_count=len(blocks),selected_old=99,selected_new=100,
                  selection_sha256=sha((self.output/'execution_selection.json').read_bytes()),
                  graph_repeats_sha256=sha((self.output/'graph_repeats.json').read_bytes()))

    def make_selection(self):
        old = {(b['workload_id'],b['round']):b for b in self.admission['source']['preserved_blocks']}
        oldrows = {r['record_sha256']:r for r in self.preserved['rows']}
        newrows = {r['record_sha256']:r for r in read_jsonl(self.output/'execution'/'measurements.jsonl')}
        new_digest = sha((self.output/'execution'/'measurements.jsonl').read_bytes())
        old_digest = sha((self.original/'execution'/'measurements.jsonl').read_bytes())
        blocks = []
        for b in build_schedule(self.design):
            key = (b['workload_id'],b['round'])
            if key in old:
                source,hashes,byhash,digest = 'original',old[key]['record_hashes'],oldrows,old_digest
                attempt = 0
            else:
                source,byhash,digest = 'recovery',newrows,new_digest
                path = self.output/'execution'/'blocks'/f"execution__{b['workload_id']}__r{b['round']}.json"
                meta = json_file(path)
                if meta['candidate_ids']!=b['candidate_ids'] or not meta['quality']['ok']:
                    raise Stop('recovery_selection_block_invalid')
                hashes = meta['record_hashes']
                attempt = meta['block_attempt']
            if not all(h in byhash for h in hashes): raise Stop('recovery_selection_source_record_missing')
            quality = block_quality([byhash[h] for h in hashes])
            if not quality['ok']: raise Stop('recovery_selection_recomputed_quality_failed')
            blocks.append(dict(workload_id=b['workload_id'],round=b['round'],block_attempt=attempt,candidate_ids=b['candidate_ids'],
                               source_root=source,source_measurements_sha256=digest,record_hashes=hashes,quality=quality))
        hashes = [h for b in blocks for h in b['record_hashes']]
        if len(blocks)!=65 or len(hashes)!=199 or len(set(hashes))!=199 or set(hashes)&set(self.preserved['orphan_record_hashes']):
            raise Stop('recovery_selection_coverage_invalid')
        return dict(schema_version='cloud-recovery-execution-selection/v1',recovery_id=self.admission['recovery_id'],
                    study_id=self.design['study_id'],admission_sha256=sha(self.args.admission.read_bytes()),blocks=blocks,
                    selected_record_count=199,selected_old_records=99,selected_new_records=100,
                    excluded_orphan_record_hashes=self.preserved['orphan_record_hashes'],
                    excluded_quarantined_guard_run_id=self.preserved['quarantined_guard_run_id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--recovery-freeze',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--design',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exe',required=True)
    p.add_argument('--plain-exe',required=True)
    p.add_argument('--build-provenance',type=Path,required=True)
    p.add_argument('--stage',choices=['health','execution','cost'])
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--preflight-only',action='store_true',help='Read-only CPU admission and runtime verification; no guard or GPU process.')
    args = p.parse_args()
    if not 0<args.timeout<=180: p.error('--timeout must be between 1 and 180 seconds')
    if args.preflight_only:
        print(json.dumps(preflight_validate(args),indent=2))
        return
    if not args.stage: p.error('--stage is required unless --preflight-only is given')
    # gpu_guard replaces this module's run_one; inherited attempt resolves through cloud.
    cloud.run_one = run_one
    controller = RecoveryController(args)
    recovery_manifest_sha256 = sha((controller.output/'recovery_manifest.json').read_bytes())
    def bound_run_one(*call_args):
        row = run_one(*call_args)
        row['recovery_manifest_sha256'] = recovery_manifest_sha256
        return row
    cloud.run_one = bound_run_one
    try:
        if args.stage in ('execution','cost'):
            controller.require('health')
            if any(row.get('recovery_manifest_sha256')!=recovery_manifest_sha256 for row in read_jsonl(controller.output/'health'/'measurements.jsonl')):
                raise Stop('recovery_health_manifest_binding_mismatch')
        if args.stage=='cost':
            controller.require_recovery_execution()
        controller.status('running')
        getattr(controller,args.stage)()
        controller.status('complete')
    except BaseException as exc:
        controller.status('stopped',reason=f'{type(exc).__name__}:{exc}')
        raise


if __name__=='__main__':
    main()
