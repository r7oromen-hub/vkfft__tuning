"""GPU-free, fail-closed evidence loading and process-level report statistics."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from scipy.stats import t
from study_common import canonical, read_jsonl, block_quality
from guard_audit import read_guard, resolve_log

STAGES = ('health','semantics','equivalence','cost_smoke','screen','confirmation','cost','headline')


def selection_eligible(row):
    """Success is necessary but cannot override an exact-cell preflight exclusion."""
    return row.get('status') == 'ok' and row.get('selection_eligible',True) is not False


def geometric(values):
    values = list(values)
    return math.exp(statistics.mean(map(math.log, values))) if values else None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(rows, reference, target, rounds):
    by = collections.defaultdict(list)
    for row in rows:
        if selection_eligible(row):
            by[row['round'], row['candidate_id']].append(row)
    ratios, hashes = [], []
    for k in range(rounds):
        a, b = by[k, reference], by[k, target]
        if len(a) != 1 or len(b) != 1:
            return None
        x, y = a[0]['result'].get('exec_ms_median'), b[0]['result'].get('exec_ms_median')
        if any(not isinstance(v, (float,int)) or not math.isfinite(v) or v <= 0 for v in (x,y)):
            return None
        if a[0].get('input_seed') != b[0].get('input_seed'):
            return None
        ratios.append(x/y)
        hashes.append([a[0]['record_sha256'], b[0]['record_sha256']])
    return dict(reference=reference,target=target,ratios=ratios,speedup=geometric(ratios),
                record_hashes=hashes,unit='paired independent process medians',
                selected_default_by_definition=reference == target)


def headline_stat(ratios, rounds=10):
    if len(ratios) != rounds or any(not math.isfinite(v) or v <= 0 for v in ratios):
        return dict(speedup=None,p=1.,inferential=False,interval95=None,reason='incomplete_or_invalid_pair')
    logs = list(map(math.log, ratios))
    effect = geometric(ratios)
    tol = 16 * sys.float_info.epsilon * max(1., max(map(abs, logs)))
    if max(logs)-min(logs) <= tol:
        return dict(speedup=effect,p=1.,inferential=False,interval95=None,
                    reason='noninferential_quantization_or_zero_variance')
    mean = statistics.mean(logs)
    se = statistics.stdev(logs)/math.sqrt(rounds)
    critical = float(t.ppf(.975, rounds-1))
    return dict(speedup=effect,p=float(t.sf((mean-math.log(1.02))/se,rounds-1)),
                inferential=True,interval95=[math.exp(mean-critical*se),math.exp(mean+critical*se)],
                t_statistic=(mean-math.log(1.02))/se,df=rounds-1,reason=None)


def holm(tests):
    previous = 0.
    for rank, i in enumerate(sorted(range(len(tests)), key=lambda k: tests[k]['p'])):
        row = tests[i]
        previous = max(previous,min(1.,row['p']*(len(tests)-rank)))
        row['holm_p'] = previous
        row['clear_gain'] = bool(row.get('quality_valid') and row['inferential'] and
                                 row['speedup'] >= 1.05 and previous < .05)
    return tests


def guard_check(row, gpu_uuid, root, cache):
    errors = []
    if row.get('guard_clean') is not True or row.get('guard_gpu_uuid') != gpu_uuid:
        return ['missing_clean_guard_linkage']
    try:
        path = resolve_log(row,root)
        if path not in cache:
            # Index once; large retained guard logs must not be reparsed per row.
            sessions = collections.defaultdict(list)
            for event in read_guard(path):
                sessions[event.get('guard_id')].append(event)
            cache[path] = sessions
        events = cache[path].get(row.get('guard_id'),[])
    except (ValueError,OSError,json.JSONDecodeError) as exc:
        return ['guard_log_unavailable:'+str(exc)]
    run_id = row.get('guard_run_id')
    starts = [e for e in events if e.get('event') == 'run_one_start' and (e.get('active_run') or {}).get('guard_run_id') == run_id]
    ends = [e for e in events if e.get('event') == 'run_one_end' and e.get('guard_run_id') == run_id and e.get('clean') is True]
    if not run_id or len(starts) != 1 or len(ends) != 1:
        errors.append('missing_unique_guard_boundaries')
    if not any(e.get('event') == 'guard_start' for e in events) or not any(e.get('event') == 'idle_complete' for e in events):
        errors.append('missing_guard_start_or_quiet_gate')
    samples = [e for e in events if e.get('event') == 'sample' and (e.get('active_run') or {}).get('guard_run_id') == run_id]
    if not {'before','after'} <= {e.get('boundary') for e in samples}:
        errors.append('missing_before_after_samples')
    for e in samples:
        if (e.get('ok') is not True or e.get('foreign_compute_pids') or e.get('violation_reasons') or
            e.get('monitor_error') is not None or e.get('gpu_uuid') != gpu_uuid or
            e.get('gpu',{}).get('memory_used_mib',float('inf')) > 2048):
            errors.append('unclean_guard_sample')
            break
    return errors


def read_evidence(root, design):
    """Only explicit root/stage files; no recursive merging or pilot fallback."""
    root = Path(root)
    accepted, ledger, errors, gates, coverage = {}, [], [], {}, {}
    cache = {}
    identity_ok = True
    manifest_path = root/'manifest.json'
    if not manifest_path.exists():
        identity_ok = False
        errors.append(dict(file='manifest.json',error='missing_runtime_identity'))
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('study_id') != design['study_id'] or manifest.get('design_sha256') != design['_sha256']:
            identity_ok = False
            errors.append(dict(file='manifest.json',error='study_or_design_identity_mismatch'))
    choices = {}
    for stage, name in (('confirmation','screen_selection.json'),('headline','headline_selection.json')):
        path = root/name
        if path.is_file():
            value = json.loads(path.read_text(encoding='utf-8'))
            entries = value.get('workloads',{})
            choices[stage] = ({wid:entry['candidates'] for wid,entry in entries.items()} if isinstance(entries,dict)
                              else {entry['workload_id']:entry['candidates'] for entry in entries})
    for stage in STAGES:
        base = root/stage
        gate_path = base/'gate.json'
        gates[stage] = json.loads(gate_path.read_text(encoding='utf-8')) if gate_path.exists() else None
        status_path = base/'status.json'
        stage_status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        try:
            rows = read_jsonl(base/'measurements.jsonl')
        except ValueError as exc:
            rows = []
            errors.append(dict(stage=stage,error=str(exc)))
        by_hash, valid = {}, {}
        for row in rows:
            rh = row.get('record_sha256')
            reasons = []
            if row.get('study_id') != design['study_id']:
                reasons.append('record_study_identity_mismatch')
            if not rh or hashlib.sha256(canonical({k:v for k,v in row.items() if k != 'record_sha256'})).hexdigest() != rh:
                reasons.append('record_hash_mismatch')
            if rh in by_hash:
                reasons.append('duplicate_record_hash')
            reasons.extend(guard_check(row,design['scope']['device_uuid'],root,cache))
            if not identity_ok:
                reasons.append('runtime_identity_not_verified')
            by_hash[rh] = row
            valid[rh] = not reasons
            ledger.append(dict(stage=stage,phase=row.get('phase'),workload_id=row.get('workload_id'),
                               candidate_id=row.get('candidate_id'),round=row.get('round'),
                               block_attempt=row.get('block_attempt',0),status=row.get('status'),
                               record_sha256=rh,audit_ok=not reasons,audit_errors=reasons,
                               selection_eligible=selection_eligible(row),
                               preflight_ineligible=row.get('preflight_ineligible'),
                               candidate_validity_class=row.get('candidate_validity_class'),
                               failure_phase=row.get('result',{}).get('failure_phase'),
                               error=row.get('result',{}).get('error'),
                               validation_errors=row.get('validation_errors',[])))
            if reasons:
                errors.append(dict(stage=stage,record_sha256=rh,errors=reasons))
        selected = []
        cost_selection = None
        cost_admission = None
        cost_selection_state = None
        if stage in ('screen','confirmation','headline'):
            seen = set()
            for path in sorted((base/'blocks').glob('*.json')):
                block = json.loads(path.read_text(encoding='utf-8'))
                if block.get('quality',{}).get('ok') is not True:
                    continue
                hashes = block.get('record_hashes',[])
                if not hashes or len(set(hashes)) != len(hashes) or not all(h in by_hash and valid[h] for h in hashes):
                    errors.append(dict(file=str(path),error='unverifiable_accepted_block'))
                    continue
                rs = [by_hash[h] for h in hashes]
                keys = {(r.get('phase'),r.get('workload_id'),r.get('round')) for r in rs}
                cids = [r.get('candidate_id') for r in rs]
                if len(keys) != 1 or len(set(cids)) != len(cids) or any(k in seen for k in keys):
                    errors.append(dict(file=str(path),error='duplicate_or_mixed_accepted_block'))
                    continue
                phase, wid, round_no = next(iter(keys))
                required = ({c['id'] for c in design['candidates']} if stage == 'screen'
                            else set(choices.get(stage,{}).get(wid,[])))
                rounds = {'screen':1,'confirmation':3,'headline':10}[stage]
                attempts = {r.get('block_attempt',0) for r in rs}
                if (phase != stage or set(cids) != required or not required or round_no not in range(rounds)
                        or len(attempts) != 1 or block.get('block_attempt',0) != next(iter(attempts))
                        or block.get('candidate_ids') != cids):
                    errors.append(dict(file=str(path),error='incomplete_or_mismatched_frozen_block'))
                    continue
                try:
                    quality = block_quality(rs)
                except (KeyError,TypeError,ValueError) as exc:
                    quality = dict(ok=False,errors=[str(exc)])
                if not quality['ok']:
                    errors.append(dict(file=str(path),error='recomputed_block_quality_failed',quality=quality))
                    continue
                seen.update(keys)
                selected.extend(rs)
        elif stage == 'cost':
            # Raw cost rows are an append-only engineering ledger. In A6 this
            # includes an orphan from the interrupted pair; success alone must
            # never choose the statistical observations or pick the latest row.
            if (base/'accepted_pairs.json').is_file():
                try:
                    from cost_recovery import selected_cost_rows
                    chosen,cost_selection,cost_admission = selected_cost_rows(root,require_complete=False,require_sessions=True)
                    if not all(valid.get(r.get('record_sha256')) for r in chosen):
                        raise ValueError('explicit_cost_selection_contains_unaudited_record')
                    selected = chosen
                    cost_selection_state = 'explicit_hash_selected_pairs'
                    if gates[stage] and gates[stage].get('accepted_pair_selection_sha256') != digest(base/'accepted_pairs.json'):
                        raise ValueError('cost_gate_pair_selection_hash_mismatch')
                except (OSError,KeyError,ValueError,TypeError) as exc:
                    selected = []
                    cost_selection_state = 'unverifiable_explicit_cost_selection'
                    errors.append(dict(stage=stage,error=str(exc)))
            else:
                cost_selection_state = 'missing_explicit_accepted_pairs'
                if gates[stage] and gates[stage].get('passed'):
                    errors.append(dict(stage=stage,error='completed_cost_gate_missing_explicit_accepted_pairs'))
        else:
            selected = [r for r in rows if valid.get(r.get('record_sha256'))]
        selected_hashes = {r.get('record_sha256') for r in selected}
        for item in ledger:
            if item['stage'] == stage:
                item['included_in_stage_analysis'] = item['record_sha256'] in selected_hashes
                item['included_in_estimators'] = (stage in ('screen','confirmation','cost','headline') and
                                                   item['included_in_stage_analysis'] and item['selection_eligible'])
                if stage == 'cost' and not item['included_in_stage_analysis']:
                    item['engineering_only_reason'] = ('A6_original_unpaired_orphan' if cost_admission and item['record_sha256'] in cost_admission.get('engineering_only_record_hashes',[])
                                                       else 'not_selected_by_explicit_accepted_pairs')
        accepted[stage] = selected
        gate = gates[stage] or {}
        expected = stage_status.get('expected_processes')
        stage_rows = [r for r in rows if r.get('phase') == stage] if stage == 'screen' else rows
        coverage[stage] = dict(raw_records=len(rows),audited_records=sum(valid.values()),
                               accepted_records=len(selected),gate_passed=bool(gate.get('passed')),
                               state=stage_status.get('state','not_started'),expected_processes=expected,
                               stage_attempts=len(stage_rows),unique_attempted_cells=len({(r['phase'],r['workload_id'],r['candidate_id'],r['round']) for r in stage_rows}),
                               successful_records=sum(r.get('status') == 'ok' for r in selected),
                               unsuccessful_records=sum(r.get('status') != 'ok' for r in selected),
                               eligible_records=sum(selection_eligible(r) for r in selected),
                               protocol_valid=gate.get('protocol_valid',bool(gate.get('passed'))),
                               all_candidates_passed=gate.get('all_candidates_passed',bool(gate.get('passed')) and bool(selected) and all(r.get('status') == 'ok' for r in selected)),
                               candidate_validation_failures=gate.get('candidate_validation_failures',[]))
        if stage == 'cost':
            coverage[stage].update(selection_state=cost_selection_state,
                accepted_pairs=len(selected)//2,declared_pairs=len((cost_selection or {}).get('pairs',[])),
                engineering_only_raw_records=len(rows)-len(selected),
                engineering_only_record_hashes=[r['record_sha256'] for r in rows if r.get('record_sha256') not in selected_hashes],
                physical_processes_estimate=(cost_selection or {}).get('physical_processes_estimate'),
                quarantine_records_outside_main_ledger=1 if cost_admission else 0,
                quarantined_guard_run_id=(cost_admission or {}).get('quarantined_guard_run_id'),
                admission_sha256=digest(root/'cost_recovery_admission.json') if cost_admission else None,
                A6_recovery_admitted=bool(cost_admission),
                old_closed_interval_session_exception_record_hashes=(cost_admission or {}).get('preserved_record_hashes',[]))
    return dict(rows=accepted,ledger=ledger,errors=errors,gates=gates,coverage=coverage,identity_ok=identity_ok)
