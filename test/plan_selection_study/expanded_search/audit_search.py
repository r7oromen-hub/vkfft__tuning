#!/usr/bin/env python3
"""Independent integrity, coverage and arithmetic audit of a finished search."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import audit_guard
import audit_admission
import audit_screen
import audit_space


def read(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def numerical(row, request, gpu):
    """Check full successful candidate validation; caller verifies guarded run linkage."""
    try:
        w, r = row['workload'], row['result']
        env = {'VKFFT_EXPANDED_RADIX': ','.join(map(str, request['radix'])),
               'VKFFT_EXPANDED_REGS': str(request['logical_registers']),
               'VKFFT_EXPANDED_BATCH': str(request['grouped_batch'])}
        if (row['direct_environment'] != env or row['returncode'] != 0 or row['status'] != 'ok' or r['status'] != 'ok' or
                (r['n'],r['batch'],r['seed'],r['pattern']) != (w['n'],w['batch'],w['seed'],w['pattern']) or
                r.get('expanded_requested_radix') != env['VKFFT_EXPANDED_RADIX'] or
                r.get('expanded_requested_regs') != env['VKFFT_EXPANDED_REGS'] or
                r.get('expanded_requested_batch') != env['VKFFT_EXPANDED_BATCH'] or
                r['gpu']['uuid'] != gpu or r['gpu']['cuda_visible_devices'] != gpu or r['header_instrumentation'] is not True or
                r['requested_config'] != {'makeForwardPlanOnly': 1} or r['single_forward_input_unchanged'] is not True or
                r['input_unchanged'] is not True or r['validation_output_initialization'] != '0xff_bytes_fp32_nan_before_single_and_final_timed_forward'):
            return False
        k = r['structural_features']['kernels'][0]; p = r['precompile_kernels'][0]; d = p['raw_descriptor']
        local, swap = k['localSize'], d['axisSwapped']; lim = r['expanded_resource_validation']
        if (r['structural_features']['numAxisUploads'] != 1 or len(r['structural_features']['kernels']) != 1 or
                r['descriptor_validation']['all_match'] is not True or p['compiler_options_match'] is not True or
                k['stageRadix'] != request['radix'] or k['groupedBatch'] != request['grouped_batch'] or
                d['logical_registers_per_thread'] != request['logical_registers'] or d['registerBoost'] != 1 or swap not in (0,1) or
                len(local) != 3 or local[2] != 1 or local[1 if swap else 0] != w['n']//request['logical_registers'] or
                local[0 if swap else 1] != request['grouped_batch'] or
                not 0 <= lim['dynamic_shared_bytes'] <= lim['function_dynamic_limit_bytes'] or lim['static_shared_bytes'] < 0 or
                lim['dynamic_shared_bytes'] + lim['static_shared_bytes'] > lim['device_shared_limit_bytes'] or
                lim['threads'] != math.prod(local) or lim['threads'] > lim['function_thread_limit']):
            return False
        reference = ('independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity' if w['pattern'] == 'random'
                     else 'analytic_pattern_reference_with_independent_CPU_FFT_crosscheck')
        for name in ('single_forward_correctness', 'correctness'):
            c = r[name]
            if not (c['passed'] is True and c['nonfinite_count'] == 0 and c['failing_batches'] == 0 and
                    c['checked_elements'] == w['n']*w['batch'] and c['criterion'] == 'criterion_v2' and c['reference'] == reference and
                    math.isclose(c['l2_scaled_error_threshold'], 1e-5, rel_tol=1e-12) and
                    math.isclose(c['linf_input_scaled_error_threshold'], 1e-4, rel_tol=1e-12)):
                return False
        return True
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return False


def original_numerical(row, gpu):
    try:
        w, r = row['workload'], row['result']
        if (row.get('direct_environment') or row.get('returncode') != 0 or row.get('status') != 'ok' or r['status'] != 'ok' or
                (r['n'],r['batch'],r['seed'],r['pattern']) != (w['n'],w['batch'],w['seed'],w['pattern']) or
                r['gpu']['uuid'] != gpu or r['gpu']['cuda_visible_devices'] != gpu or r['header_instrumentation'] is not False or
                r['single_forward_input_unchanged'] is not True or r['input_unchanged'] is not True or
                r['validation_output_initialization'] != '0xff_bytes_fp32_nan_before_single_and_final_timed_forward' or
                len(r['postinit_source_kernels']) != 1 or r['init_trace']['available'] is not False):
            return False
        ref = ('independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity' if w['pattern']=='random'
               else 'analytic_pattern_reference_with_independent_CPU_FFT_crosscheck')
        for name in ('single_forward_correctness','correctness'):
            c = r[name]
            if not (c['passed'] is True and c['nonfinite_count'] == 0 and c['failing_batches'] == 0 and
                    c['checked_elements'] == w['n']*w['batch'] and c['criterion'] == 'criterion_v2' and c['reference'] == ref and
                    math.isclose(c['l2_scaled_error_threshold'],1e-5,rel_tol=1e-12) and
                    math.isclose(c['linf_input_scaled_error_threshold'],1e-4,rel_tol=1e-12)):
                return False
        return True
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def source_signature(result, original):
    p = (result['postinit_source_kernels'] if original else result['precompile_kernels'])[0]
    return {'structural_features':result['structural_features'], 'compiled_diagnostics':result['compiled_diagnostics'],
            'resolved_config':result['resolved_config'], 'canonical_descriptor':p['canonical_descriptor'],
            'source_sha256':p['source_sha256'], 'source_bytes':p['source_bytes'],
            'compiler_input':p['expected_compiler_input'] if original else p['compiler_input'],
            'actual_compiler_options':p['actual_compiler_options']}


def audit(rows, guard, incidents, manifest_dir):
    errors = []
    heads = [x for x in rows if x.get('record_type') == 'search_protocol']
    terminals = [x for x in rows if x.get('record_type') == 'search_complete']
    if len(heads) != 1 or len(terminals) != 1:
        return {'status': 'fail', 'errors': ['missing or duplicate search protocol/terminal']}
    h, t = heads[0], terminals[0]; protocol = h['protocol']; gpu = protocol['gpu_uuid']
    guard_audit = audit_guard.audit(rows, guard, incidents, gpu)
    screen_audit = audit_screen.audit(rows)
    errors.extend('guard: '+e for e in guard_audit['errors'])
    errors.extend('screen: '+e for e in screen_audit['errors'])
    for row in (x for x in rows if x.get('record_type') == 'execution'):
        binary = 'original' if row.get('candidate_id') == 'original' else 'expanded'
        try:
            parsed = [json.loads(x) for x in row['stdout'].splitlines() if x.strip()]
            if parsed != [row['result']] or row['executable_sha256'] != h['sha256'][binary]:
                raise ValueError('stdout or executable hash differs')
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{row.get('guard_run_id')}: {exc}")
    if t.get('status') not in ('budget_exhausted', 'completed_selected'):
        errors.append('search did not terminate normally or at budget boundary')
    expected = {(n,b) for b in protocol['batches'] for n in protocol['lengths']}
    terminal_statuses = t.get('workload_statuses', [])
    if len(terminal_statuses) != len(expected) or {(x['n'],x['batch']) for x in terminal_statuses} != expected:
        errors.append('terminal workload list is missing, duplicated or unregistered')
    selected = {}; counts = Counter(); manifests = {}
    workload_records = {(x['n'],x['batch']):x for x in rows if x.get('record_type') == 'workload_status'}
    incomplete_workloads = []
    for item in terminal_statuses:
        key=(item['n'],item['batch']); state=item.get('status')
        if state in ('not_started','identity_pending_or_incomplete'):
            counts['terminal_'+state] += 1
            if key in workload_records:
                errors.append(f'{key}: incomplete terminal conflicts with completed workload record')
            partial=Path(manifest_dir)/f'n{key[0]}_b{key[1]}.jsonl'
            incomplete_workloads.append({'n':key[0],'batch':key[1],'status':state,
                                         'partial_manifest_present':partial.is_file(),
                                         'partial_manifest_sha256':sha(partial) if partial.is_file() else None})
        elif state in ('identity_passed','unsupported_identity'):
            counts['terminal_'+state] += 1
            if workload_records.get(key,{}).get('status') != state:
                errors.append(f'{key}: terminal disagrees with workload record')
        else:
            errors.append(f'{key}: unknown terminal workload status {state!r}')
    if t.get('status') == 'completed_selected' and incomplete_workloads:
        errors.append('completed search contains incomplete workloads')
    for status in (x for x in rows if x.get('record_type') == 'workload_status'):
        key = (status['n'], status['batch'])
        if key not in expected or key in manifests:
            errors.append(f'{key}: unregistered or duplicate workload status'); continue
        if status.get('status') == 'unsupported_identity':
            controls = [r for r in rows if r.get('record_type') == 'execution' and r.get('phase') == 'identity_control' and
                        (r.get('workload',{}).get('n'),r.get('workload',{}).get('batch')) == key]
            try:
                defaults = [r for r in controls if r.get('candidate_id') == 'expanded_default' and r.get('returncode') == 0]
                originals = [r for r in controls if r.get('candidate_id') == 'original' and original_numerical(r,gpu)]
                if len(defaults) != 1 or len(originals) != 1 or audit_admission.default_identity(defaults[0]['result']) is not None:
                    raise ValueError('unsupported identity lacks matching control evidence')
            except (KeyError, TypeError, IndexError, ValueError) as exc:
                errors.append(f'{key}: {exc}')
            manifests[key] = None; counts['unsupported_identity'] += 1; continue
        if status.get('status') != 'identity_passed':
            errors.append(f'{key}: unknown identity status'); continue
        controls = [r for r in rows if r.get('record_type') == 'execution' and r.get('phase') == 'identity_control' and
                    (r.get('workload',{}).get('n'),r.get('workload',{}).get('batch')) == key]
        try:
            original = next(r for r in controls if r.get('candidate_id') == 'original')
            default = next(r for r in controls if r.get('candidate_id') == 'expanded_default')
            replay = next(r for r in controls if r.get('candidate_id') == 'identity')
            if (len(controls) != 3 or default.get('returncode') != 0 or default.get('status') != 'ok' or
                    default.get('result',{}).get('status') != 'ok' or not original_numerical(original,gpu) or not numerical(replay,status['identity'],gpu)):
                raise ValueError('missing or invalid three-way identity controls')
            dr,rr,orig=default['result'],replay['result'],original['result']
            if audit_admission.default_identity(dr) != status['identity'] or source_signature(dr,False) != source_signature(rr,False):
                raise ValueError('identity not representable or changed generated plan')
            dp,op=dr['precompile_kernels'][0],orig['postinit_source_kernels'][0]
            dc,oc=dict(dr['resolved_config']),dict(orig['resolved_config'])
            if (dc.pop('keepShaderCode',None) != 0 or oc.pop('keepShaderCode',None) != 1 or dc != oc or
                    dr['structural_features'] != orig['structural_features'] or dr['compiled_diagnostics'] != orig['compiled_diagnostics'] or
                    dp['source_sha256'] != op['source_sha256'] or dp['source_bytes'] != op['source_bytes'] or
                    dp['canonical_descriptor'] != op['canonical_descriptor'] or
                    op['expected_compiler_input']['options'] != dp['actual_compiler_options'] or
                    op['expected_compiler_input']['original_compile_header_sha256'] != dp['compiler_input']['original_compile_header_sha256']):
                raise ValueError('original and expanded defaults differ')
        except (StopIteration, KeyError, TypeError, IndexError, ValueError) as exc:
            errors.append(f'{key}: {exc}')
        path = Path(manifest_dir)/f'n{key[0]}_b{key[1]}.jsonl'
        try:
            if sha(path) != status['manifest_sha256']:
                raise ValueError('manifest file digest mismatch')
            doc = read(path)
            ma = audit_space.audit_document(doc, key[0], key[1], status['identity'], protocol['selection_seed'],
                                            protocol['max_nondefault_selected_requests_per_workload'])
            if ma['status'] != 'pass':
                raise ValueError('manifest content/selection audit failed: '+str(ma['errors'][:2]))
            manifests[key] = ma['counts']
            for row in doc:
                if row.get('selection') == 'selected':
                    selected[(key[0],key[1],row['candidate_id'])] = row
            counts['manifest_candidates_selected'] += ma['counts'].get('selected', 0)
            counts['manifest_static_rejected'] += ma['counts'].get('rejected_static', 0)
            counts['manifest_not_selected'] += ma['counts'].get('not_selected', 0)
            del doc  # release this potentially large manifest before loading the next one
        except (OSError, KeyError, ValueError, TypeError) as exc:
            errors.append(f'{key}: {exc}')
    selected_events, start_ids, finished, blocks = set(), set(), set(), defaultdict(list)
    current = None
    for row in rows:
        if row.get('record_type') == 'candidate_selected':
            key = (row['n'],row['batch'],row['candidate_id'])
            manifest_row = selected.get(key, {})
            if (key in selected_events or key not in selected or manifest_row.get('request') != row.get('request') or
                    manifest_row.get('selection_index') != row.get('selection_index') or
                    manifest_row.get('planned_fft_threads') != row.get('planned_fft_threads') or current is not None):
                errors.append(f'{key}: selected event missing/duplicated/not in manifest')
            selected_events.add(key); current = key
        elif row.get('record_type') == 'candidate_start':
            key = (row['n'],row['batch'],row['candidate_id'])
            describes = [r for r in blocks[key] if r.get('phase') == 'candidate_describe' and r.get('candidate_id') == key[2]]
            if key != current or key in start_ids or len(describes) != 1 or row.get('guard_run_id') != describes[0].get('guard_run_id'):
                errors.append(f'{key}: candidate_start does not match describe attempt')
            start_ids.add(key)
        elif row.get('record_type') == 'candidate_disposition':
            key = (row['n'],row['batch'],row['candidate_id'])
            if key != current or key in finished:
                errors.append(f'{key}: disposition not linked to unique candidate start')
            finished.add(key); counts['disposition_'+str(row.get('status'))] += 1; current = None
        elif row.get('record_type') == 'execution' and current is not None:
            blocks[current].append(row)
    counts['selected_not_started'] = len(set(selected)-start_ids)
    counts['started_not_disposed'] = len(start_ids-finished)
    for d in (x for x in rows if x.get('record_type') == 'candidate_disposition' and x.get('status') == 'screen_complete'):
        key = (d['n'],d['batch'],d['candidate_id']); selected_row = selected.get(key)
        if selected_row is None:
            errors.append(f'{key}: screen_complete not selected'); continue
        req = selected_row['request']
        candidate_validation = [x for x in blocks[key] if x.get('phase') == 'candidate_validation' and x.get('candidate_id') == key[2]]
        original_validation = [x for x in blocks[key] if x.get('phase') == 'candidate_validation' and x.get('candidate_id') == 'original']
        want = {(p,s) for p in ('zero','impulse','constant','quarter-tone','random') for s in protocol['validation_seeds']}
        observed = Counter((x['workload']['pattern'],x['workload']['seed']) for x in candidate_validation)
        original_observed = Counter((x['workload']['pattern'],x['workload']['seed']) for x in original_validation)
        if (set(observed) != want or any(x != 1 for x in observed.values()) or set(original_observed) != want or
                any(x != 1 for x in original_observed.values()) or not all(numerical(x,req,gpu) for x in candidate_validation) or
                not all(original_numerical(x,gpu) for x in original_validation)):
            errors.append(f'{key}: candidate validation evidence incomplete or invalid')
        validation_sequence = [x for x in blocks[key] if x.get('phase') == 'candidate_validation']
        if len(validation_sequence) == 30:
            for i in range(0,30,2):
                left,right=validation_sequence[i:i+2]
                pattern, seed = left['workload']['pattern'], left['workload']['seed']
                payload=json.dumps([protocol['selection_seed'],key[0],key[1],key[2],
                                    'candidate_validation',0,pattern,seed],separators=(',',':')).encode()
                expected_order=['original','candidate']
                random.Random(int.from_bytes(hashlib.sha256(payload).digest(),'big')).shuffle(expected_order)
                actual_order=['original' if x.get('candidate_id') == 'original' else 'candidate' for x in (left,right)]
                if ({left.get('candidate_id'),right.get('candidate_id')} != {'original',key[2]} or
                        (left['workload']['pattern'],left['workload']['seed']) != (right['workload']['pattern'],right['workload']['seed']) or
                        [left.get('order'),right.get('order')] != [0,1] or actual_order != expected_order):
                    errors.append(f'{key}: numerical validation is not paired'); break
        describes = [x for x in blocks[key] if x.get('phase') == 'candidate_describe' and x.get('candidate_id') == key[2] and x.get('returncode') == 0]
        try:
            if len(describes) != 1:
                raise ValueError('missing or duplicate describe')
            dp = describes[0]['result']['precompile_kernels'][0]
            for row in candidate_validation:
                ep = row['result']['precompile_kernels'][0]
                if any(dp[field] != ep[field] for field in ('canonical_descriptor','source_sha256','source_bytes','compiler_input')):
                    raise ValueError('describe/validation source or descriptor mismatch')
            timing = [x for x in blocks[key] if x.get('phase') == 'screen_timing']
            for row in timing:
                is_original = row['candidate_id'] == 'original'
                if not (original_numerical(row,gpu) if is_original else numerical(row,req,gpu)):
                    raise ValueError('screen execution numerical/resource evidence invalid')
                expected_signature = d['original_signature' if is_original else 'candidate_signature']
                if source_signature(row['result'], is_original) != expected_signature:
                    raise ValueError('screen source/plan/resource signature mismatch')
            for row in original_validation:
                if source_signature(row['result'], True) != d['original_signature']:
                    raise ValueError('validation original signature differs from screening')
        except (KeyError, IndexError, ValueError) as exc:
            errors.append(f'{key}: {exc}')
        counts['fully_validated_screened'] += 1
    return {'status': 'fail' if errors else 'pass', 'errors': errors, 'counts': dict(counts),
            'screen': screen_audit, 'workloads_with_manifest': len([x for x in manifests.values() if x is not None]),
            'incomplete_workloads': incomplete_workloads,
            'scope': 'search log/guard/manifest and validated screening, not an independent confirmation or speedup claim; budget ledger audited separately'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=Path,required=True);p.add_argument('--guard',type=Path,required=True)
    p.add_argument('--incidents',type=Path,required=True);p.add_argument('--manifests',type=Path,required=True)
    a=p.parse_args();result=audit(read(a.rows),read(a.guard),read(a.incidents),a.manifests)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
