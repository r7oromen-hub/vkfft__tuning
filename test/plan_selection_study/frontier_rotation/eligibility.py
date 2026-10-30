"""Re-audit hot confirmation evidence and identify eligible rotation requests."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CONFIRM = HERE.parent / 'frontier_confirmation'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_original_path = sys.path[:]
try:
    sys.path.insert(0, str(CONFIRM))
    evidence = load('frontier_rotation_confirmation_evidence', CONFIRM / 'evidence.py')
    audit = load('frontier_rotation_confirmation_audit', CONFIRM / 'audit.py')
finally:
    sys.path[:] = _original_path
frontier = evidence.frontier
search = evidence.search_controller


def need(condition, message):
    if not condition:
        raise ValueError(message)


def segment(rows, key):
    selected = []
    active = False
    found = False
    for row in rows:
        if row.get('record_type') == 'candidate_start':
            this_key = (row['n'], row['batch'], row['candidate_id'])
            if active:
                raise ValueError('confirmation segment was not closed')
            if this_key == key:
                need(not found, 'duplicate confirmation candidate segment')
                active = found = True
        if active:
            selected.append(row)
            if row.get('record_type') == 'candidate_disposition':
                need((row['n'], row['batch'], row['candidate_id']) == key, 'confirmation segment disposition mismatch')
                active = False
    need(found and not active, 'missing or incomplete confirmation candidate segment')
    return selected


def pool(bytes_per_buffer, l2_bytes, hot_repeats):
    """Same integer limits as the CUDA rotation_pool helper; no GPU claim."""
    if (not all(type(x) is int and x > 0 for x in (bytes_per_buffer, l2_bytes, hot_repeats)) or
        hot_repeats & (hot_repeats-1) or hot_repeats > 2048):
        raise ValueError('invalid workload, L2, or hot graph repetitions')
    slots = 2
    while slots <= 2048 and slots * bytes_per_buffer <= l2_bytes:
        slots *= 2
    if slots > 2048 or 2 * slots * bytes_per_buffer > 1280 * 1024 * 1024:
        return None
    repeats = max(64, hot_repeats, slots)
    if repeats > 2048 or repeats & (repeats-1) or repeats % slots:
        return None
    return {'slots': slots, 'total_bytes': 2 * slots * bytes_per_buffer,
            'graph_repeats': repeats, 'l2_bytes': l2_bytes}


def verify(plan_path, screen_paths, first_paths, second_paths, binaries, gpu, confirmation_commit):
    """Recompute screen and both sessions from raw immutable inputs on every call."""
    names = {'rows', 'guard', 'incidents', 'budget'}
    need(all(set(paths) == names for paths in (screen_paths, first_paths, second_paths)), 'missing evidence files')
    need(set(binaries) == {'original', 'expanded'}, 'missing hot binaries')
    plan = json.loads(Path(plan_path).read_text(encoding='utf8'))
    screen = evidence.audit_screen(plan, screen_paths, binaries)
    need(screen['gpu'] == gpu and screen['screen_contract']['gpu'] == gpu, 'GPU mismatch')
    hot_generated = screen['screen_contract']['generated_provenance']['sha256']
    need(isinstance(hot_generated, str) and len(hot_generated) == 64, 'missing hot generated provenance hash')
    proofs = {(c['n'], c['batch'], c['candidate_id']): c for c in screen['cases']}
    need(len(proofs) == len(screen['cases']), 'duplicate screen proof')
    raw = {side: {k: frontier.read_log(v) for k, v in paths.items()}
           for side, paths in (('first', first_paths), ('second', second_paths))}
    audits = {}
    for side in ('first', 'second'):
        x = raw[side]
        need(x['rows'] and x['rows'][0]['contract']['generated_provenance']['sha256'] == hot_generated,
             side + ' generated provenance differs from screen')
        report = audit.audit_session(x['rows'], x['guard'], x['incidents'], x['budget'], plan,
                                     frontier.sha(plan_path), gpu, confirmation_commit, binaries, proofs,
                                     evidence.pair_order)
        need(report.get('status') == 'pass' and report.get('session_kind') == side,
             side + ' confirmation audit failed: ' + str(report.get('errors')))
        need(report.get('vkfft_commit') == screen['screen_contract']['vkfft_commit'], 'VkFFT commit differs from screen')
        audits[side] = report
    first_hashes = {key: frontier.sha(path) for key, path in first_paths.items()}
    both = audit.audit_two(audits['first'], audits['second'], first_hashes)
    need(both.get('status') == 'pass', 'two-session audit failed: ' + str(both.get('errors')))
    confirmed = [r for r in both['results'] if r.get('role') == 'winner' and r.get('positive_confirmation') is True]
    cases = []
    for item in confirmed:
        key = (item['n'], item['batch'], item['candidate_id'])
        need(key in proofs and item['request'] == proofs[key]['request'], 'confirmed request differs from screen')
        case_rows = [segment(raw[side]['rows'], key) for side in ('first', 'second')]
        timing = [[r for r in part if r.get('record_type') == 'execution' and r.get('phase') == 'confirmation_timing']
                  for part in case_rows]
        need(all(len(rows) == 20 for rows in timing), 'missing hot confirmation executions')
        metadata = {(r['result']['gpu']['uuid'], r['result']['gpu']['l2_cache_bytes'])
                    for rows in timing for r in rows}
        need(len(metadata) == 1, 'hot device or L2 changed between measurements')
        actual_gpu, l2 = next(iter(metadata))
        need(actual_gpu == gpu and type(l2) is int and l2 > 0, 'invalid confirmed device L2')
        first_round = [r for r in timing[0] if r['round'] == 0]
        original = next(r for r in first_round if r['candidate_id'] == 'original')
        candidate = next(r for r in first_round if r['candidate_id'] == key[2])
        original_signature = search.source_signature(original['result'], original=True)
        candidate_signature = search.source_signature(candidate['result'], original=False)
        need(evidence.signature_hash(original_signature) == proofs[key]['screen_original_signature_sha256'] and
             evidence.signature_hash(candidate_signature) == proofs[key]['screen_candidate_signature_sha256'],
             'confirmed hot signatures differ from screen')
        repeats = max(r['workload']['graph_repeats'] for rows in timing for r in rows)
        rotation = pool(item['n'] * item['batch'] * 8, l2, repeats)
        cases.append({'n': key[0], 'batch': key[1], 'candidate_id': key[2], 'request': item['request'],
                      'hot_original_signature': original_signature, 'hot_candidate_signature': candidate_signature,
                      'rotation': rotation, 'l2_bytes': l2,
                      'status': 'eligible' if rotation else 'unsupported_pool_or_graph_limit'})
    hashes = {side: {k: frontier.sha(v) for k, v in paths.items()}
              for side, paths in (('screen', screen_paths), ('first', first_paths), ('second', second_paths))}
    return {'schema': 'frontier-rotation-eligibility/v1', 'gpu': gpu,
            'confirmation_commit': confirmation_commit, 'vkfft_commit': screen['screen_contract']['vkfft_commit'],
            'hot_binaries': {k: frontier.sha(v) for k, v in binaries.items()},
            'hot_generated_provenance_sha256': hot_generated,
            'plan_sha256': frontier.sha(plan_path), 'evidence_sha256': hashes,
            'cases': cases, 'scope': 'two-session positive winners only; rotation remains unmeasured'}
