"""Index pre-Git evidence without inventing historical execution commits.

Writes records only; committing each record is a separate explicit operation.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / 'test/plan_selection_study/expanded_search'
RESULTS = EXPANDED / 'results_20261030'
HISTORY = ROOT / 'experiments/history'


def file_info(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def main():
    baseline = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip()
    HISTORY.mkdir(parents=True, exist_ok=True)
    evidence_cache = {}
    def record(identity, files, status):
        target = HISTORY / (identity + '.json')
        if target.exists():
            raise ValueError('record already exists: ' + identity)
        entries = []
        for p in files:
            key = str(p)
            if key not in evidence_cache:
                evidence_cache[key] = file_info(p)
            entries.append(evidence_cache[key])
        obj = {'schema':'historical-experiment-index/v1', 'id':identity,
               'execution_commit':None, 'archive_basis_commit':baseline,
               'provenance_status':'retrospective_evidence_index_not_execution_commit',
               'status_description':status, 'evidence':entries,
               'limitations':['Original run predates this experiment Git repository.',
                             'Current imported source is not asserted identical to historical executed source.',
                             'Local evidence paths and hashes do not imply raw data has been uploaded.']}
        target.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    groups = [
        ('20261030-admission-v1','admission_dc66*.jsonl','Stopped on identity-check issue; retained failed admission'),
        ('20261030-admission-v2','admission_v2_b6*.jsonl','Fresh repaired admission passed; snapshot and final logs retained'),
        ('20261030-search','search_7c*.jsonl','393 complete candidates, 255 scheduler rejects, 1 partial'),
        ('20261030-confirmation-session1','confirmation_first_d0*.jsonl','11 workloads, 110 pairs'),
        ('20261030-confirmation-session2','confirmation_second_d3*.jsonl','9 workloads, 90 pairs'),
        ('20261030-rotation-v1','rotation_build_v1_*.json','Build failed before GPU experiment'),
        ('20261030-rotation-v2','rotation_uncovered_local_audit.json','Repaired build passed; GPU rotation experiment not started'),
    ]
    for identity, pattern, status in groups:
        files = sorted(RESULTS.glob(pattern))
        assert files, identity
        files += [RESULTS/'expanded_provenance_v2.json', EXPANDED/'study_protocol.json']
        if 'confirmation' in identity:
            files += [EXPANDED/'confirmation_source_manifest_20261030.json', RESULTS/'two_session_local_combined.json']
        if identity.endswith('search'):
            files += [RESULTS/'search_local_distribution.json', RESULTS/'search_evidence_immutable.tar']
        record(identity, files, status)
    # Older result folders are registered separately, without assuming their outcomes.
    for base in [ROOT/'test/plan_tuning/results', ROOT/'test/plan_selection_study/results']:
        if not base.exists(): continue
        for manifest in sorted(base.rglob('manifest.json')):
            rel = manifest.relative_to(ROOT).as_posix()
            identity = 'legacy-' + hashlib.sha256(rel.encode()).hexdigest()[:16]
            record(identity, [manifest], 'Manifest discovered; historical outcome/source correspondence not re-audited in this import')
    snapshots = HISTORY / 'source_snapshots'
    snapshots.mkdir(exist_ok=True)
    for p in [RESULTS/'expanded_generated_repair_v2.tar', EXPANDED/'post_search_toolchain_v2_20261030.tar']:
        shutil.copyfile(p, snapshots/p.name)
    for p in [RESULTS/'search_local_distribution.json', RESULTS/'two_session_local_combined.json', RESULTS/'详细分析汇总.json']:
        shutil.copyfile(p, HISTORY/p.name)
    print(json.dumps({'historical_records':len(list(HISTORY.glob('2026*.json')))+len(list(HISTORY.glob('legacy-*.json'))),
                      'source_archives':2,'archive_basis_commit':baseline}))


if __name__ == '__main__':
    main()
