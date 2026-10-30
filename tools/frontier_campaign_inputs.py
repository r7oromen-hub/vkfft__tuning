"""Resolve and bind committed frontier campaign inputs to their admission evidence."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import uuid

DEFAULT_PROTOCOL = 'experiments/frontier_20261031_protocol.json'
DEFAULT_ADMISSION_RUN = 'frontier-20261031-admission-001'
DEFAULT_GATE = 'experiments/evidence/frontier-20261031-admission-gate.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def git(root, *args):
    return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()


def checked_path(root, value):
    root = Path(root).resolve(strict=True)
    raw = Path(value)
    raw = raw if raw.is_absolute() else root/raw
    if raw.is_symlink(): raise ValueError('symlink input is not allowed')
    path = raw.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('input must be a file inside checkout')
    return path


def tracked(root, path):
    rel=Path(path).relative_to(root).as_posix()
    git(root,'ls-files','--error-unmatch',rel)
    git(root,'diff','--quiet','HEAD','--',rel)


def load_protocol(root, path):
    root=Path(root).resolve(strict=True); path=checked_path(root,path); tracked(root,path)
    cfg=json.loads(path.read_text(encoding='utf8'))
    gpu=cfg.get('gpu_uuid'); cuda=cfg.get('cuda_root')
    try:
        parsed=uuid.UUID(gpu.removeprefix('GPU-')) if isinstance(gpu,str) and gpu.startswith('GPU-') else None
    except ValueError:
        parsed=None
    if (cfg.get('schema')!='frontier-campaign/v1' or parsed is None or gpu!='GPU-'+str(parsed) or
        not isinstance(cuda,str) or not PurePosixPath(cuda).is_absolute() or '..' in PurePosixPath(cuda).parts or
        not isinstance(cfg.get('admission_workloads'),list) or not cfg['admission_workloads']):
        raise ValueError('invalid frontier protocol, GPU, CUDA root, or admission workloads')
    budget=cfg.get('total_reserved_wall_seconds')
    if type(budget) not in (int,float) or not math.isfinite(budget) or budget<=0:
        raise ValueError('invalid total campaign wall-clock budget')
    workloads=set()
    for work in cfg['admission_workloads']:
        n,b=work['n'],work['batch']
        if (type(n) is not int or type(b) is not int or n<=0 or b<=0 or (n,b) in workloads or
            not isinstance(work.get('requests'),list) or not work['requests']):
            raise ValueError('invalid or duplicate admission workload')
        workloads.add((n,b))
    return path,cfg


def load_completed_run(root, run_id, protocol_path):
    root=Path(root).resolve(strict=True)
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',run_id) or run_id in ('.','..'):
        raise ValueError('invalid experiment ID')
    protocol=checked_path(root,protocol_path);tracked(root,protocol)
    record_path=checked_path(root,root/'experiments/records'/(run_id+'.json'));tracked(root,record_path)
    unresolved=root/'experiments/artifacts'/run_id
    if unresolved.is_symlink():raise ValueError('artifact directory cannot be a symlink')
    artifacts=unresolved.resolve(strict=True)
    intended=(root/'experiments/artifacts').resolve(strict=True)
    if not artifacts.is_relative_to(intended) or not artifacts.is_dir():raise ValueError('artifact directory escapes root')
    record=json.loads(record_path.read_text(encoding='utf8'))
    if record.get('schema')!='git-experiment/v1' or record.get('id')!=run_id or record.get('status')!='completed' or record.get('returncode')!=0:
        raise ValueError('experiment record is not completed successfully')
    info=record.get('protocol') or {}; rel=protocol.relative_to(root).as_posix()
    if info.get('path')!=rel or info.get('bytes')!=protocol.stat().st_size or info.get('sha256')!=sha(protocol):
        raise ValueError('experiment protocol differs')
    commit=record.get('execution_commit')
    if not isinstance(commit,str) or not re.fullmatch(r'[0-9a-f]{40}',commit):raise ValueError('invalid execution commit')
    git(root,'merge-base','--is-ancestor',commit,'HEAD')
    parent=git(root,'rev-parse',commit+'^')
    rel_record=record_path.relative_to(root).as_posix()
    if git(root,'diff','--name-only',parent,commit).splitlines()!=[rel_record]:
        raise ValueError('execution commit is not a single run declaration')
    declaration=json.loads(git(root,'show',commit+':'+rel_record))
    if (declaration.get('schema')!='git-experiment/v1' or declaration.get('status')!='declared' or
        declaration.get('id')!=run_id or declaration.get('parent_code_commit')!=parent or
        declaration.get('protocol')!=info):
        raise ValueError('execution declaration does not match completed record')
    return record,artifacts,record_path


def verify_artifacts(root, record, paths):
    root=Path(root).resolve(strict=True)
    artifact_root=(root/'experiments/artifacts'/record['id']).resolve(strict=True)
    entries=record.get('artifacts')
    if not isinstance(entries,list):raise ValueError('artifact inventory missing')
    index={}
    for entry in entries:
        rel=entry.get('path')
        if not isinstance(rel,str) or rel in index:raise ValueError('invalid or duplicate artifact entry')
        p=PurePosixPath(rel)
        if p.is_absolute() or '..' in p.parts or str(p)!=rel:raise ValueError('artifact path escapes checkout or is not canonical')
        index[rel]=entry
    requested=set()
    for raw in paths:
        path=checked_path(root,raw)
        if not path.is_relative_to(artifact_root):raise ValueError('artifact outside experiment')
        rel=path.relative_to(root).as_posix()
        if rel in requested:raise ValueError('duplicate requested artifact')
        requested.add(rel);entry=index.get(rel)
        if not entry or entry.get('bytes')!=path.stat().st_size or entry.get('sha256')!=sha(path):
            raise ValueError('artifact missing or changed: '+rel)


def load_campaign(root, protocol, admission_run, gate):
    root=Path(root).resolve(strict=True)
    protocol,cfg=load_protocol(root,protocol)
    record,artifacts,record_path=load_completed_run(root,admission_run,protocol)
    gate=checked_path(root,gate);tracked(root,gate)
    doc=json.loads(gate.read_text(encoding='utf8'));commit=record['execution_commit'];gpu=cfg['gpu_uuid']
    count=sum(len(w['requests']) for w in cfg['admission_workloads'])
    if (doc.get('schema')!='frontier-admission-gate/v1' or doc.get('status')!='pass' or
        doc.get('gpu')!=gpu or doc.get('execution_commit')!=commit or doc.get('candidates')!=count or
        doc.get('workloads')!=len(cfg['admission_workloads']) or doc.get('binary_hashes_rechecked') is not True or
        (doc.get('guard_audit') or {}).get('status')!='pass'):
        raise ValueError('admission gate identity or completeness mismatch')
    evidence=doc.get('evidence_sha256');files={}
    if not isinstance(evidence,dict):raise ValueError('admission gate evidence missing')
    for raw,digest in evidence.items():
        name=str(raw).replace('\\','/').rsplit('/',1)[-1]
        if not name or name in files:raise ValueError('ambiguous gate evidence filename')
        files[name]=digest
    names={'admission.jsonl','guard.jsonl','incidents.jsonl','budget.jsonl'}
    names.update(f"n{w['n']}_b{w['batch']}.manifest.jsonl" for w in cfg['admission_workloads'])
    if set(files)!=names:raise ValueError('gate evidence set differs from admission workloads')
    paths=[artifacts/name for name in sorted(names)]
    paths += [artifacts/'campaign_start.json',artifacts/'build/compile_commands.json',
              artifacts/'build/vkfft_frontier_original',artifacts/'build/vkfft_frontier_v2',
              artifacts/'generated/expanded_provenance.json']
    verify_artifacts(root,record,paths)
    if any(sha(artifacts/name)!=digest for name,digest in files.items()):raise ValueError('gate evidence hash differs')
    with (artifacts/'admission.jsonl').open(encoding='utf8') as f: first_row=json.loads(next(f))
    contract=first_row['contract']
    protocol_rel=protocol.relative_to(root).as_posix()
    source_matches=[h for name,h in (contract.get('source_files') or {}).items()
                    if str(name).replace('\\','/').endswith('/'+protocol_rel)]
    compile_matches=[h for name,h in (contract.get('build_artifacts') or {}).items()
                     if str(name).replace('\\','/').endswith('/build/compile_commands.json')]
    manifest_hashes={}
    for name,digest in (contract.get('manifests') or {}).items():
        basename=str(name).replace('\\','/').rsplit('/',1)[-1]
        if basename in manifest_hashes:raise ValueError('ambiguous admission manifest paths')
        manifest_hashes[basename]=digest
    expected_manifest_hashes={name:files[name] for name in names if name.endswith('.manifest.jsonl')}
    if (first_row.get('record_type')!='session_start' or contract.get('gpu')!=gpu or contract.get('execution_commit')!=commit or
        source_matches!=[sha(protocol)] or compile_matches!=[sha(artifacts/'build/compile_commands.json')] or
        manifest_hashes!=expected_manifest_hashes or contract.get('vkfft_commit')!=git(root,'rev-parse','HEAD:VkFFT') or
        contract.get('binaries')!={'original':sha(artifacts/'build/vkfft_frontier_original'),
                                   'expanded':sha(artifacts/'build/vkfft_frontier_v2')} or
        contract.get('generated_provenance',{}).get('sha256')!=sha(artifacts/'generated/expanded_provenance.json')):
        raise ValueError('admission execution, binaries, GPU or generated source mismatch')
    start=json.loads((artifacts/'campaign_start.json').read_text(encoding='utf8'))
    with (artifacts/'budget.jsonl').open(encoding='utf8') as f:
        first=json.loads(next(f));last=first
        for line in f:
            if line.strip():last=json.loads(line)
    state=first['state'];terminal=last['state']
    if (first.get('event')!='start' or state.get('contract')!=contract or terminal.get('contract')!=contract or
        terminal.get('inflight') is not None or terminal.get('boot')!=state.get('boot') or
        terminal.get('start')!=state.get('start') or terminal.get('last',-1)<state.get('last',0) or
        start.get('gpu')!=gpu or start.get('execution_commit')!=commit or
        start.get('total_reserved_wall_seconds')!=cfg['total_reserved_wall_seconds'] or
        type(start.get('start_monotonic')) not in (int,float) or not math.isfinite(start['start_monotonic']) or
        state['start']<start['start_monotonic']):
        raise ValueError('admission budget or campaign identity mismatch')
    return dict(protocol=protocol,gate=gate,admission_dir=artifacts,record_path=record_path,
                config=cfg,gate_data=doc,record=record,campaign_start=start,budget_state=terminal,
                admission_contract=contract,gpu=gpu,cuda_root=cfg['cuda_root'],admission_run=admission_run,execution_commit=commit)


def validate_clock(campaign, *, now=None, boot=None):
    now=time.monotonic() if now is None else now
    boot=Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip() if boot is None else boot
    start=campaign['campaign_start']['start_monotonic'];state=campaign['budget_state']
    if (not isinstance(now,(int,float)) or not math.isfinite(now) or now<start or now<state['last'] or
        not isinstance(boot,str) or not boot or state['boot']!=boot):
        raise ValueError('campaign boot or monotonic clock changed')
    deadline=start+campaign['config']['total_reserved_wall_seconds']
    if now>=deadline:raise ValueError('campaign deadline exhausted')
    return deadline
