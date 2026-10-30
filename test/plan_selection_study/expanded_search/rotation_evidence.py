"""Validate a two-session eligibility gate and derive frozen rotation cases."""
import hashlib
import json
from pathlib import Path

import audit_confirmation
import audit_two_sessions


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf8'))


def verify(a):
    plan=load(a.plan);gate=load(a.gate);first_gate=load(a.first_gate)
    reports={'first':load(a.first_audit),'second':load(a.second_audit)}
    files={name:{field:getattr(a,name+'_'+field) for field in ('rows','guard','incidents','budget')}
           for name in ('first','second')}
    evidence={name:{field:sha(path) for field,path in values.items()} for name,values in files.items()}
    if (gate.get('schema')!='expanded-two-session-gate/v1' or gate.get('status')!='pass' or
        gate.get('plan_sha256')!=sha(a.plan) or gate.get('first_gate_sha256')!=sha(a.first_gate) or
        gate.get('first_audit_sha256')!=sha(a.first_audit) or gate.get('second_audit_sha256')!=sha(a.second_audit) or
        gate.get('protocol_sha256')!=sha(a.protocol) or
        gate.get('combined_audit_sha256')!=sha(a.combined_audit) or gate.get('session_evidence_sha256')!=evidence or
        gate.get('binaries_sha256')!=plan['sha256']):
        raise ValueError('two-session gate is missing, unapproved or bound to different evidence')
    if first_gate.get('evidence_sha256')!=evidence['first'] or first_gate.get('binaries_sha256')!=plan['sha256']:
        raise ValueError('first-session gate does not bind the provided evidence/binaries')
    raw={name:audit_confirmation.read(values['rows']) for name,values in files.items()}
    core={}
    for name in ('first','second'):
        report=reports[name]
        if report.get('status')!='pass': raise ValueError(name+' independent audit wrapper failed')
        core[name]=audit_confirmation.audit(raw[name],audit_confirmation.read(files[name]['guard']),
                                             audit_confirmation.read(files[name]['incidents']),plan,sha(a.plan))
        if core[name].get('status')!='pass' or core[name] != report.get(name+'_session',report):
            raise ValueError(name+' session audit does not match raw evidence')
    second_header=next((r for r in raw['second'] if r.get('record_type')=='confirmation_protocol'),None)
    if not second_header: raise ValueError('second session header missing')
    combined=audit_two_sessions.combine(core['first'],core['second'],first_gate,second_header,
                                         sha(a.first_gate),sha(a.plan),sha(a.first_audit))
    combined_report=load(a.combined_audit)
    if (combined.get('status')!='pass' or combined_report.get('status')!='pass' or
        combined!=combined_report.get('combined') or combined_report.get('first_session')!=core['first'] or
        combined_report.get('second_session')!=core['second'] or
        reports['first'].get('budget')!=combined_report.get('first_budget')):
        raise ValueError('two-session independent audit does not match raw evidence')
    for name in ('first_budget','second_budget'):
        entry=combined_report.get(name,{})
        if entry.get('status')!='pass' or entry.get('phase')!='confirmation' or entry.get('inflight') is not None:
            raise ValueError(name+' audit missing or incomplete')
    confirmed={(x['n'],x['batch'],x['candidate_id']) for x in combined['candidates']
               if x['status']=='confirmed_under_preregistered_rule'}
    lookup={}
    for item in plan['workloads']:
        c=item.get('candidate')
        if c: lookup[(item['n'],item['batch'],c['candidate_id'])]=c
    expected=[]
    for key in sorted(confirmed):
        if key not in lookup: raise ValueError('confirmed candidate missing from plan')
        c=lookup[key]
        expected.append({'n':key[0],'batch':key[1],'candidate_id':key[2],
                         'request':c['request'],'graph_repeats':c['graph_repeats'],
                         'original_signature':c['original_signature'],'candidate_signature':c['candidate_signature']})
    if gate.get('confirmed')!=expected: raise ValueError('gate confirmed candidates differ from independent audit/plan')
    l2={}
    for key in confirmed:
        values=[]
        for name in ('first','second'):
            rows=[r for r in raw[name] if r.get('record_type')=='execution' and r.get('candidate_id')=='original' and
                  (r.get('workload',{}).get('n'),r.get('workload',{}).get('batch'))==key[:2]]
            if not rows: raise ValueError('confirmed workload lacks original device metadata')
            values.extend((r['result']['gpu']['uuid'],r['result']['gpu']['l2_cache_bytes']) for r in rows)
        if len(set(values))!=1 or values[0][1]<=0: raise ValueError('confirmation GPU/L2 metadata varies')
        l2[key]=values[0]
    return plan,gate,expected,l2


def pool(bytes_per_buffer,l2_bytes,hot_repeats):
    if bytes_per_buffer<=0 or l2_bytes<=0: return None
    slots=2
    while slots<=2048 and slots*bytes_per_buffer<=l2_bytes: slots*=2
    if slots>2048 or 2*slots*bytes_per_buffer>1280*1024*1024: return None
    R=max(64,hot_repeats,slots)
    if R>2048 or R&(R-1) or R%slots: return None
    return {'slots':slots,'total_bytes':2*slots*bytes_per_buffer,'graph_repeats':R}
