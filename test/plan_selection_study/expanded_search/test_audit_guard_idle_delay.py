import copy
import unittest

import audit_guard


class IdleDelayAuditTests(unittest.TestCase):
    def fixture(self):
        shared={'guard_id':'g','gpu_uuid':'GPU-x'}
        active={'guard_run_id':'g:1','phase':'screen_timing','workload_id':'n16_b1',
                'candidate_id':'c','round':0,'order':0}
        row=dict(record_type='execution',guard_id='g',guard_gpu_uuid='GPU-x',guard_clean=True,
                 guard_run_id='g:1',command=['/bin/test'],
                 **{k:active[k] for k in ('phase','workload_id','candidate_id','round','order')})
        sample=dict(shared,event='sample',active_run=active,ok=True,gpu={'uuid':'GPU-x'},
                    violation_reasons=[],foreign_compute_pids=[],monitor_error=None)
        events=[dict(shared,event='guard_start',own_signal_method='pidfd_with_rechecked_PPID_exe_starttime'),
                dict(shared,event='run_one_start',active_run=active),dict(sample,boundary='before'),
                dict(shared,event='own_harness_registered',active_run=active,pidfd_retained=True,identity={'exe':'/bin/test'}),
                dict(sample,boundary='after'),dict(shared,event='run_one_end',guard_run_id='g:1'),
                dict(shared,event='guard_complete',clean=True,success=True)]
        return [row],events

    def delayed(self):
        rows,events=self.fixture()
        delay=dict(event='sample',guard_id='g',gpu_uuid='GPU-x',boundary='idle',active_run=None,
                   idle_monitor_delay=True,ok=False,violation_reasons=[],monitor_error=None,
                   foreign_compute_pids=[],compute_apps=[],gpu={'uuid':'GPU-x','memory_used_mib':1024},
                   query_wall_seconds=2.5,sample_started_monotonic=7.5,sample_completed_monotonic=10.0)
        events.insert(1,delay)
        next(e for e in events if e.get('event')=='run_one_start')['run_transition_monotonic']=11.0
        next(e for e in events if e.get('event')=='run_one_end')['run_transition_monotonic']=16.0
        before=next(e for e in events if e.get('boundary')=='before')
        before.update(query_wall_seconds=2.0,sample_started_monotonic=11.0,
                      sample_completed_monotonic=13.0,idle_monitor_delay=False)
        reg=next(e for e in events if e.get('event')=='own_harness_registered')
        reg['before_sample_age_seconds']=2.0
        reg['harness_launch_check_monotonic']=15.0
        return rows,events

    def status(self,rows,events):
        return audit_guard.audit(rows,events,[],'GPU-x')['status']

    def test_clean_legacy_without_timing_and_idle_delay_without_later_run(self):
        rows,events=self.fixture();self.assertEqual(self.status(rows,events),'pass')
        _,new_events=self.delayed();delay=new_events[1]
        next(e for e in events if e.get('event')=='run_one_start')['run_transition_monotonic']=11.0
        next(e for e in events if e.get('event')=='run_one_end')['run_transition_monotonic']=16.0
        delay.update(sample_started_monotonic=17.5,sample_completed_monotonic=20.0)
        events.insert(-1,delay)
        self.assertEqual(self.status(rows,events),'pass')

    def test_valid_idle_delay_then_fresh_successful_before(self):
        rows,events=self.delayed()
        self.assertEqual(self.status(rows,events),'pass')
        self.assertEqual(audit_guard.audit(rows,events,[{'incident':True}],'GPU-x')['status'],'fail')

    def test_untrusted_idle_delay_fields_fail_closed(self):
        mutations=[('boundary','during'),('active_run',{'guard_run_id':'g:1'}),('ok',True),
                   ('violation_reasons',['foreign_compute_process']),('foreign_compute_pids',[99]),
                   ('monitor_error','query failed'),('compute_apps',[{'pid':99}]),
                   ('query_wall_seconds',2),('query_wall_seconds',float('nan')),
                   ('sample_completed_monotonic',None),('sample_started_monotonic',None),
                   ('sample_started_monotonic',8.0),('gpu',{'uuid':'GPU-x','memory_used_mib':2049}),
                   ('gpu',{'uuid':'GPU-wrong','memory_used_mib':0}),('gpu',{'uuid':'GPU-x'}),
                   ('guard_run_id','g:1')]
        for name,value in mutations:
            with self.subTest(field=name,value=value):
                rows,events=self.delayed();events[1][name]=value
                self.assertEqual(self.status(rows,events),'fail')
        for name in ('active_run','compute_apps','monitor_error','query_wall_seconds','sample_completed_monotonic','sample_started_monotonic'):
            with self.subTest(missing=name):
                rows,events=self.delayed();events[1].pop(name)
                self.assertEqual(self.status(rows,events),'fail')

    def test_delay_may_not_overlap_run_even_if_it_claims_no_active_process(self):
        rows,events=self.delayed();delay=events.pop(1);events.insert(2,delay)
        self.assertEqual(self.status(rows,events),'fail')

    def test_delay_interval_and_run_transitions_are_independently_verified(self):
        rows,events=self.delayed()
        next(e for e in events if e.get('event')=='run_one_start')['run_transition_monotonic']=9.0
        self.assertEqual(self.status(rows,events),'fail')
        rows,events=self.delayed()
        next(e for e in events if e.get('event')=='run_one_end')['run_transition_monotonic']=10.0
        self.assertEqual(self.status(rows,events),'fail')
        rows,events=self.delayed()
        next(e for e in events if e.get('event')=='run_one_end').pop('run_transition_monotonic')
        self.assertEqual(self.status(rows,events),'fail')

    def test_delay_requires_fast_before_and_recent_registration(self):
        for field,value in [('query_wall_seconds',None),('query_wall_seconds',2.001),
                            ('sample_completed_monotonic',None),('before_sample_age_seconds',None),
                            ('before_sample_age_seconds',2.001),('before_sample_age_seconds',-1),
                            ('before_sample_age_seconds',float('nan')),
                            ('harness_launch_check_monotonic',None),('harness_launch_check_monotonic',16.0),
                            ('harness_launch_check_monotonic',float('nan'))]:
            with self.subTest(field=field,value=value):
                rows,events=self.delayed()
                target=next(e for e in events if e.get('boundary')=='before') if field in ('query_wall_seconds','sample_completed_monotonic') else next(e for e in events if e.get('event')=='own_harness_registered')
                target[field]=value
                self.assertEqual(self.status(rows,events),'fail')
        for field in ('query_wall_seconds','sample_completed_monotonic','before_sample_age_seconds','harness_launch_check_monotonic'):
            with self.subTest(missing=field):
                rows,events=self.delayed()
                target=next(e for e in events if e.get('boundary')=='before') if field in ('query_wall_seconds','sample_completed_monotonic') else next(e for e in events if e.get('event')=='own_harness_registered')
                target.pop(field)
                self.assertEqual(self.status(rows,events),'fail')

    def test_before_must_precede_child_registration(self):
        rows,events=self.delayed()
        before=next(i for i,e in enumerate(events) if e.get('boundary')=='before')
        reg=next(i for i,e in enumerate(events) if e.get('event')=='own_harness_registered')
        events[before],events[reg]=events[reg],events[before]
        self.assertEqual(self.status(rows,events),'fail')

    def test_before_interval_must_be_inside_run_and_end_before_launch(self):
        for field,value in [('sample_started_monotonic',None),('sample_started_monotonic',10.0),
                            ('sample_started_monotonic',11.5),('sample_started_monotonic',14.0),
                            ('sample_completed_monotonic',16.0)]:
            with self.subTest(field=field,value=value):
                rows,events=self.delayed()
                next(e for e in events if e.get('boundary')=='before')[field]=value
                self.assertEqual(self.status(rows,events),'fail')
        rows,events=self.delayed()
        next(e for e in events if e.get('boundary')=='before').pop('sample_started_monotonic')
        self.assertEqual(self.status(rows,events),'fail')
        rows,events=self.delayed()
        next(e for e in events if e.get('event')=='run_one_end')['run_transition_monotonic']=14.0
        self.assertEqual(self.status(rows,events),'fail')

    def test_false_or_untyped_delay_must_not_bypass_normal_checks(self):
        rows,events=self.delayed();events[1]['idle_monitor_delay']=False
        self.assertEqual(self.status(rows,events),'fail')
        rows,events=self.delayed();events[1]['idle_monitor_delay']='true'
        self.assertEqual(self.status(rows,events),'fail')
        rows,events=self.fixture();before=next(e for e in events if e.get('boundary')=='before');before['query_wall_seconds']=2.001
        self.assertEqual(self.status(rows,events),'pass')  # Legacy used a second clock reading.
        before['idle_monitor_delay']=False
        self.assertEqual(self.status(rows,events),'fail')


if __name__=='__main__':unittest.main()
