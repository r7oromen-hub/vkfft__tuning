#!/usr/bin/env python3
"""Independently audit the downloaded three-way records and write a Chinese report."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

def require(condition, reason):
    if not condition: raise ValueError(reason)

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args(); root=args.directory
    data=root/'original_compare_n256_b4096_01.jsonl';guard=root/'original_compare_n256_b4096_01.guard.jsonl'; incidents=root/'original_compare_n256_b4096_01.incidents.jsonl'
    rows=read(data); gs=read(guard); require(not incidents.read_bytes(),'guard incident log nonempty')
    require(rows[0].get('record_type')=='protocol' and rows[-1].get('record_type')=='controller_complete' and rows[-1].get('status')=='ok','missing protocol/completion')
    protocol=rows[0];gpu=protocol['gpu'];require((protocol['n'],protocol['batch'],protocol['seed'],protocol['graph_repeats'],protocol['rounds'])==(256,4096,42,512,10),'protocol mismatch')
    require(protocol['winner_environment']=={'VKFFT_DIRECT_RADIX':'16,16','VKFFT_DIRECT_BATCH':'1','VKFFT_DIRECT_BLOCK':'16,1,1'},'winner request mismatch')
    for name,file in [('joint','vkfft_direct_candidate'),('original','vkfft_original_control')]:
        require(sha(root/file)==protocol['sha256'][name],name+' binary checksum mismatch')
    require(protocol['sha256']['winner']==protocol['sha256']['joint'],'winner binary mismatch')
    build=(root/'build_original.log').read_text(encoding='utf8')
    commands=json.loads((root/'original_compile_commands.json').read_text(encoding='utf8'))
    require(len(commands)==1,'unexpected original compiler commands')
    cmd=commands[0]['command'];deps=(root/'original_compile_dependencies.d').read_text(encoding='utf8')
    original_include='/source_joint/VkFFT/vkFFT';generated_include='/source_joint/build_direct_candidates/generated/include'
    require(original_include in cmd and generated_include in cmd and cmd.index(original_include)<cmd.index(generated_include),'original header include precedence missing')
    toolkit=re.search(r'Found CUDAToolkit: /usr/local/cuda-([0-9]+\.[0-9]+)/include \(found version "([^"]+)"\)',build)
    require(toolkit is not None,'original CUDA toolkit evidence missing')
    toolkit_path,toolkit_version=toolkit.groups()
    require('-DVKFFT_STUDY_UNINSTRUMENTED=1' in cmd and 'generated/direct_benchmark.cpp' in cmd and '-std=gnu++17' in cmd and '-O3' in cmd and f'/usr/local/cuda-{toolkit_path}/include' in cmd,'unexpected original compiler options')
    require('GNU 11.4.0' in build and 'Built target vkfft_original_control' in build,'original build evidence missing')
    require('/generated/include/vkFFT/' not in deps and '/generated/include/vkFFT.h' not in deps,'generated VkFFT headers in dependencies')
    header_dependencies=re.findall(r'[^\s\\]*?/source_joint/VkFFT/vkFFT/[^\s\\]+',deps)
    require(len(header_dependencies)>0 and any(x.endswith('/vkFFT.h') for x in header_dependencies),'original VkFFT dependencies missing')
    require(gs[0]['event']=='guard_start' and gs[-1]['event']=='guard_complete' and gs[-1]['success'] is True and gs[-1]['clean'] is True,'guard did not complete cleanly')
    gid=gs[0]['guard_id'];require(all(x.get('gpu_uuid')==gpu and x.get('guard_id')==gid for x in gs),'guard identity mismatch')
    idle=[x for x in gs if x['event']=='idle_complete'];require(len(idle)==1 and idle[0]['clean'] is True and idle[0]['observed_idle_seconds']>=30,'idle precheck invalid')
    samples=[x for x in gs if x['event']=='sample'];require(len(samples)==gs[-1]['sample_count'] and all(x['ok'] is True and not x['violation_reasons'] and not x['foreign_compute_pids'] and x['monitor_error'] is None and x['gpu']['memory_used_mib']<=2048 for x in samples),'bad telemetry')
    start={x['active_run']['guard_run_id']:x for x in gs if x['event']=='run_one_start'}
    end={x['guard_run_id']:x for x in gs if x['event']=='run_one_end'}
    require(len(start)==len(end)==45 and set(start)==set(end) and all(x['clean'] is True and x['status']=='ok' for x in end.values()),'guard process registration/completion mismatch')
    measured=[x for x in rows if x.get('candidate_id') in ('joint','original','winner')];require(len(measured)==45 and not any(x.get('record_type')=='comparison_failure' for x in rows),'unexpected measurement count or failed comparison')
    require({x['guard_run_id'] for x in measured}==set(start),'measurement/guard IDs mismatch')
    variants=('joint','original','winner');patterns=('zero','impulse','constant','quarter-tone','random'); groups={}; structures={}; all_correct=[]
    for x in measured:
        v=x['candidate_id']; r=x['result'];w=x['workload'];rid=x['guard_run_id'];a=start[rid]['active_run'];
        require(x['status']=='ok' and x['returncode']==0 and x['guard_clean'] is True and x['guard_id']==gid and x['guard_gpu_uuid']==gpu and not x['validation_errors'],'bad process row')
        require((a['candidate_id'],a['phase'],a['round'],a['order'])==(v,x['phase'],x['round'],x['order']) and end[rid]['candidate_id']==v,'guard linkage mismatch')
        require(json.loads(x['stdout'])==r,'stdout/result mismatch')
        require(r['gpu']['uuid']==gpu and 'RTX 4090' in r['gpu']['name'] and (r['n'],r['batch'],r['seed'])==(256,4096,42),'GPU or workload mismatch')
        require((r['precision'],r['placement'],r['timing_mode'],r['forward_direction'],r['graph_repeats'],r['graph_kernel_count'],r['graph_event_record_count'])==('complex_fp32','outofplace','graph',-1,512,512,2),'execution protocol mismatch')
        require(r['header_instrumentation']==(v!='original') and r['requested_config']=={'makeForwardPlanOnly':1},'header mode or requested knobs mismatch')
        require(x['direct_environment']==(protocol['winner_environment'] if v=='winner' else {}),'direct override mismatch')
        require(w['config']=={} and r['pattern']==w['pattern'] and r['validation_output_initialization']=='0xff_bytes_fp32_nan_before_single_and_final_timed_forward','input/poison mismatch')
        require(r['single_forward_input_unchanged'] is True and r['input_unchanged'] is True,'immutable input changed')
        for k in ('single_forward_correctness','correctness'):
            c=r[k];require(c['passed'] is True and c['criterion']=='criterion_v2' and c['checked_elements']==256*4096 and c['nonfinite_count']==0 and c['failing_batches']==0,k+' failed')
            require(c['worst_batch_l2_scaled_error']<=1e-5 and c['worst_batch_linf_input_scaled_error']<=1e-4,'error thresholds exceeded');all_correct.append(c)
        expected=31 if x['phase']=='timing' else 1
        require(len(r['exec_ms_samples'])==len(r['raw_interval_ms_samples'])==expected,'sample count invalid')
        require(all(math.isfinite(a) and a>0 and math.isclose(a*512,b,rel_tol=1e-6) for a,b in zip(r['exec_ms_samples'],r['raw_interval_ms_samples'])),'time samples invalid')
        if expected==31: require(min(r['raw_interval_ms_samples'])>=.5 and r['warmup_wall_ms']>=1000 and r['warmup_actual']>=5,'timing warmup/interval failed')
        k=r['structural_features']['kernels'][0]; require(r['structural_features']['numAxisUploads']==1,'upload mismatch')
        if v=='winner':
            require(k['stageRadix']==[16,16] and k['groupedBatch']==1 and k['localSize']==[16,1,1],'winner plan mismatch')
            z=r['direct_resource_validation'];require(z['dynamic_shared_bytes']<=z['function_dynamic_limit_bytes'] and z['dynamic_shared_bytes']+z['static_shared_bytes']<=z['device_shared_limit_bytes'] and z['threads']<=z['function_thread_limit'],'winner resource limit exceeded')
        else: require(k['stageRadix']==[16,16] and k['groupedBatch']==8 and k['localSize']==[8,16,1],'default plan mismatch')
        source=(r['postinit_source_kernels'] if v=='original' else r['precompile_kernels'])[0]
        fingerprint=(r['structural_features'],r['compiled_diagnostics'],source['canonical_descriptor'],source['source_sha256'],source['source_bytes'])
        if v in structures: require(structures[v]==fingerprint,'structure changed within variant')
        structures[v]=fingerprint
        if v=='original': require(r['init_trace']['available'] is False and source['actual_compiler_options'] is None,'original trace claim inaccurate')
        else: require(r['descriptor_validation']['all_match'] is True and source['compiler_options_match'] is True,'joint descriptor/options mismatch')
        group=(x['phase'],w['pattern'] if x['phase']=='numerical' else x['round']);require(v not in groups.setdefault(group,{}),'duplicate variant');groups[group][v]=x
    require(set(groups)=={('numerical',x) for x in patterns}|{('timing',x) for x in range(10)},'missing pattern or timing round')
    require(all(set(group)==set(variants) and {x['order'] for x in group.values()}=={0,1,2} for group in groups.values()),'missing or unordered triple')
    require(structures['joint']==structures['original'],'original/joint actual structure, resources, canonical descriptor or source mismatch')
    for group in groups.values():
        j=group['joint']['result'];o=group['original']['result'];a=dict(j['resolved_config']);b=dict(o['resolved_config'])
        require(a.pop('keepShaderCode')==0 and b.pop('keepShaderCode')==1 and a==b,'resolved configuration mismatch')
        pc=j['precompile_kernels'][0]; post=o['postinit_source_kernels'][0];opts=post['expected_compiler_input']
        require(opts['options']==pc['actual_compiler_options'] and opts['original_compile_header_sha256']==pc['compiler_input']['original_compile_header_sha256'],'expected original compiler options/header mismatch')
    jr=[];wr=[];medians=[]
    for i in range(10):
        v={name:statistics.median(groups[('timing',i)][name]['result']['exec_ms_samples']) for name in variants}
        medians.append(v);jr.append(v['joint']/v['original']);wr.append(v['winner']/v['original'])
    done=rows[-1];require(all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(jr,done['joint_over_original_median_ratios'])) and all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(wr,done['winner_over_original_median_ratios'])),'reported ratios differ from raw samples')
    require(math.isclose(statistics.median(jr),done['joint_over_original_median'],rel_tol=1e-12) and math.isclose(statistics.median(wr),done['winner_over_original_median'],rel_tol=1e-12),'reported median mismatch')
    original=structures['original'];winner=structures['winner'];ok=original[0]['kernels'][0];wk=winner[0]['kernels'][0]
    digest={'input_sha256':{path.name:sha(path) for path in (data,guard,incidents,root/'vkfft_direct_candidate',root/'vkfft_original_control',root/'build_original.log',root/'original_compile_commands.json',root/'original_compile_dependencies.d')},'gpu':gpu,'original_header_dependency_entries':len(header_dependencies),'joint_over_original':jr,'winner_over_original':wr,'round_medians_ms':medians,'guard_samples':len(samples),'gpu_max_memory_mib':max(x['gpu']['memory_used_mib'] for x in samples),'max_l2_scaled_error':max(x['worst_batch_l2_scaled_error'] for x in all_correct),'max_linf_scaled_error':max(x['worst_batch_linf_input_scaled_error'] for x in all_correct)}
    (root/'original_compare_n256_b4096_audit.json').write_text(json.dumps(digest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    lines=['# 原始 VkFFT 默认与 joint 三路独立核验','',f"GPU：RTX 4090，UUID `{gpu}`。工作负载为 FP32 C2C 正变换，N=256、batch=4096、out-of-place；相同 seed=42，R=512 的 CUDA graph 重复读取不可变输入。时间为单次完整 batch 的 device event 时间，按 R 平均。",
           '',f"原始默认与实验默认的实际结构、完整记录的 kernel 资源、规范化描述及源码指纹逐项一致。两者均为 radix `[16,16]`、groupedBatch=8、实际 localSize `[8,16,1]`、动态共享内存 18432 字节、每线程 64 个 CUDA 寄存器、local memory 0 字节；源码 SHA-256 `{original[3]}`（{original[4]} 字节）。", 
           '',f"胜者 radix `[16,16]`、groupedBatch=1、localSize `[16,1,1]`、动态共享内存 2176 字节、每线程 64 个 CUDA 寄存器、local memory 0 字节；源码 SHA-256 `{winner[3]}`（{winner[4]} 字节）。原始默认与胜者逐轮的结构和源码记录保存在原始 JSONL 的 `original_winner_comparison` 中。",
           '',f"复算 10 轮配对中位数：实验默认/原始默认 = **{statistics.median(jr):.6f}**；胜者/原始默认 = **{statistics.median(wr):.6f}**，即原始默认/胜者 **{1/statistics.median(wr):.4f}×**，胜者耗时低约 **{100*(1-statistics.median(wr)):.2f}%**。这些是相同图执行和缓存条件下的设备执行时间，不是初始化或端到端应用加速。",
           '', '| 轮次 | 原始默认 ms | 实验默认 / 原始 | 胜者 ms | 胜者 / 原始 |','|---:|---:|---:|---:|---:|']
    for i,(v,a,b) in enumerate(zip(medians,jr,wr),1):lines.append(f"| {i} | {v['original']:.9f} | {a:.6f} | {v['winner']:.9f} | {b:.6f} |")
    lines += ['',f"5 种输入 × 3 路的数值检查和 30 次计时进程的前后验证均通过；每次检查全部 1,048,576 个输出且输入逐位不变。两次检查共 {len(all_correct)} 项，最坏 L2 缩放误差 {digest['max_l2_scaled_error']:.3g}，最坏 Linf 输入缩放误差 {digest['max_linf_scaled_error']:.3g}。逐次图事件区间至少 0.5 ms，计时暖机至少 1 秒。",
              '',f"保护日志关联 45 个进程，{len(samples)} 次监控采样全部合格，外部计算进程和事故记录均为零，采样显存峰值 {digest['gpu_max_memory_mib']:.0f} MiB。采样不能排除采样间隙内的短暂干扰。两份本地二进制 SHA-256 与协议记录相符。",
              '',f"构建记录显示控制程序从同一 joint 生成的 harness 编译，使用 GNU 11.4.0、CUDA {toolkit_version}，头文件搜索顺序中原始 VkFFT 目录在生成目录之前；依赖文件中有 {len(header_dependencies)} 条原始 VkFFT 头路径，未见生成目录中的 VkFFT 头路径。已下载构建证据不含远端整棵源码的逐文件校验日志，因此路径证据本身不能独立证明远端各头文件内容未改。",
              '',"原始头文件路径采用初始化后保留源码的工程校验：其 precompile 快照、实际 NVRTC options 和初始化 trace 不可用。原始路径的预期选项与实验实际选项一致，但不是对原始编译参数的直接观察；该路径初始化期间 `keepShaderCode=1`，执行前释放源码并恢复为 0，因此不比较初始化时间。10 轮结果限定于此工作负载、该机器、graph 重复访问的缓存状态及本次运行，不能推广为所有尺寸的收益。",'']
    (root/'original_compare_n256_b4096_audit_zh.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(digest,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
