#!/usr/bin/env python3
"""Derive isolated multi-address graph benchmark from a frozen generated harness."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def once(text, old, new):
    if text.count(old)!=1: raise ValueError(f'expected one anchor, got {text.count(old)}: {old[:100]!r}')
    return text.replace(old,new,1)


ROTATION = r'''
        if (o.placement != "outofplace" || o.timing_mode != "graph" || o.pattern != "random" || o.profile_only)
            throw std::runtime_error("rotation requires execute/outofplace/graph/random");
        if (resources.app.localFFTPlan->numAxisUploads[0] != 1)
            throw std::runtime_error("rotation requires one kernel upload");
        cudaDeviceProp rotation_prop{}; cuda_check(cudaGetDeviceProperties(&rotation_prop,0), "rotation GPU properties");
        rotation_pool::Plan rotation{};
        constexpr std::size_t max_pool_bytes = std::size_t(1280) * 1024 * 1024;
        if (rotation_prop.l2CacheSize <= 0 ||
            !rotation_pool::plan(bytes, (std::size_t)rotation_prop.l2CacheSize, max_pool_bytes, o.graph_repeats, rotation))
            throw std::runtime_error("rotation_unsupported_pool_or_graph_size");
        std::size_t free_device=0,total_device=0;
        cuda_check(cudaMemGetInfo(&free_device,&total_device), "rotation memory headroom");
        if (rotation.total_bytes - 2*bytes > free_device)
            throw std::runtime_error("rotation_unsupported_device_memory");
        struct RotationResources {
            Resources& owner; std::vector<void*> inputs, outputs;
            RotationResources(Resources& r,std::size_t slots):owner(r),inputs(slots,nullptr),outputs(slots,nullptr) {
                inputs[0]=owner.input_buffer; outputs[0]=owner.buffer;
            }
            ~RotationResources() {
                if (owner.graph_exec) { cudaGraphExecDestroy(owner.graph_exec); owner.graph_exec=nullptr; }
                if (owner.graph) { cudaGraphDestroy(owner.graph); owner.graph=nullptr; }
                owner.app.configuration.buffer=&owner.buffer;
                owner.app.configuration.inputBuffer=&owner.input_buffer;
                for (std::size_t i=1;i<inputs.size();++i) {
                    if (inputs[i]) cudaFree(inputs[i]);
                    if (outputs[i]) cudaFree(outputs[i]);
                }
            }
        } pool(resources,rotation.slots);
        for (std::size_t i=1;i<rotation.slots;++i) {
            cuda_check(cudaMalloc(&pool.inputs[i],bytes), "rotation allocate input");
            cuda_check(cudaMalloc(&pool.outputs[i],bytes), "rotation allocate output");
        }
        std::map<std::uintptr_t,std::size_t> input_index, output_index;
        std::vector<std::pair<std::uintptr_t,std::uintptr_t>> regions;
        for (std::size_t i=0;i<rotation.slots;++i) {
            const auto in=(std::uintptr_t)pool.inputs[i],out=(std::uintptr_t)pool.outputs[i];
            if (!in || !out || in > UINTPTR_MAX-bytes || out > UINTPTR_MAX-bytes ||
                !input_index.emplace(in,i).second || !output_index.emplace(out,i).second)
                throw std::runtime_error("rotation invalid or duplicate device address");
            regions.push_back({in,in+bytes});regions.push_back({out,out+bytes});
        }
        std::sort(regions.begin(),regions.end());
        for (std::size_t i=1;i<regions.size();++i) if (regions[i-1].second > regions[i].first)
            throw std::runtime_error("rotation overlapping device buffers");
        result.str("rotation_mode","multi_address_graph_random_unique_scaled_inputs");
        result.integer("rotation_pool_slots",rotation.slots);result.integer("rotation_pool_total_bytes",rotation.total_bytes);
        result.integer("rotation_input_pool_bytes",rotation.slots*bytes);result.integer("rotation_output_pool_bytes",rotation.slots*bytes);
        result.integer("rotation_device_l2_bytes",rotation_prop.l2CacheSize);result.integer("rotation_pool_ceiling_bytes",max_pool_bytes);
        result.str("rotation_sequence","graph node i uses slot i modulo pool_slots; equal visits per replay");
        result.str("rotation_cache_scope","working set exceeds twice L2; not a measurement of cache misses or proof of cold cache");
        auto fill_host = [&](std::size_t slot) {
            const double scale=rotation_pool::scale(slot);
            for (std::size_t b=0;b<o.batch;++b) for (std::size_t k=0;k<o.n;++k) {
                const auto v=scale*motivation::input_value(base[k],b,o.pattern);
                host[b*o.n+k]=make_float2(float(v.real()),float(v.imag()));
            }
        };
        std::set<std::pair<std::uint32_t,std::uint32_t>> representatives;
        std::size_t representative_index=count;
        for (std::size_t i=0;i<rotation.slots;++i) {
            fill_host(i);
            if(i==0) for(std::size_t j=0;j<count;++j) if(host[j].x!=0 || host[j].y!=0) { representative_index=j;break; }
            if(representative_index==count)throw std::runtime_error("rotation input cannot distinguish slots");
            std::uint32_t real_bits=0,imag_bits=0;
            std::memcpy(&real_bits,&host[representative_index].x,sizeof(real_bits));
            std::memcpy(&imag_bits,&host[representative_index].y,sizeof(imag_bits));
            if(!representatives.emplace(real_bits,imag_bits).second)throw std::runtime_error("rotation slot inputs are not distinguishable");
            cuda_check(cudaMemcpyAsync(pool.inputs[i],host.data(),bytes,cudaMemcpyHostToDevice,resources.stream), "rotation input H2D");
            cuda_check(cudaStreamSynchronize(resources.stream), "rotation synchronize input H2D");
        }
        result.add("rotation_distinct_input_representatives_verified","true");
        auto poison = [&]() {
            for (auto output : pool.outputs) cuda_check(cudaMemsetAsync(output,0xff,bytes,resources.stream), "rotation poison output");
            cuda_check(cudaStreamSynchronize(resources.stream), "rotation synchronize poison");
        };
        auto append_slot = [&](std::size_t slot) {
            VkFFTLaunchParams launch{};
            launch.inputBuffer=&pool.inputs[slot];launch.buffer=&pool.outputs[slot];
            auto r=VkFFTAppend(&resources.app,-1,&launch);vkfft_code=int(r);
            if(r!=VKFFT_SUCCESS) throw std::runtime_error("rotation VkFFTAppend failed with code "+std::to_string(vkfft_code));
        };
        phase="graph_build";
        const auto graph_begin=Clock::now();
        cuda_check(cudaStreamBeginCapture(resources.stream,cudaStreamCaptureModeThreadLocal), "rotation begin graph capture");
        try {
            cuda_check(cudaEventRecordWithFlags(resources.start,resources.stream,cudaEventRecordExternal), "rotation capture start");
            for (int i=0;i<o.graph_repeats;++i) append_slot(rotation_pool::slot(i,rotation.slots));
            cuda_check(cudaEventRecordWithFlags(resources.stop,resources.stream,cudaEventRecordExternal), "rotation capture stop");
            cuda_check(cudaStreamEndCapture(resources.stream,&resources.graph), "rotation end graph capture");
        } catch (...) {
            cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(resources.stream,&abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);throw;
        }
        cuda_check(cudaGraphInstantiateWithFlags(&resources.graph_exec,resources.graph,0), "rotation instantiate graph");
        result.add("graph_build_ms",number(std::chrono::duration<double,std::milli>(Clock::now()-graph_begin).count()));
        std::size_t node_count=0;cuda_check(cudaGraphGetNodes(resources.graph,nullptr,&node_count), "rotation graph node count");
        std::vector<cudaGraphNode_t> nodes(node_count);cuda_check(cudaGraphGetNodes(resources.graph,nodes.data(),&node_count), "rotation graph nodes");
        if (node_count != (std::size_t)o.graph_repeats+2) throw std::runtime_error("rotation unexpected graph node count");
        std::map<cudaGraphNode_t,std::size_t> index;
        for(std::size_t i=0;i<nodes.size();++i) index.emplace(nodes[i],i);
        std::vector<int> successor(nodes.size(),-1),slot_for_node(nodes.size(),-1),visits(rotation.slots,0);
        std::vector<cudaGraphNodeType> types(nodes.size());int root=-1;std::size_t kernels=0,events=0;
        for(std::size_t i=0;i<nodes.size();++i) {
            cuda_check(cudaGraphNodeGetType(nodes[i],&types[i]), "rotation graph node type");
            if (types[i]==cudaGraphNodeTypeKernel) {
                cudaKernelNodeParams params{};
                cuda_check(cudaGraphKernelNodeGetParams(nodes[i],&params), "rotation graph kernel parameters");
                if(!params.kernelParams || !params.kernelParams[0] || !params.kernelParams[1])
                    throw std::runtime_error("rotation graph kernel argument inspection unavailable");
                void* in=nullptr;void* out=nullptr;
                std::memcpy(&in,params.kernelParams[0],sizeof(in));std::memcpy(&out,params.kernelParams[1],sizeof(out));
                auto ii=input_index.find((std::uintptr_t)in),oi=output_index.find((std::uintptr_t)out);
                if(ii==input_index.end() || oi==output_index.end() || ii->second!=oi->second)
                    throw std::runtime_error("rotation graph kernel address mismatch");
                slot_for_node[i]=(int)ii->second;++visits[ii->second];++kernels;
            } else if(types[i]==cudaGraphNodeTypeEventRecord) ++events;
            else throw std::runtime_error("rotation graph contains a non-kernel/non-event node");
            std::size_t count_dep=0;cuda_check(cudaGraphNodeGetDependencies(nodes[i],nullptr,&count_dep), "rotation graph dependencies count");
            if (count_dep==0) { if(root!=-1)throw std::runtime_error("rotation graph has multiple roots");root=(int)i; }
            else {
                if(count_dep!=1)throw std::runtime_error("rotation graph is not a single chain");
                cudaGraphNode_t predecessor=nullptr;cuda_check(cudaGraphNodeGetDependencies(nodes[i],&predecessor,&count_dep), "rotation graph dependency");
                auto found=index.find(predecessor);
                if(found==index.end() || successor[found->second]!=-1)throw std::runtime_error("rotation graph branches");
                successor[found->second]=(int)i;
            }
        }
        if(kernels!=(std::size_t)o.graph_repeats || events!=2 || root<0 || types[root]!=cudaGraphNodeTypeEventRecord)
            throw std::runtime_error("rotation graph node protocol mismatch");
        int current=successor[root];
        for(int i=0;i<o.graph_repeats;++i) {
            if(current<0 || types[current]!=cudaGraphNodeTypeKernel || slot_for_node[current]!=(int)rotation_pool::slot(i,rotation.slots))
                throw std::runtime_error("rotation graph node order mismatch");
            current=successor[current];
        }
        if(current<0 || types[current]!=cudaGraphNodeTypeEventRecord || successor[current]!=-1)
            throw std::runtime_error("rotation graph terminal event mismatch");
        for(auto n:visits) if(n!=o.graph_repeats/(int)rotation.slots)throw std::runtime_error("rotation unequal slot visits");
        result.integer("graph_node_count",node_count);result.integer("graph_kernel_count",kernels);result.integer("graph_event_record_count",events);
        result.integer("graph_expected_kernel_count",o.graph_repeats);result.add("rotation_graph_kernel_arguments_and_order_verified","true");
        auto execute_graph = [&]() {
            cuda_check(cudaGraphLaunch(resources.graph_exec,resources.stream), "rotation launch graph");
            cuda_check(cudaStreamSynchronize(resources.stream), "rotation synchronize graph");
        };
        auto verify_pool = [&](bool& all) {
            std::ostringstream entries;entries<<'[';all=true;
            std::vector<float2> observed(count);
            for(std::size_t i=0;i<rotation.slots;++i) {
                if(i)entries<<',';
                fill_host(i);
                cuda_check(cudaMemcpyAsync(observed.data(),pool.inputs[i],bytes,cudaMemcpyDeviceToHost,resources.stream), "rotation read input");
                cuda_check(cudaStreamSynchronize(resources.stream), "rotation synchronize input read");
                bool unchanged=std::memcmp(observed.data(),host.data(),bytes)==0;
                cuda_check(cudaMemcpyAsync(observed.data(),pool.outputs[i],bytes,cudaMemcpyDeviceToHost,resources.stream), "rotation read output");
                cuda_check(cudaStreamSynchronize(resources.stream), "rotation synchronize output read");
                bool numeric=false;auto diagnostic=correctness(o,observed,ref,base,numeric,rotation_pool::scale(i));
                Object entry;entry.integer("slot",i);entry.add("input_unchanged",unchanged?"true":"false");
                entry.add("correctness",diagnostic);entries<<entry.finish();all &= unchanged && numeric;
            }
            entries<<']';Object summary;summary.add("passed",all?"true":"false");summary.integer("slots_checked",rotation.slots);
            summary.integer("elements_per_slot",count);summary.add("slots",entries.str());return summary.finish();
        };
        phase="rotation_initial_validation";poison();execute_graph();
        float first_ms=0;cuda_check(cudaEventElapsedTime(&first_ms,resources.start,resources.stop), "rotation first elapsed");
        if(!std::isfinite(first_ms)||first_ms<=0)throw std::runtime_error("rotation invalid first interval");
        result.add("rotation_first_graph_interval_ms",number(first_ms));
        bool initial_ok=false;result.add("rotation_initial_validation",verify_pool(initial_ok));
        if(!initial_ok)throw std::runtime_error("rotation initial full-pool validation failed");
        phase="warmup";const auto warmup_begin=Clock::now();int warmup_actual=0;double warmup_wall_ms=0;
        while(warmup_actual<o.warmup || warmup_wall_ms<o.warmup_ms) {
            execute_graph();++warmup_actual;warmup_wall_ms=std::chrono::duration<double,std::milli>(Clock::now()-warmup_begin).count();
        }
        result.integer("warmup_actual",warmup_actual);result.add("warmup_wall_ms",number(warmup_wall_ms));
        result.integer("warmup_forward_count",std::uint64_t(warmup_actual)*std::uint64_t(o.graph_repeats));
        phase="execution";
        for(int i=0;i<o.repeats;++i) {
            execute_graph();float ms=0;cuda_check(cudaEventElapsedTime(&ms,resources.start,resources.stop), "rotation timed elapsed");
            if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("rotation nonpositive event interval");
            raw_samples.push_back(ms);samples.push_back(ms/float(o.graph_repeats));
        }
        phase="rotation_final_validation";poison();execute_graph();
        float final_validation_ms=0;cuda_check(cudaEventElapsedTime(&final_validation_ms,resources.start,resources.stop), "rotation validation elapsed");
        if(!std::isfinite(final_validation_ms)||final_validation_ms<=0)throw std::runtime_error("rotation invalid final validation interval");
        result.add("rotation_final_validation_graph_interval_ms",number(final_validation_ms));
        bool final_ok=false;result.add("rotation_final_validation",verify_pool(final_ok));
        result.str("rotation_output_initialization","0xff_bytes_fp32_nan_before_initial_and_separate_post_timing_validation_graph");
        result.str("rotation_correctness_scope","initial and separate post-timing graph replays validated; timed replays not individually read back");
        passed=initial_ok && final_ok;result.str("status",passed?"ok":"correctness_failed");
'''


def prepare(generated, output):
    generated, output=Path(generated).resolve(),Path(output).resolve()
    if output==generated or generated in output.parents:
        raise ValueError('rotation output must be outside the frozen generated tree')
    if output.exists() and any(output.iterdir()): raise ValueError('rotation output must be new or empty')
    source=generated/'expanded_benchmark.cpp'
    if not source.is_file() or not (generated/'include/candidate.hpp').is_file(): raise ValueError('frozen expanded generated tree required')
    text=source.read_text(encoding='utf8')
    text='#include <set>\n#include "rotation_pool.hpp"\n'+text
    old='''    result.str("timing_scope", o.placement == "outofplace" ?
               "full_batch_forward_immutable_input_separate_output_same_stream_excluding_input_restore" :
               "one_full_batch_in_place_forward_same_stream_excluding_input_restore");'''
    text=once(text,old,'    result.str("timing_scope", "multi_address_graph_forward_same_stream_no_transfers_or_validation_inside_event_interval");')
    old='const std::vector<motivation::Complex>& base, bool& passed) {'
    text=once(text,old,'const std::vector<motivation::Complex>& base, bool& passed, double slot_scale = 1.0) {')
    text=once(text,'const auto expected = motivation::reference_value(ref[k], k, b, o.n, o.pattern);',
                   'const auto expected = slot_scale * motivation::reference_value(ref[k], k, b, o.n, o.pattern);')
    text=once(text,'batch_input2 += std::norm(motivation::input_value(base[k], b, o.pattern));',
                   'batch_input2 += std::norm(slot_scale * motivation::input_value(base[k], b, o.pattern));')
    start='        VkFFTLaunchParams launch{}; launch.buffer = &resources.buffer;'
    end='\n        }\n    } catch (const std::exception& error) {'
    if text.count(start)!=1 or text.count(end)!=1: raise ValueError('execution anchors missing or ambiguous')
    before,rest=text.split(start,1);_,after=rest.split(end,1)
    text=before+ROTATION+end+after
    # An original-header control must never silently accept candidate overrides.
    anchor='        Options o = options(argc, argv); bool all_passed = true;'
    text=once(text,anchor,anchor+'''
#ifdef VKFFT_STUDY_UNINSTRUMENTED
        if (expanded_candidate::enabled()) throw std::runtime_error("rotation original control forbids expanded request");
#endif
        if (o.study_mode != "execute" || o.placement != "outofplace" || o.timing_mode != "graph" || o.pattern != "random" || o.grid)
            throw std::runtime_error("rotation requires execute/outofplace/graph/random");
''')
    output.mkdir(parents=True,exist_ok=True)
    (output/'rotating_benchmark.cpp').write_text(text,encoding='utf8')
    for name in ('rotation_pool.hpp',): (output/name).write_bytes((HERE/name).read_bytes())
    (output/'CMakeLists.txt').write_bytes((HERE/'CMakeLists_rotation.txt').read_bytes())
    files={'expanded_benchmark_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    for name in ('rotating_benchmark.cpp','rotation_pool.hpp','CMakeLists.txt'):
        files[name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
    (output/'rotation_provenance.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf8')
    return files


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expanded-generated-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();print(json.dumps(prepare(a.expanded_generated_root,a.output),indent=2))
