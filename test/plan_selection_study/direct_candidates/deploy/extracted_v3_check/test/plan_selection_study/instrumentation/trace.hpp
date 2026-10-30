#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vkfft_trace {
using Clock = std::chrono::steady_clock;
inline bool enabled = false, describe_only = false;
struct Span { int id, parent, axis, upload; std::string stage; int64_t start, end = 0; };
struct Kernel { int axis, upload; std::string raw, canonical, source_hash, compiler_input; std::size_t source_bytes; std::string expected_options,actual_options="null"; bool options_match=true; };
inline std::vector<Span> spans;
inline std::vector<int> stack;
inline std::vector<Kernel> kernels;
inline int64_t epoch_ns = 0;
inline double empty_scope_ns = 0, clock_pair_ns = 0;
inline int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
inline std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << char(c);
    }
    out << '"'; return out.str();
}
struct Object {
    std::ostringstream out; bool first = true;
    Object() { out << '{'; }
    void add(const std::string& key, const std::string& raw) { if (!first) out << ','; first = false; out << quote(key) << ':' << raw; }
    template<class T> void num(const std::string& key, T value) { std::ostringstream s; s << std::setprecision(17) << value; add(key, s.str()); }
    void str(const std::string& key, const std::string& value) { add(key, quote(value)); }
    std::string json() const { return out.str() + '}'; }
};
template<class T> std::string array(const T* values, std::size_t size) {
    std::ostringstream out; out << '[';
    for (std::size_t i = 0; i < size; ++i) { if (i) out << ','; out << values[i]; }
    out << ']'; return out.str();
}
struct Scope {
    int id = -1;
    Scope(const char* stage, const void*, int axis = -1, int upload = -1) {
        if (!enabled) return;
        id = int(spans.size());
        spans.push_back({id, stack.empty() ? -1 : stack.back(), axis, upload, stage, now_ns() - epoch_ns, 0});
        stack.push_back(id);
    }
    void finish() {
        if (id < 0) return;
        spans[id].end = now_ns() - epoch_ns;
        if (stack.empty() || stack.back() != id) std::terminate();
        stack.pop_back(); id = -1;
    }
    ~Scope() { finish(); }
};
template<class F> auto invoke(const char* stage, const void* app, int axis, int upload, F&& function) -> decltype(function()) {
    Scope scope(stage, app, axis, upload); return function();
}
inline void reset(bool describe = false) { enabled = false; spans.clear(); stack.clear(); kernels.clear(); epoch_ns = now_ns(); describe_only = describe; enabled = true; }
inline void pause() { enabled = false; }
inline std::size_t count_stage(const std::string& stage) { return std::count_if(spans.begin(),spans.end(),[&](const Span& span){return span.stage==stage;}); }
inline void measure_overhead() {
    std::vector<int64_t> pairs; pairs.reserve(1001);
    for (int i = 0; i < 1001; ++i) { auto a = now_ns(); pairs.push_back(now_ns() - a); }
    std::sort(pairs.begin(), pairs.end()); clock_pair_ns = double(pairs[pairs.size()/2]);
    reset(); spans.reserve(10000); const auto start = now_ns();
    for (int i = 0; i < 10000; ++i) { Scope scope("overhead_empty_scope", nullptr); }
    empty_scope_ns = double(now_ns() - start) / 10000; pause(); spans.clear(); stack.clear();
}
inline uint32_t rotate(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
inline std::string sha256(const char* data, std::size_t length) {
    static constexpr uint32_t constants[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::vector<unsigned char> bytes(data, data + length); bytes.push_back(0x80);
    while (bytes.size() % 64 != 56) bytes.push_back(0);
    const uint64_t bits = uint64_t(length) * 8;
    for (int shift = 56; shift >= 0; shift -= 8) bytes.push_back(static_cast<unsigned char>(bits >> shift));
    std::array<uint32_t,8> hash={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (std::size_t offset = 0; offset < bytes.size(); offset += 64) {
        uint32_t w[64];
        for (int i=0;i<16;++i) w[i]=(uint32_t(bytes[offset+4*i])<<24)|(uint32_t(bytes[offset+4*i+1])<<16)|(uint32_t(bytes[offset+4*i+2])<<8)|bytes[offset+4*i+3];
        for(int i=16;i<64;++i){auto s0=rotate(w[i-15],7)^rotate(w[i-15],18)^(w[i-15]>>3);auto s1=rotate(w[i-2],17)^rotate(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
        auto a=hash[0],b=hash[1],c=hash[2],d=hash[3],e=hash[4],f=hash[5],g=hash[6],h=hash[7];
        for(int i=0;i<64;++i){auto s1=rotate(e,6)^rotate(e,11)^rotate(e,25);auto ch=(e&f)^((~e)&g);auto t1=h+s1+ch+constants[i]+w[i];auto s0=rotate(a,2)^rotate(a,13)^rotate(a,22);auto maj=(a&b)^(a&c)^(b&c);auto t2=s0+maj;h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
        hash[0]+=a;hash[1]+=b;hash[2]+=c;hash[3]+=d;hash[4]+=e;hash[5]+=f;hash[6]+=g;hash[7]+=h;
    }
    std::ostringstream out; out<<std::hex<<std::setfill('0'); for(auto value:hash)out<<std::setw(8)<<value; return out.str();
}
template<class App, class Axis> std::string descriptor(App* app, Axis* axis, bool canonical) {
    const auto& s = axis->specializationConstants; const auto& c = app->configuration; Object d;
    d.num("axis",s.axis_id);d.num("upload",s.axis_upload_id);d.num("FFTdim",c.FFTdim);
    d.num("actualNumBatches",app->actualNumBatches);d.num("resolved_numberBatches",c.numberBatches);
    d.num("numAxisUploads",s.numAxisUploads);d.num("fftDim",s.fftDim.data.i);d.num("fft_dim_full",s.fft_dim_full.data.i);
    d.num("precision",s.precision);d.num("complexSize",s.complexSize);d.num("inverse",s.inverse);d.num("actualInverse",s.actualInverse);
    int64_t local[3]={s.localSize[0].data.i,s.localSize[1].data.i,s.localSize[2].data.i}; d.add("localSize",array(local,3));
    uint64_t block[4]={axis->axisBlock[0],axis->axisBlock[1],axis->axisBlock[2],axis->axisBlock[3]};
    if(canonical && s.axisSwapped)std::swap(block[0],block[1]);
    d.add("axisBlock",array(block,4)); if(!canonical)d.num("axisSwapped",s.axisSwapped);
    d.num("groupedBatch",axis->groupedBatch);d.num("numStages",s.numStages);d.add("stageRadix",array(s.stageRadix,std::size_t(s.numStages)));
    d.num("logical_registers_per_thread",s.registers_per_thread);d.num("min_logical_registers_per_thread",s.min_registers_per_thread);
    d.add("registers_per_radix",array(s.registers_per_thread_per_radix,33));d.num("usedLocRegs",s.usedLocRegs);d.num("registerBoost",s.registerBoost);
    d.num("dynamic_shared_bytes",s.usedSharedMemory.data.i);d.num("sharedMemSize",s.sharedMemSize);d.num("sharedMemSizePow2",s.sharedMemSizePow2);
    d.num("LUT",s.LUT);d.num("LUT_4step",s.LUT_4step);d.num("reorderFourStep",s.reorderFourStep);
    d.num("readToRegisters",s.readToRegisters);d.num("writeFromRegisters",s.writeFromRegisters);d.num("stridedSharedLayout",s.stridedSharedLayout);
    d.num("inputNumberByteSize",s.inputNumberByteSize);d.num("outputNumberByteSize",s.outputNumberByteSize);
    d.num("inputOffset",s.inputOffset.data.i);d.num("outputOffset",s.outputOffset.data.i);
    d.num("performR2C",s.performR2C);d.num("performDCT",s.performDCT);d.num("performDST",s.performDST);
    d.num("useRader",s.useRader);d.num("useBluesteinFFT",s.useBluesteinFFT);d.num("bufferLUTSize",axis->bufferLUTSize);
    const auto* plan = app->localFFTPlan;
    if(plan){d.add("axisSplit",array(plan->axisSplit[s.axis_id],4));d.add("actualFFTSizePerAxis",array(plan->actualFFTSizePerAxis[s.axis_id],4));}
    std::vector<int64_t> input,output;for(std::size_t i=0;i<c.FFTdim+2;++i){input.push_back(s.inputStride[i].data.i);output.push_back(s.outputStride[i].data.i);}
    d.add("inputStride",array(input.data(),input.size()));d.add("outputStride",array(output.data(),output.size()));
    d.num("isInputFormatted",c.isInputFormatted);d.num("isOutputFormatted",c.isOutputFormatted);
    return d.json();
}
template<class App,class Axis> void capture_precompile(App* app,Axis* axis) {
    Scope scope("precompile_snapshot_and_source_hash",app,axis->specializationConstants.axis_id,axis->specializationConstants.axis_upload_id);
    const char* source=axis->specializationConstants.code0;if(!source)throw std::runtime_error("Missing generated source before compilation");
    const auto bytes=std::strlen(source);const auto hash=sha256(source,bytes);Object input;
    input.str("function",axis->VkFFTFunctionName);input.str("source_sha256",hash);
    input.str("architecture_option","--gpu-architecture=sm_"+std::to_string(app->configuration.computeCapabilityMajor)+std::to_string(app->configuration.computeCapabilityMinor));
    const std::string options="["+quote("--gpu-architecture=sm_"+std::to_string(app->configuration.computeCapabilityMajor)+std::to_string(app->configuration.computeCapabilityMinor))+"]";
    input.add("options",options);input.str("options_provenance","restricted FP32 compiler path; full mode verifies against actual NVRTC option array");
#ifdef VKFFT_ORIGINAL_COMPILE_HEADER_SHA
    input.str("original_compile_header_sha256",VKFFT_ORIGINAL_COMPILE_HEADER_SHA);
#endif
    input.str("binary_kind","CUBIN_for_CUDA_11.3_or_newer");input.num("backend",1);
    kernels.push_back({axis->specializationConstants.axis_id,axis->specializationConstants.axis_upload_id,
                      descriptor(app,axis,false),descriptor(app,axis,true),hash,input.json(),bytes,options,"null",true});
}
template<class Axis> void capture_actual_options(Axis* axis,int count,char** options){
    std::ostringstream actual;actual<<'[';for(int i=0;i<count;++i){if(i)actual<<',';actual<<quote(options[i]);}actual<<']';
    for(auto&k:kernels)if(k.axis==axis->specializationConstants.axis_id&&k.upload==axis->specializationConstants.axis_upload_id){k.actual_options=actual.str();k.options_match=k.actual_options==k.expected_options;if(!k.options_match)throw std::runtime_error("Actual NVRTC options differ from restricted describe compiler metadata");return;}
    throw std::runtime_error("No precompile snapshot for actual NVRTC options");
}
inline std::string kernels_json(){std::ostringstream out;out<<'[';for(std::size_t i=0;i<kernels.size();++i){if(i)out<<',';const auto&k=kernels[i];Object j;j.num("axis",k.axis);j.num("upload",k.upload);j.add("raw_descriptor",k.raw);j.add("canonical_descriptor",k.canonical);j.str("source_sha256",k.source_hash);j.num("source_bytes",k.source_bytes);j.add("compiler_input",k.compiler_input);j.add("actual_compiler_options",k.actual_options);j.add("compiler_options_match",k.options_match?"true":"false");out<<j.json();}out<<']';return out.str();}
template<class App> std::string validate(App* app){bool all=true;std::ostringstream pairs;pairs<<'[';for(std::size_t i=0;i<kernels.size();++i){if(i)pairs<<',';const auto&k=kernels[i];auto post=descriptor(app,&app->localFFTPlan->axes[k.axis][k.upload],true);bool equal=post==k.canonical;all&=equal;Object p;p.num("axis",k.axis);p.num("upload",k.upload);p.add("equal",equal?"true":"false");p.add("precompile_canonical",k.canonical);p.add("postinit_canonical",post);pairs<<p.json();}pairs<<']';Object result;result.add("all_match",all&&!kernels.empty()?"true":"false");result.add("pairs",pairs.str());return result.json();}
inline std::string trace_json(){
    struct Total{std::size_t calls=0;double inclusive=0,exclusive=0;};std::map<std::string,Total> totals;std::vector<int64_t> child(spans.size(),0);
    for(const auto&s:spans)if(s.parent>=0)child[s.parent]+=s.end-s.start;
    std::ostringstream items;items<<'[';for(std::size_t i=0;i<spans.size();++i){if(i)items<<',';const auto&s=spans[i];const double inc=double(s.end-s.start)/1e6,exc=double(s.end-s.start-child[i])/1e6;auto&t=totals[s.stage];++t.calls;t.inclusive+=inc;t.exclusive+=exc;Object j;j.num("id",s.id);j.num("parent_id",s.parent);j.str("stage",s.stage);j.num("axis",s.axis);j.num("upload",s.upload);j.num("start_ns",s.start);j.num("end_ns",s.end);j.num("inclusive_ms",inc);j.num("exclusive_ms",exc);items<<j.json();}items<<']';
    Object aggregate,counters;for(const auto&entry:totals){Object t;t.num("calls",entry.second.calls);t.num("inclusive_ms",entry.second.inclusive);t.num("exclusive_ms",entry.second.exclusive);aggregate.add(entry.first,t.json());}
    for(const auto&name:std::vector<std::string>{"nvrtc_compile","module_load","driver_kernel_launch"})counters.num(name,totals[name].calls);
    Object overhead;overhead.num("empty_scope_mean_ns",empty_scope_ns);overhead.num("clock_pair_median_ns",clock_pair_ns);
    Object result;result.add("spans",items.str());result.add("stage_totals",aggregate.json());result.add("counters",counters.json());result.add("overhead",overhead.json());result.str("scope","initialize plus separate cleanup; disabled during execution");result.str("exclusive_definition","inclusive minus direct-child inclusive; sum exclusive across nested tree avoids double counting");return result.json();
}
}
#define VKFFT_TRACE_JOIN_I(a,b) a##b
#define VKFFT_TRACE_JOIN(a,b) VKFFT_TRACE_JOIN_I(a,b)
#define VKFFT_TRACE_SCOPE(stage,app,axis,upload) vkfft_trace::Scope VKFFT_TRACE_JOIN(vkfft_trace_scope_,__LINE__)(stage,app,int(axis),int(upload))
