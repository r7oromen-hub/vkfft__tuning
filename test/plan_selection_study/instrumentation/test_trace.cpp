#include "trace.hpp"
#include <iostream>
int main() {
    if(vkfft_trace::sha256("abc",3)!="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")return 1;
    if(vkfft_trace::sha256("",0)!="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")return 2;
    const std::string million(1000000,'a');
    if(vkfft_trace::sha256(million.data(),million.size())!="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0")return 5;
    vkfft_trace::measure_overhead();vkfft_trace::reset();
    {vkfft_trace::Scope parent("parent",nullptr);{vkfft_trace::Scope child("child",nullptr);}}
    vkfft_trace::pause();
    if(vkfft_trace::spans.size()!=2 || vkfft_trace::spans[1].parent!=0 || !vkfft_trace::stack.empty())return 3;
    if(vkfft_trace::spans[0].end<vkfft_trace::spans[1].end)return 4;
    std::cout<<vkfft_trace::trace_json()<<'\n';return 0;
}
