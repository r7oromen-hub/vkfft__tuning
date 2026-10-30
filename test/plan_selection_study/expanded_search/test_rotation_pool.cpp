#include "rotation_pool.hpp"
#include <cassert>
#include <limits>
#include <set>
int main() {
    using namespace rotation_pool;
    Plan p{};
    assert(plan(1024,4096,1<<20,16,p) && p.slots==8 && p.total_bytes==16384);
    assert(p.total_bytes>2*4096);
    assert(!plan(1024,4096,16000,16,p));
    assert(!plan(1024,4096,1<<20,4,p));
    assert(!plan(1024,4096,1<<20,15,p));
    assert(!plan(0,4096,1<<20,16,p));
    assert(!plan(1,0,1<<20,16,p));
    assert(!plan(std::numeric_limits<std::size_t>::max(),1,1024,2,p));
    assert(plan(8192,4096,1<<20,2,p) && p.slots==2);
    std::set<double> unique;
    std::set<float> unique_float;
    for (std::size_t i=0;i<2048;++i) unique.insert(scale(i));
    for (std::size_t i=0;i<2048;++i) unique_float.insert(float(scale(i) / 8192.0));
    assert(unique.size()==2048 && scale(1)-scale(0) > 1e-5);
    assert(unique_float.size()==2048);
    for (std::size_t i=0;i<32;++i) assert(slot(i,8)==i%8);
    return 0;
}
