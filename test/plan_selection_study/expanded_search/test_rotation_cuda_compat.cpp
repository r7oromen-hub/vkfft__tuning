#include <cstddef>
#include <cassert>
using cudaError_t = int;
struct Node {};
using cudaGraphNode_t = Node*;
static int calls=0;
static Node node, predecessor;
#if CUDART_VERSION >= 13000
struct cudaGraphEdgeData {};
cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t n,cudaGraphNode_t* dependencies,
                                         cudaGraphEdgeData* edge,std::size_t* count) {
    assert(edge==nullptr);
#else
cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t n,cudaGraphNode_t* dependencies,std::size_t* count) {
#endif
    assert(n==&node);assert(count);++calls;
    if (dependencies) { assert(*count==1);dependencies[0]=&predecessor; }
    else *count=1;
    return 37;
}
#include "rotation_cuda_compat.hpp"
int main() {
    std::size_t n=0;assert(rotation_graph_dependencies(&node,nullptr,&n)==37);assert(n==1);
    cudaGraphNode_t dep=nullptr;assert(rotation_graph_dependencies(&node,&dep,&n)==37);
    assert(dep==&predecessor && calls==2);return 0;
}
