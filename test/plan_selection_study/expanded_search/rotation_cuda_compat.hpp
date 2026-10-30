#pragma once
#include <cstddef>
// Include after cuda_runtime_api.h. The null edge-data query intentionally
// fails closed if a graph has non-default dependency edge metadata.
#if !defined(CUDART_VERSION)
#error "CUDART_VERSION is required for the graph dependency API"
#endif
inline cudaError_t rotation_graph_dependencies(cudaGraphNode_t node, cudaGraphNode_t* dependencies,
                                               std::size_t* count) {
#if CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependencies(node, dependencies, nullptr, count);
#else
    return cudaGraphNodeGetDependencies(node, dependencies, count);
#endif
}
