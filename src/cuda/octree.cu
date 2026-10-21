#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"

#include <cub/block/block_scan.cuh>

namespace {

constexpr int kMortonBits = 63;
constexpr int kOctantBits = 3;
constexpr int kMaxTreeDepth = kMortonBits / kOctantBits;
constexpr int kBlockSize = 256;

__device__ __forceinline__ int mortonOctantAtDepth(uint64_t code, int depth) {
    const int shift = kMortonBits - kOctantBits * (depth + 1);
    return static_cast<int>((code >> shift) & 0x7ULL);
}

__device__ __forceinline__ int lowerBoundOctant(
    const uint64_t* __restrict__ morton,
    int begin, int end,
    int depth, int target_octant
) {
    while (begin < end) {
        const int mid = begin + (end - begin) / 2;
        if (mortonOctantAtDepth(morton[mid], depth) < target_octant) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}

__global__ void splitOctreeLevelKernel(
    const uint64_t* __restrict__ d_morton,
    int* __restrict__ d_node_start,
    int* __restrict__ d_node_count,
    int* __restrict__ d_child_idx,
    int* __restrict__ d_node_parent,
    int* __restrict__ d_global_node_counter,
    int level_start_idx,
    int nodes_in_level,
    int depth, int N_crit
) {
    using BlockScan = cub::BlockScan<int, kBlockSize>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int block_child_base;
    
    const int lane = static_cast<int>(threadIdx.x);
    const int level_offset = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + lane;
    const bool valid = level_offset < nodes_in_level;
    const int node = level_start_idx + level_offset;
    
    const int child_slots = valid && d_node_count[node] > N_crit ? 8 : 0;
    int local_child_offset = 0;
    int children_in_block = 0;
    BlockScan(scan_storage).ExclusiveSum(child_slots, local_child_offset, children_in_block);
    
    if (lane == 0) {
        block_child_base = children_in_block == 0
        ? 0
        : atomicAdd(d_global_node_counter, children_in_block);
    }
    
    __syncthreads();
    
    if (!valid) return;
    
    if (child_slots == 0) {
        d_child_idx[node] = -1;
        return;
    }
    
    const int first_child = block_child_base + local_child_offset;
    const int parent_begin = d_node_start[node];
    const int parent_end = parent_begin + d_node_count[node];
    d_child_idx[node] = first_child;
    
    int cursor = parent_begin;
    for (int octant = 0; octant < 8; ++octant) {
        const int child = first_child + octant;
        const int next = (octant == 7)
        ? parent_end
        : lowerBoundOctant(d_morton, cursor, parent_end, depth, octant + 1);
        
        d_node_start[child] = cursor;
        d_node_count[child] = next - cursor;
        d_child_idx[child] = -1;
        d_node_parent[child] = node;
        cursor = next;
    }
}

} // namespace

__host__ void buildTreeCUDA(
    const uint64_t* d_morton,
    size_t num_particles,
    int N_crit,
    int* d_node_start, int* d_node_count,
    int* d_child_idx, int* d_node_parent,
    int* out_total_nodes,
    std::vector<int>& out_level_bounds
) {
    if (num_particles == 0) {
        throw std::invalid_argument("num_particles must be positive.");
    }
    if (num_particles > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "num_particles exceeds the current int-based tree index limit.");
    }
    if (N_crit <= 0) {
        throw std::invalid_argument("N_crit must be positive.");
    }
    if (d_morton == nullptr || d_node_start == nullptr ||
        d_node_count == nullptr || d_child_idx == nullptr ||
        d_node_parent == nullptr) {
        throw std::invalid_argument("Octree device input/output pointers must be non-null.");
    }

    const int root_start = 0;
    const int root_count = static_cast<int>(num_particles);
    const int root_parent = -1;

    CUDA_CHECK(cudaMemcpy(d_node_start, &root_start, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_count, &root_count, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_parent, &root_parent, sizeof(int), cudaMemcpyHostToDevice));

    int* d_global_node_count = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_global_node_count), sizeof(int)));
    const int initial_node_count = 1;
    CUDA_CHECK(cudaMemcpy(d_global_node_count, &initial_node_count, sizeof(int), cudaMemcpyHostToDevice));

    int level_start = 0;
    int nodes_in_level = 1;
    int total_nodes = initial_node_count;
    out_level_bounds.clear();
    out_level_bounds.push_back(level_start);

    for (int depth = 0; depth < kMaxTreeDepth && nodes_in_level > 0; ++depth) {
        const int blocks = 1 + (nodes_in_level - 1) / kBlockSize;
        splitOctreeLevelKernel<<<blocks, kBlockSize>>>(
            d_morton,
            d_node_start, d_node_count,
            d_child_idx, d_node_parent,
            d_global_node_count, level_start,
            nodes_in_level,
            depth, N_crit
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(
            &total_nodes, d_global_node_count, sizeof(int), cudaMemcpyDeviceToHost
        ));

        level_start += nodes_in_level;
        nodes_in_level = total_nodes - level_start;
        if (nodes_in_level > 0) out_level_bounds.push_back(level_start);
    }

    out_level_bounds.push_back(level_start + nodes_in_level);

    if (out_total_nodes != nullptr) {
        *out_total_nodes = total_nodes;
    }

    CUDA_CHECK(cudaFree(d_global_node_count));
}
