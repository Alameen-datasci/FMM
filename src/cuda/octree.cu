#include <cuda_runtime.h>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"

#include <cub/block/block_scan.cuh>

namespace {

constexpr int kMortonBits = 63;
constexpr int kOctantBits = 3;
constexpr int kMaxTreeDepth = kMortonBits / kOctantBits;
constexpr int kBlockSize = 256;

} // namespace

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

    const int lane = threadIdx.x;
    const int level_offset = blockIdx.x * blockDim.x + lane;
    const bool valid = level_offset < nodes_in_level;
    const int node = level_start_idx + level_offset;

    const int child_slots = valid && d_node_count[node] > N_crit ? 8 : 0;
    int local_child_offset = 0;
    int children_in_block = 0;
    BlockScan(scan_storage).ExclusiveSum(child_slots, local_child_offset, children_in_block);
}

__host__ void buildTreeCUDA(
    const uint64_t* d_morton,
    size_t num_particles,
    int N_crit,
    int* d_node_start, int* d_node_count,
    int* d_child_idx, int* d_node_parent,
    int* out_total_nodes,
    std::vector<int>& out_level_bounds
) {
    const int root_start = 0;
    const int root_count = static_cast<int>(num_particles);
    const int root_parent = -1;

    CUDA_CHECK(cudaMemcpy(d_node_start, &root_start, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_count, &root_count, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_root_parent, &root_parent, sizeof(int), cudaMemcpyHostToDevice));

    int* d_global_node_count = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_global_node_count), sizeof(int)));
    const int initial_node_count = 1;
    CUDA_CHECK(cudaMemcpy(d_global_node_count, initial_node_count, sizeof(int), cudaMemcpyHostToDevice));

    int level_start_idx = 0;
    int nodes_in_level = 1;
    int total_nodes = initial_node_count;
    out_level_bounds.clear();
    out_level_bounds.push_back(level_start_idx);

    for (int depth = 0; depht < kMaxTreeDepth && nodes_in_level > 0; ++depth) {
        const int blocks = (nodes_in_level + kBlockSize - 1) / kBlockSize;
        splitOctreeLevelKernel<<<blocks, kBlockSize>>>(
            d_morton,
            d_node_start, d_node_count,
            d_child_idx, d_node_parent,
            d_global_node_count, level_start,
            nodes_in_level,
            depth, N_crit
        );

        CUDA_CHECK(cudaGetLastError());
    }
}