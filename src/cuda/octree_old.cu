#include <cuda_runtime.h>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"

__device__ inline int find_lower_bound(
    const uint64_t* morton,
    int start, int end,
    uint64_t target_prefix,
    int shift_amount
) {
    int count = end - start;
    while (count > 0) {
        int step = count / 2;
        int mid = start + step;

        uint64_t mid_prefix = morton[mid] >> shift_amount;

        if (mid_prefix < target_prefix) {
            start = mid + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return start;
}

__global__ void splitOctreeNodesKernel(
    const uint64_t* __restrict__ morton,
    int* __restrict__ d_node_start,
    int* __restrict__ d_node_count,
    int* __restrict__ d_child_idx,
    int* __restrict__ d_node_parent,
    int* __restrict__ d_global_node_counter,
    int level_start_idx,
    int num_nodes_in_level,
    int current_level,
    int N_crit
) {
    int idx = level_start_idx + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= level_start_idx + num_nodes_in_level) return;

    int parent_start = d_node_start[idx];
    int parent_count = d_node_count[idx];

    if (parent_count <= N_crit) {
        d_child_idx[idx] = -1;
        return;
    }

    int first_child = atomicAdd(d_global_node_counter, 8);
    d_child_idx[idx] = first_child;

    int prefix_bits = (current_level + 1) * 3;
    int shift_amount = 63 - prefix_bits;

    uint64_t parent_prefix = morton[parent_start] >> (current_level * 3);
    uint64_t child_base_prefix = parent_prefix << 3;

    int current_start = parent_start;

    for (int i = 0; i < 8; i++) {
        int c_idx = first_child + i;
        d_node_parent[c_idx] = idx;

        int next_start;
        if (i == 7) {
            next_start = parent_start + parent_count;
        } else {
            uint64_t target_prefix = child_base_prefix | (i + 1);
            next_start = find_lower_bound(morton, current_start, parent_start + parent_count, target_prefix, shift_amount);
        }

        d_node_start[c_idx] = current_start;
        d_node_count[c_idx] = next_start - current_start;

        current_start = next_start;
    }
}

__host__ void buildTreeCUDA(
    const uint64_t* morton,
    size_t num_particles,
    int N_crit,
    int* d_node_start, int* d_node_count, int* d_child_idx, int* d_node_parent,
    int* out_total_nodes,
    std::vector<int>& out_level_bounds
) {
    int root_start = 0;
    int root_count = static_cast<int>(num_particles);
    int root_parent = -1;

    CUDA_CHECK(cudaMemcpy(d_node_start, &root_start, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_count, &root_count, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_node_parent, &root_parent, sizeof(int), cudaMemcpyHostToDevice));

    int level_start_idx = 0;
    int num_nodes_in_level = 1;
    int current_level = 0;

    int* d_global_node_counter;
    CUDA_CHECK(cudaMalloc((void**)&d_global_node_counter, sizeof(int)));
    int h_global_node_counter = 1;
    CUDA_CHECK(cudaMemcpy(d_global_node_counter, &h_global_node_counter, sizeof(int), cudaMemcpyHostToDevice));

    out_level_bounds.clear();
    out_level_bounds.push_back(level_start_idx); // if we use index 0 does it make any difference ?

    while (num_nodes_in_level > 0 && current_level < 21) {
        dim3 block(256);
        int num_blocks = (num_nodes_in_level + block.x - 1) / block.x;
        dim3 grid(num_blocks);

        splitOctreeNodesKernel<<<grid, block>>>(
            morton,
            d_node_start, d_node_count,
            d_child_idx, d_node_parent,
            d_global_node_counter,
            level_start_idx, num_nodes_in_level,
            current_level,
            N_crit
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int new_total_nodes;
        CUDA_CHECK(cudaMemcpy(&new_total_nodes, d_global_node_counter, sizeof(int), cudaMemcpyDeviceToHost));

        level_start_idx += num_nodes_in_level;
        num_nodes_in_level = new_total_nodes - level_start_idx;
        current_level++;

        if (num_nodes_in_level > 0) {
            out_level_bounds.push_back(level_start_idx);
        }
    }

    out_level_bounds.push_back(level_start_idx + num_nodes_in_level);

    if (out_total_nodes) {
        CUDA_CHECK(cudaMemcpy(out_total_nodes, d_global_node_counter, sizeof(int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_global_node_counter));
}
