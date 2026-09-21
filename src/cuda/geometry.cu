#include <cuda_runtime.h>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"

__global__ void computeNodeGeometryKernel(
    const int* __restrict__ d_node_parent,
    const int* __restrict__ d_child_idx,
    double* __restrict__ d_center_x,
    double* __restrict__ d_center_y,
    double* __restrict__ d_center_z,
    double* __restrict__ d_half_width,
    int* __restrict__ d_node_level,
    int level_start_idx,
    int num_nodes_in_level,
    int current_level
) {
    int idx = level_start_idx + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= level_start_idx + num_nodes_in_level) return;

    d_node_level[idx] = current_level;

    if (current_level == 0) return;

    int parent_idx = d_node_parent[idx];

    int first_child = d_child_idx[parent_idx];
    int octant = idx - first_child;

    double parent_cx = d_center_x[parent_idx];
    double parent_cy = d_center_y[parent_idx];
    double parent_cz = d_center_z[parent_idx];

    double child_half_width = d_half_width[parent_idx] * 0.5;
    d_half_width[idx] = child_half_width;

    double shift_x = (octant & 1) ? child_half_width : -child_half_width;
    double shift_y = (octant & 2) ? child_half_width : -child_half_width;
    double shift_z = (octant & 4) ? child_half_width : -child_half_width;

    d_center_x[idx] = parent_cx + shift_x;
    d_center_y[idx] = parent_cy + shift_y;
    d_center_z[idx] = parent_cz + shift_z;
}

__host__ void mapTreeGeometryCUDA(
    int total_nodes,
    const std::vector<int>& level_bounds,
    int* d_child_idx, int* d_node_parent, int* d_node_level,
    double* d_center_x, double* d_center_y, double* d_center_z,
    double* d_half_width, double root_cx, double root_cy,
    double root_cz, double root_half_width
) {
    CUDA_CHECK(cudaMemcpy(d_center_x, &root_cx, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_center_y, &root_cy, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_center_z, &root_cz, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_half_width, &root_half_width, sizeof(double), cudaMemcpyHostToDevice));

    int num_levels = level_bounds.size() - 1;

    for (int current_level = 0; current_level < num_levels; current_level++) {
        
        int level_start_idx = level_bounds[current_level];
        int next_level_start = level_bounds[current_level + 1];
        int num_nodes_in_level = next_level_start - level_start_idx;

        if (num_nodes_in_level == 0) continue;

        dim3 block(256);
        int num_blocks = (num_nodes_in_level + block.x - 1) / block.x;
        dim3 grid(num_blocks);

        computeNodeGeometryKernel<<<grid, block>>>(
            d_node_parent, d_child_idx,
            d_center_x, d_center_y, d_center_z, d_half_width,
            d_node_level,
            level_start_idx, num_nodes_in_level, current_level
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
