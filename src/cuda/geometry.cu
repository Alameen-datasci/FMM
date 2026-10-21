#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"


namespace {

constexpr int kBlockSize = 256;

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
    const std::size_t local_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_idx >= static_cast<std::size_t>(num_nodes_in_level)) return;

    const int node = level_start_idx + static_cast<int>(local_idx);
    d_node_level[node] = current_level;

    if (current_level == 0) return;

    const int parent = d_node_parent[node];

    const int first_child = d_child_idx[parent];
    const int octant = node - first_child;

    const double child_half_width = 0.5 * d_half_width[parent];
    d_half_width[node] = child_half_width;

    const double shift_x = (octant & 1) ? child_half_width : -child_half_width;
    const double shift_y = (octant & 2) ? child_half_width : -child_half_width;
    const double shift_z = (octant & 4) ? child_half_width : -child_half_width;

    d_center_x[node] = d_center_x[parent] + shift_x;
    d_center_y[node] = d_center_y[parent] + shift_y;
    d_center_z[node] = d_center_z[parent] + shift_z;

}

} // namespace

__host__ void mapTreeGeometryCUDA(
    int total_nodes,
    const std::vector<int>& level_bounds,
    const int* d_child_idx, const int* d_node_parent,
    int* d_node_level, double* d_center_x,
    double* d_center_y, double* d_center_z,
    double* d_half_width, double root_cx,
    double root_cy, double root_cz,
    double root_half_width
) {
    if (total_nodes < 0) {
        throw std::invalid_argument("total_nodes must be non-negative.");
    }
    if (total_nodes == 0) return;

    if (
        level_bounds.size() < 2 ||
        level_bounds.size() - 1 >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        level_bounds.front() != 0 ||
        level_bounds.back() != total_nodes
    ) throw std::invalid_argument("Invalid level bounds for tree geometry.");

    if (d_child_idx == nullptr || d_node_parent == nullptr ||
        d_node_level == nullptr || d_center_x == nullptr ||
        d_center_y == nullptr || d_center_z == nullptr ||
        d_half_width == nullptr) {
        throw std::invalid_argument("Tree geometry device pointers must be non-null.");
    }
    if (!std::isfinite(root_cx) || !std::isfinite(root_cy) ||
        !std::isfinite(root_cz) || !std::isfinite(root_half_width) ||
        root_half_width < 0.0) {
        throw std::invalid_argument(
            "Root center must be finite and root_half_width must be finite and non-negative.");
    }

    for (std::size_t i = 1; i < level_bounds.size(); i++) {
        if (level_bounds[i] < level_bounds[i - 1]) {
            throw std::invalid_argument("Tree level bounds must be nondecreasing.");
        }
    }

    CUDA_CHECK(cudaMemcpy(d_center_x, &root_cx, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_center_y, &root_cy, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_center_z, &root_cz, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_half_width, &root_half_width, sizeof(double), cudaMemcpyHostToDevice));

    const int num_levels = static_cast<int>(level_bounds.size() - 1);

    for (int level = 0; level < num_levels; level++) {
        const int begin = level_bounds[level];
        const int count = level_bounds[level + 1] - begin;

        if (count == 0) continue;

        const int blocks = 1 + (count - 1) / kBlockSize;
        computeNodeGeometryKernel<<<blocks, kBlockSize>>>(
            d_node_parent, d_child_idx,
            d_center_x, d_center_y,
            d_center_z, d_half_width,
            d_node_level, begin, count,
            level
        );

        CUDA_CHECK(cudaGetLastError());
    }
}
