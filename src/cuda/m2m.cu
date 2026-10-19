#include <cuda_runtime.h>

#include <vector>
#include <stdexcept>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"

namespace {

constexpr int kMaxExpansionOrder = 10;
constexpr int kBlocksize = 128;
constexpr int kMaxTerms = (kMaxExpansionOrder + 1) * (kMaxExpansionOrder + 2) / 2;
constexpr int kChildrenPerParent = 8;

__global__ void m2mKernel(
    int num_nodes_level, int p,
    int level_start,
    const int* __restrict__ d_child_idx,
    const double* __restrict__ d_center_x,
    const double* __restrict__ d_center_y,
    const double* __restrict__ d_center_z,
    double* __restrict__ d_M_real,
    double* __restrict__ d_M_imag
) {
    const int parent = level_start + static_cast<int>(blockIdx.x);
    if (parent >= total_nodes) return;

    const int first_child = d_child_idx[parent];
    if (first_child == -1) return;

    __shared__ double shift_real[kChildrenPerParent][kMaxTerms];
    __shared__ double shift_imag[kChildrenPerParent][kMaxTerms];

    if (threadIdx.x < kChildrenPerParent) {
        const int child_slot = static_cast<int>(threadidx.x);
        const int child = first_child + child_slot;
        const double dx = d_center_x[child] - d_center_x[parent];
        const double dy = d_center_y[child] - d_center_y[parent];
        const double dz = d_center_z[child] - d_center_z[parent];

        computePositiveRegularHarmonics(
            dx, dy, dz, p,
            shift_real[child_slot], shift_imag[child_slot]
        );
    }

    __syncthreads();
}

} // namespace

__host__ void computeM2MCUDA(
    int total_nodes, int p,
    const std::vector<int>& level_bounds,
    const int* d_child_idx,
    const double* d_center_x,
    const double* d_center_y,
    const double* d_center_z,
    double* d_M_real,
    double* d_M_imag
) {
    if (total_nodes < 0) {
        throw std::invalid_argument("total_nodes must be non-negative.");
    }
    if (p < 0 || p > kMaxExpansionOrder) {
        throw std::invalid_argument("p must be between 0 and 10.");
    }
    if (total_nodes == 0) {
        return;
    }
    if (d_child_idx == nullptr || d_center_x == nullptr ||
        d_center_y == nullptr || d_center_z == nullptr ||
        d_M_real == nullptr || d_M_imag == nullptr) {
        throw std::invalid_argument("M2M device pointers must be non-null.");
    }
    if (level_bounds.size() < 2) {
        throw std::invalid_argument("level_bounds must contain a start and end.");
    }
    if (level_bounds.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Too many tree levels for the current int indexing.");
    }
    if (level_bounds.front() != 0 || level_bounds.back() != total_nodes) {
        throw std::invalid_argument(
            "level_bounds must start at zero and end at total_nodes.");
    }
    for (std::size_t i = 1; i < level_bounds.size(); ++i) {
        if (level_bounds[i] < level_bounds[i - 1]) {
            throw std::invalid_argument(
                "level_bounds must be nondecreasing.");
        }
    }
    
    const int max_terms = (p + 1) * (p + 2) / 2;
    const int max_level = level_bounds.size() - 2;

    for (int level = max_level; level >= 0; level--) {
        const int level_start = level_bounds[level];
        const int level_end = level_bounds[level + 1];
        const int num_nodes_in_level = level_end - level_start;

        if (num_nodes_in_level == 0) continue;

        m2mKernel<<<num_nodes_in_level, kBlocksize>>>(
            num_nodes_in_level,
            num_terms,
            p, level_start,
            d_child_idx, d_center_x,
            d_center_y, d_center_z,
            d_M_real, d_M_imag
        );

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}