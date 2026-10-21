#include <cuda_runtime.h>

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <limits>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"

namespace {

constexpr int kMaxExpansionOrder = 10;
constexpr int kBlocksize = 128;
constexpr int kMaxTerms = (kMaxExpansionOrder + 1) * (kMaxExpansionOrder + 2) / 2;
constexpr int kChildrenPerParent = 8;

__host__ __device__ __forceinline__ int termIndex(int degree, int order) {
    return degree * (degree + 1) / 2 + order;
}

__device__ __forceinline__ double signedOrderParity(int order_magnitude) {
    // checks order_magnitude is odd or even.
    return (order_magnitude & 1) ? -1.0 : 1.0;
}

__device__ void computePositiveRegularHarmonics(
    double dx, double dy, double dz, int p,
    double* __restrict__ out_real,
    double* __restrict__ out_imag
) {
    out_real[termIndex(0, 0)] = 1.0;
    out_imag[termIndex(0, 0)] = 0.0;

    const double r2 = dx * dx + dy * dy + dz * dz;

    for (int degree = 1; degree <= p; degree++) {
        const int diagonal = termIndex(degree, degree);
        const int previous_diagonal = termIndex(degree - 1, degree - 1);
        const double factor = -1.0 / (2.0 * degree);
        const double previous_real = out_real[previous_diagonal];
        const double previous_imag = out_imag[previous_diagonal];

        out_real[diagonal] = factor * (dx * previous_real - dy * previous_imag);
        out_imag[diagonal] = factor * (dx * previous_imag + dy * previous_real);

        const int subdiagonal = termIndex(degree, degree - 1);
        out_real[subdiagonal] = dz * previous_real;
        out_imag[subdiagonal] = dz * previous_imag;

        for (int order = degree - 2; order >= 0; order--) {
            const int current = termIndex(degree, order);
            const int previous_l = termIndex(degree - 1, order);
            const int previous_l2 = termIndex(degree - 2, order);
            const double denominator = static_cast<double>((degree + order) * (degree - order));
            const double z_scale = (2.0 * degree - 1.0) * dz;

            out_real[current] = (z_scale * out_real[previous_l] - r2 * out_real[previous_l2]) / denominator;
            out_imag[current] = (z_scale * out_imag[previous_l] - r2 * out_imag[previous_l2]) / denominator;
        }
    }
}

__device__ __forceinline__ void loadSignedMultipole(
    const double* __restrict__ multipole_real,
    const double* __restrict__ multipole_imag,
    size_t node_offset, int degree, int order,
    double& value_real, double& value_imag
) {
    const int magnitude = (order < 0) ? -order : order;
    const int index = termIndex(degree, magnitude);
    const double real = multipole_real[node_offset + index];
    const double imag = multipole_imag[node_offset + index];

    if (order >= 0) {
        value_real = real;
        value_imag = imag;
    } else {
        const double parity = signedOrderParity(magnitude);
        value_real = parity * real;
        value_imag = -parity * imag;
    }
}

__device__ __forceinline__ void loadSignedRegularConjugate(
    const double* __restrict__ regular_real,
    const double* __restrict__ regular_imag,
    int degree, int order,
    double& value_real, double& value_imag
) {
    const int magnitude = (order < 0) ? -order : order;
    const int index = termIndex(degree, magnitude);
    const double real = regular_real[index];
    const double imag = regular_imag[index];

    if (order >= 0) {
        value_real = real;
        value_imag = -imag;
    } else {
        const double parity = signedOrderParity(magnitude);
        value_real = parity * real;
        value_imag = parity * imag;
    }
}

__global__ void m2mKernel(
    int total_nodes, int p,
    int num_terms, int level_start,
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
        const int child_slot = static_cast<int>(threadIdx.x);
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

    for (int output_term = static_cast<int>(threadIdx.x);
         output_term < num_terms;
         output_term += static_cast<int>(blockDim.x)) {
            
            int degree = 0;
            while ((degree + 1) * (degree + 2) / 2 <= output_term) {
                degree++;
            }

            const int order = output_term - termIndex(degree, 0);

            double parent_real = 0.0;
            double parent_imag = 0.0;

            for (int child_slot = 0; child_slot < kChildrenPerParent; child_slot++) {
                const int child = first_child + child_slot;
                const size_t child_offset = static_cast<size_t>(child) * num_terms;

                for (int child_degree = 0; child_degree <= degree; child_degree++) {
                    const int shift_degree = degree - child_degree;

                    for (int child_order = -child_degree; child_order <= child_degree; child_order++) {
                        const int shift_order = order - child_order;

                        if (shift_order < -shift_degree ||
                            shift_order > shift_degree) {
                                continue;
                            }

                        double child_real, child_imag;
                        loadSignedMultipole(
                            d_M_real, d_M_imag,
                            child_offset, child_degree, child_order,
                            child_real, child_imag
                        );

                        double translation_real, translation_imag;
                        loadSignedRegularConjugate(
                            shift_real[child_slot], shift_imag[child_slot],
                            shift_degree, shift_order,
                            translation_real, translation_imag
                        );

                        parent_real += child_real * translation_real - child_imag * translation_imag;
                        parent_imag += child_real * translation_imag + child_imag * translation_real;
                    }
                }
            }

            const size_t parent_offset = static_cast<size_t>(parent) * num_terms;
            d_M_real[parent_offset + output_term] = parent_real;
            d_M_imag[parent_offset + output_term] = parent_imag;
         }
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
    
    const int num_terms = (p + 1) * (p + 2) / 2;
    const int max_level = level_bounds.size() - 2;

    for (int level = max_level - 1; level >= 0; level--) {
        const int level_start = level_bounds[level];
        const int level_end = level_bounds[level + 1];
        const int num_nodes_in_level = level_end - level_start;

        if (num_nodes_in_level == 0) continue;

        m2mKernel<<<num_nodes_in_level, kBlocksize>>>(
            total_nodes, p,
            num_terms, level_start,
            d_child_idx, d_center_x,
            d_center_y, d_center_z,
            d_M_real, d_M_imag
        );

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}