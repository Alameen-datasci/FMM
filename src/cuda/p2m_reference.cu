#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"

namespace {

constexpr int kMaxExpansionOrder = 10;
constexpr int kBlockSize = 128;
constexpr int kMaxTerms =
    (kMaxExpansionOrder + 1) * (kMaxExpansionOrder + 2) / 2;

// Term (l,m) is stored at l*(l+1)/2 + m, for 0 <= m <= l.
__global__ void p2mKernel(
    int total_nodes, int p, int num_terms,
    const int* __restrict__ node_start,
    const int* __restrict__ node_count,
    const int* __restrict__ child_idx,
    const uint32_t* __restrict__ particle_id,
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    const double* __restrict__ masses,
    const double* __restrict__ center_x,
    const double* __restrict__ center_y,
    const double* __restrict__ center_z,
    double* __restrict__ M_real,
    double* __restrict__ M_imag) {
    const int node = blockIdx.x;
    if (node >= total_nodes) return;

    const int tid = threadIdx.x;
    __shared__ double accum_real[kMaxTerms];
    __shared__ double accum_imag[kMaxTerms];

    for (int term = tid; term < num_terms; term += blockDim.x) {
        accum_real[term] = 0.0;
        accum_imag[term] = 0.0;
    }
    __syncthreads();

    // Every node receives a defined output. Internal nodes are populated later
    // by M2M; writing zero here also prevents uninitialized device memory from
    // leaking into that stage when it reads a node before accumulation.
    if (child_idx[node] == -1) {
        const int begin = node_start[node];
        const int count = node_count[node];
        const double cx = center_x[node];
        const double cy = center_y[node];
        const double cz = center_z[node];

        for (int particle = tid; particle < count; particle += blockDim.x) {
            const uint32_t pid = particle_id[begin + particle];
            const double dx = pos_x[pid] - cx;
            const double dy = pos_y[pid] - cy;
            const double dz = pos_z[pid] - cz;
            const double mass = masses[pid];
            const double r2 = dx * dx + dy * dy + dz * dz;

            // Regular solid harmonics for the project's triangular (l,m)
            // convention. These per-particle work arrays are bounded by p<=10.
            double R_real[kMaxTerms];
            double R_imag[kMaxTerms];
            R_real[0] = 1.0;
            R_imag[0] = 0.0;

            for (int l = 1; l <= p; ++l) {
                const int diagonal = l * (l + 1) / 2 + l; // R_{l}^{l}
                const int previous_diagonal = l * (l - 1) / 2 + (l - 1); // R_{l-1}^{l-1}
                const double factor = -1.0 / (2.0 * l);
                const double previous_real = R_real[previous_diagonal];
                const double previous_imag = R_imag[previous_diagonal];

                R_real[diagonal] = factor *
                    (dx * previous_real - dy * previous_imag);
                R_imag[diagonal] = factor *
                    (dx * previous_imag + dy * previous_real);

                const int subdiagonal = l * (l + 1) / 2 + (l - 1); // R_{l}^{l-1}
                R_real[subdiagonal] = dz * previous_real;
                R_imag[subdiagonal] = dz * previous_imag;

                for (int m = l - 2; m >= 0; --m) {
                    const int current = l * (l + 1) / 2 + m;
                    const int previous_l = l * (l - 1) / 2 + m;
                    const int previous_l2 = (l - 2) * (l - 1) / 2 + m;
                    const double denominator =
                        static_cast<double>((l + m) * (l - m));
                    const double scale = (2.0 * l - 1.0) * dz;

                    R_real[current] =
                        (scale * R_real[previous_l] -
                         r2 * R_real[previous_l2]) / denominator;
                    R_imag[current] =
                        (scale * R_imag[previous_l] -
                         r2 * R_imag[previous_l2]) / denominator;
                }
            }

            // First reduce contributions from this warp, then let one lane
            // update shared memory. This reduces shared atomic traffic by up
            // to a factor of warpSize compared with one atomic per particle.
            const unsigned active = __activemask();
            const int lane = threadIdx.x & (warpSize - 1);
            const int active_lanes = __popc(active);
            for (int term = 0; term < num_terms; ++term) {
                double real_sum = mass * R_real[term];
                double imag_sum = mass * R_imag[term];
                for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                    const double real_other =
                        __shfl_down_sync(active, real_sum, offset);
                    const double imag_other =
                        __shfl_down_sync(active, imag_sum, offset);
                    if (lane + offset < active_lanes) {
                        real_sum += real_other;
                        imag_sum += imag_other;
                    }
                }
                if (lane == 0) {
                    atomicAdd(&accum_real[term], real_sum);
                    atomicAdd(&accum_imag[term], imag_sum);
                }
            }
        }
    }

    __syncthreads();
    const size_t node_offset = static_cast<size_t>(node) * num_terms;
    for (int term = tid; term < num_terms; term += blockDim.x) {
        M_real[node_offset + term] = accum_real[term];
        M_imag[node_offset + term] = accum_imag[term];
    }
}

} // namespace

// Produces triangular complex multipole coefficients for every node. Compile
// this file in place of p2m.cu or p2m_old.cu; each exports computeP2MCUDA.
void computeP2MCUDA(
    int total_nodes, int p,
    const int* d_node_start,
    const int* d_node_count,
    const int* d_child_idx,
    const uint32_t* d_particle_id,
    const double* d_pos_x,
    const double* d_pos_y,
    const double* d_pos_z,
    const double* d_masses,
    const double* d_center_x,
    const double* d_center_y,
    const double* d_center_z,
    double** d_M_real,
    double** d_M_imag) {
    if (total_nodes < 0) {
        throw std::invalid_argument("total_nodes must be non-negative.");
    }
    if (p < 0 || p > kMaxExpansionOrder) {
        throw std::invalid_argument("p must be between 0 and 10.");
    }
    if (d_M_real == nullptr || d_M_imag == nullptr) {
        throw std::invalid_argument("output pointer arguments must be non-null.");
    }
    if (total_nodes == 0) {
        *d_M_real = nullptr;
        *d_M_imag = nullptr;
        return;
    }
    if (d_node_start == nullptr || d_node_count == nullptr ||
        d_child_idx == nullptr || d_particle_id == nullptr ||
        d_pos_x == nullptr || d_pos_y == nullptr || d_pos_z == nullptr ||
        d_masses == nullptr || d_center_x == nullptr || d_center_y == nullptr ||
        d_center_z == nullptr) {
        throw std::invalid_argument("P2M input device pointers must be non-null.");
    }

    const int num_terms = (p + 1) * (p + 2) / 2;
    const size_t max_elements = std::numeric_limits<size_t>::max() / sizeof(double);
    if (static_cast<size_t>(total_nodes) > max_elements /
            static_cast<size_t>(num_terms)) {
        throw std::overflow_error("P2M output allocation size overflows size_t.");
    }
    const size_t bytes = static_cast<size_t>(num_terms) *
                         static_cast<size_t>(total_nodes) * sizeof(double);

    *d_M_real = nullptr;
    *d_M_imag = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d_M_real), bytes));
    const cudaError_t imag_alloc =
        cudaMalloc(reinterpret_cast<void**>(d_M_imag), bytes);
    if (imag_alloc != cudaSuccess) {
        cudaFree(*d_M_real);
        *d_M_real = nullptr;
        CUDA_CHECK(imag_alloc);
    }

    p2mKernel<<<total_nodes, kBlockSize>>>(
        total_nodes, p, num_terms,
        d_node_start, d_node_count, d_child_idx, d_particle_id,
        d_pos_x, d_pos_y, d_pos_z, d_masses,
        d_center_x, d_center_y, d_center_z,
        *d_M_real, *d_M_imag);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
