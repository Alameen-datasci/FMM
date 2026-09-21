#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"

constexpr int MAX_P = 10;
constexpr int MAX_TERMS = 66;

__global__ void p2mKernel(
    int total_nodes, int p,
    const int* __restrict__ d_node_start,
    const int* __restrict__ d_node_count,
    const int* __restrict__ d_child_idx,
    const uint32_t* __restrict__ d_particle_id,
    const double* __restrict__ d_pos_x,
    const double* __restrict__ d_pos_y,
    const double* __restrict__ d_pos_z,
    const double* __restrict__ d_masses,
    const double* __restrict__ d_center_x,
    const double* __restrict__ d_center_y,
    const double* __restrict__ d_center_z,
    double* __restrict__ d_M_real,
    double* __restrict__ d_M_imag
) {
    int node_idx = blockIdx.x;
    if (node_idx >= total_nodes) return;

    if (d_child_idx[node_idx] != -1) return;

    int tid = threadIdx.x;
    int num_terms = (p + 1) * (p + 2) / 2;

    __shared__ double s_M_real[MAX_TERMS];
    __shared__ double s_M_imag[MAX_TERMS];

    for (int i = tid; i < num_terms; i += blockDim.x) {
        s_M_real[i] = 0.0;
        s_M_imag[i] = 0.0;
    }

    __syncthreads();

    int parent_start = d_node_start[node_idx];
    int parent_count = d_node_count[node_idx];

    double cx = d_center_x[node_idx];
    double cy = d_center_y[node_idx];
    double cz = d_center_z[node_idx];

    for (int i = tid; i < parent_count; i += blockDim.x) {
        int pid = d_particle_id[parent_start + i];

        double dx = d_pos_x[pid] - cx;
        double dy = d_pos_y[pid] - cy;
        double dz = d_pos_z[pid] - cz;
        double mass = d_masses[pid];
        double r2 = dx*dx + dy*dy + dz*dz;

        double R_real[MAX_TERMS];
        double R_imag[MAX_TERMS];

        R_real[0] = 1.0;
        R_imag[0] = 0.0;

        for (int l = 1; l <= p; l++) {
            int idx_diag = l * (l + 1) / 2 + l;
            int idx_prev = l * (l - 1) / 2 + (l - 1);

            double factor = -1.0 / (2.0 * l);

            double Rp = R_real[idx_prev];
            double Ip = R_imag[idx_prev];

            R_real[idx_diag] = factor * (dx * Rp - dy * Ip);
            R_imag[idx_diag] = factor * (dx * Ip + dy * Rp);

            int idx_off_diag = l * (l + 1) / 2 + (l - 1);

            R_real[idx_off_diag] = dz * Rp;
            R_imag[idx_off_diag] = dz * Ip;

            if (l >= 2) {
                for (int m = l - 2; m >= 0; m--) {
                    int idx_lm = l * (l + 1) / 2 + m;
                    int prev1 = l * (l - 1) / 2 + m;
                    int prev2 = (l - 2) * (l - 1) / 2 + m;

                    double denom = static_cast<double>((l + m) * (l - m));
                    double inv_denom = 1.0 / denom;

                    R_real[idx_lm] = ((2.0 * l - 1) * dz * R_real[prev1] - r2 * R_real[prev2]) * inv_denom;
                    R_imag[idx_lm] = ((2.0 * l - 1) * dz * R_imag[prev1] - r2 * R_imag[prev2]) * inv_denom;
                }
            }
        }

        for (int k = 0; k < num_terms; k++) {
            atomicAdd(&s_M_real[k], mass * R_real[k]);
            atomicAdd(&s_M_imag[k], mass * R_imag[k]);
        }
    }

    __syncthreads();

    int node_offset = node_idx * num_terms;
    for (int i = tid; i < num_terms; i += blockDim.x) {
        d_M_real[node_offset + i] = s_M_real[i];
        d_M_imag[node_offset + i] = s_M_imag[i];
    }
}

__host__ void computeP2MCUDA(
    int total_nodes, int p,
    const int* d_node_start, const int* d_node_count,
    const int* d_child_idx, const uint32_t* d_particle_id,
    const double* d_pos_x, const double* d_pos_y, const double* d_pos_z,
    const double* d_masses, const double* d_center_x,
    const double* d_center_y, const double* d_center_z,
    double** d_M_real, double** d_M_imag
) {
    if (p > MAX_P) {
        std::cerr << "Error: Requested expansion order exceeds MAX_P (" << MAX_P << ")." << std::endl;
        exit(EXIT_FAILURE);
    }

    int num_terms = (p + 1) * (p + 2) / 2;
    size_t bytes_multipole = num_terms * total_nodes * sizeof(double);

    CUDA_CHECK(cudaMalloc((void**)d_M_real, bytes_multipole));
    CUDA_CHECK(cudaMalloc((void**)d_M_imag, bytes_multipole));

    dim3 block(128);
    dim3 grid(total_nodes);

    p2mKernel<<<grid, block>>>(
        total_nodes, p,
        d_node_start, d_node_count,
        d_child_idx, d_particle_id,
        d_pos_x, d_pos_y, d_pos_z,
        d_masses, d_center_x,
        d_center_y, d_center_z,
        *d_M_real, *d_M_imag
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
