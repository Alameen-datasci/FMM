#include "FMM_CUDA.hpp"
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define MAX_P 10
#define MAX_TERMS 66 // (p + 1) * (p + 2) / 2
#define MAX_LIST1 27 // 3 * 3 * 3
#define MAX_LIST2 189 // 27 * 8 - 27
#define MAX_2P_TERMS 231 // (2p + 1) * (2p + 2) / 2

__constant__ int d_l_map[MAX_TERMS];
__constant__ int d_m_map[MAX_TERMS];

__device__ inline uint64_t BitExpansionDevice(uint32_t v) {
        uint64_t x = v & 0x1FFFFF;
        x = (x | (x << 32)) & 0x1F00000000FFFFULL;
        x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
        x = (x | (x << 8)) & 0x10F00F00F00F00FULL;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3ULL;
        x = (x | (x << 2)) & 0x1249249249249249ULL;
        return x;
}

__device__ inline uint64_t MortonCodeDevice(uint32_t x, uint32_t y, uint32_t z) {
        return BitExpansionDevice(x) | (BitExpansionDevice(y) << 1) | (BitExpansionDevice(z) << 2);
}

__global__ void generateMortonCodesKernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    double xmin, double ymin, double zmin,
    double side,
    uint32_t resolution,
    size_t num_particles,
    uint64_t* __restrict__ morton,
    uint32_t* __restrict__ particle_id
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    double inv_side = (side > 0.0) ? (1.0 / side) : 0.0;
    uint32_t max_val = resolution - 1;

    for (size_t i = idx; i < num_particles; i += stride) {
        double nx = (pos_x[i] - xmin) * inv_side * resolution;
        double ny = (pos_y[i] - ymin) * inv_side * resolution;
        double nz = (pos_z[i] - zmin) * inv_side * resolution;

        uint32_t ux = static_cast<uint32_t>(fmin(fmax(nx, 0.0), static_cast<double>(max_val)));
        uint32_t uy = static_cast<uint32_t>(fmin(fmax(ny, 0.0), static_cast<double>(max_val)));
        uint32_t uz = static_cast<uint32_t>(fmin(fmax(nz, 0.0), static_cast<double>(max_val)));

        morton[i] = MortonCodeDevice(ux, uy, uz);
        particle_id[i] = static_cast<uint32_t>(i);
    }
}

__host__ void computeMortonCodesCUDA(
    const double* pos_x, const double* pos_y, const double* pos_z,
    double xmin, double ymin, double zmin, double side,
    uint32_t resolution, size_t num_particles,
    uint64_t* morton, uint32_t* particle_id
) {
    double* d_pos_x = nullptr, *d_pos_y = nullptr, *d_pos_z = nullptr;
    uint64_t* d_morton = nullptr;
    uint32_t* d_particle_id = nullptr;

    size_t bytes_pos = num_particles * sizeof(double);
    size_t bytes_morton = num_particles * sizeof(uint64_t);
    size_t bytes_particle_id = num_particles * sizeof(uint32_t);

    CUDA_CHECK(cudaMalloc((void**)&d_pos_x, bytes_pos));
    CUDA_CHECK(cudaMalloc((void**)&d_pos_y, bytes_pos));
    CUDA_CHECK(cudaMalloc((void**)&d_pos_z, bytes_pos));
    CUDA_CHECK(cudaMalloc((void**)&d_morton, bytes_morton));
    CUDA_CHECK(cudaMalloc((void**)&d_particle_id, bytes_particle_id));

    CUDA_CHECK(cudaMemcpy(d_pos_x, pos_x, bytes_pos, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, pos_y, bytes_pos, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, pos_z, bytes_pos, cudaMemcpyHostToDevice));

    dim3 block(256);
    int num_blocks = (num_particles + block.x - 1) / block.x;
    dim3 grid(num_blocks);

    generateMortonCodesKernel<<<grid, block>>>(
        d_pos_x, d_pos_y, d_pos_z,
        xmin, ymin, zmin, side,
        resolution, num_particles,
        d_morton, d_particle_id
    );

    CUDA_CHECK(cudaGetLastError());

    thrust::sort_by_key(thrust::device, d_morton, d_morton + num_particles, d_particle_id);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(morton, d_morton, bytes_morton, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(particle_id, d_particle_id, bytes_particle_id, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_pos_x));
    CUDA_CHECK(cudaFree(d_pos_y));
    CUDA_CHECK(cudaFree(d_pos_z));
    CUDA_CHECK(cudaFree(d_morton));
    CUDA_CHECK(cudaFree(d_particle_id));
}

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
    out_level_bounds.push_back(level_start_idx);

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

__host__ void init_lm_maps(int p) {
    int l_map[MAX_TERMS];
    int m_map[MAX_TERMS];
    for (int l = 0; l <= p; l++) {
        for (int m = 0; m <= l; m++) {
            int idx = l * (l + 1) / 2 + m;
            l_map[idx] = l;
            m_map[idx] = m;
        }
    }
    
    cudaMemcpyToSymbol(d_l_map, l_map, MAX_TERMS * sizeof(int));
    cudaMemcpyToSymbol(d_m_map, m_map, MAX_TERMS * sizeof(int));
}

__global__ void m2mKernel(
    int total_nodes, int p,
    const int* __restrict__ d_child_idx,
    const double* __restrict__ d_center_x,
    const double* __restrict__ d_center_y,
    const double* __restrict__ d_center_z,
    double* __restrict__ d_M_real,
    double* __restrict__ d_M_imag,
    int level_start
) {
    int parent_idx = level_start + blockIdx.x;

    if (parent_idx >= total_nodes) return;

    int first_child = d_child_idx[parent_idx];

    if (first_child == -1) return;

    int tid = threadIdx.x;
    int num_terms = (p + 1) * (p + 2) / 2;

    // shared memory arrays to store child's multipole
    __shared__ double s_child_M_real[MAX_TERMS * 8];
    __shared__ double s_child_M_imag[MAX_TERMS * 8];

    int total_child_terms = num_terms * 8;

    for (int i = tid; i < total_child_terms; i += blockDim.x) {
        int c_idx = i / num_terms;
        int term_idx = i % num_terms;

        int global_child_idx = first_child + c_idx;
        int global_offset = global_child_idx * num_terms + term_idx;

        s_child_M_real[i] = d_M_real[global_offset];
        s_child_M_imag[i] = d_M_imag[global_offset];
    }

    __syncthreads();

    for (int term_idx = tid; term_idx < num_terms; term_idx += blockDim.x) {
        int l = d_l_map[term_idx];
        int m = d_m_map[term_idx];

        double my_parent_term_real = 0.0;
        double my_parent_term_imag = 0.0;

        double cx = d_center_x[parent_idx];
        double cy = d_center_y[parent_idx];
        double cz = d_center_z[parent_idx];

        for (int i = 0; i < 8; i++) {
            int c_idx = first_child + i;

            double dx = d_center_x[c_idx] - cx;
            double dy = d_center_y[c_idx] - cy;
            double dz = d_center_z[c_idx] - cz;
            double r2 = dx*dx + dy*dy + dz*dz;

            double child_R_real[MAX_TERMS];
            double child_R_imag[MAX_TERMS];

            child_R_real[0] = 1.0;
            child_R_imag[0] = 0.0;

            for (int current_l = 1; current_l <= l; current_l++) {
                int idx_diag = current_l * (current_l + 1) / 2 + current_l;
                int idx_prev = current_l * (current_l - 1) / 2 + (current_l - 1);

                double factor = -1.0 / (2.0 * current_l);

                double Rp = child_R_real[idx_prev];
                double Ip = child_R_imag[idx_prev];

                child_R_real[idx_diag] = factor * (dx * Rp - dy * Ip);
                child_R_imag[idx_diag] = factor * (dy * Ip + dx * Rp);

                int idx_off_diag = current_l * (current_l + 1) / 2 + (current_l - 1);

                child_R_real[idx_off_diag] = dz * Rp;
                child_R_imag[idx_off_diag] = dz * Ip;

                if (current_l >= 2) {
                    for (int current_m = current_l - 2; current_m >= 0; current_m--) {
                        int idx_lm = current_l * (current_l + 1) / 2 + current_m;
                        int prev1 = current_l * (current_l - 1) / 2 + current_m;
                        int prev2 = (current_l - 2) * (current_l - 1) / 2 + current_m;

                        double denom = static_cast<double>((current_l + current_m) * (current_l - current_m));
                        double inv_denom = 1.0 / denom;

                        child_R_real[idx_lm] = ((2.0 * current_l - 1) * dz * child_R_real[prev1] - r2 * child_R_real[prev2]) * inv_denom;
                        child_R_imag[idx_lm] = ((2.0 * current_l - 1) * dz * child_R_imag[prev1] - r2 * child_R_imag[prev2]) * inv_denom;
                    }
                }
            }

            for (int j = 0; j <= l; j++) {
                for (int k = 0; k <= j; k++) {
                    int delta_m = m - k;
                    int abs_delta_m = abs(delta_m);
                    int child_term_idx = (l - j) * (l - j + 1) / 2 + abs_delta_m;
                    int shared_child_offset = i * num_terms + child_term_idx;

                    double C_M_real = s_child_M_real[shared_child_offset];
                    double C_M_imag = s_child_M_imag[shared_child_offset];

                    if (delta_m < 0) {
                        C_M_imag = -C_M_imag; // Conjugate

                        if (abs_delta_m % 2 != 0) { // (-1)^m {odd becomes negative}
                            C_M_real = -C_M_real;
                            C_M_imag = -C_M_imag;
                        }
                    }

                    int shift_term_idx = j * (j + 1) / 2 + k;
                    double R_real = child_R_real[shift_term_idx];
                    double R_imag = child_R_imag[shift_term_idx];

                    my_parent_term_real += (C_M_real * R_real - C_M_imag * R_imag);
                    my_parent_term_imag += (C_M_real * R_imag + C_M_imag * R_real);
                }
            }
        }

        // Write the fully accumulated term to global memory
        int global_parent_offset = parent_idx * num_terms + term_idx;
        d_M_real[global_parent_offset] = my_parent_term_real;
        d_M_imag[global_parent_offset] = my_parent_term_imag;
    }
}

__host__ void computeM2MCUDA(
    int total_nodes, int p,
    const int* d_child_idx, const double* d_center_x,
    const double* d_center_y, const double* d_center_z,
    double* d_M_real, double* d_M_imag,
    const std::vector<int>& level_bounds
) {
    int max_level = level_bounds.size() - 2;

    for (int level = max_level; level >= 0; level--) {
        int level_start = level_bounds[level];
        int level_end = level_bounds[level + 1];
        int num_nodes_in_level = level_end - level_start;

        if (num_nodes_in_level == 0) continue;

        dim3 block(128);
        dim3 grid(num_nodes_in_level);

        m2mKernel<<<grid, block>>>(
            total_nodes, p,
            d_child_idx, d_center_x,
            d_center_y, d_center_z,
            d_M_real, d_M_imag,
            level_start
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

__global__ void buildHierarchicalListsKernel(
    int level_start,
    int num_nodes_in_level,
    const int* __restrict__ d_node_parent,
    const int* __restrict__ d_child_idx,
    const double* __restrict__ d_center_x,
    const double* __restrict__ d_center_y,
    const double* __restrict__ d_center_z,
    const double* __restrict__ d_half_width,
    int* __restrict__ d_list1_count,
    int* __restrict__ d_list1,
    int* __restrict__ d_list2_count,
    int* __restrict__ d_list2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes_in_level) return;

    int target_idx = level_start + idx;
    int p_idx = d_node_parent[target_idx];

    double t_cx = d_center_x[target_idx];
    double t_cy = d_center_y[target_idx];
    double t_cz = d_center_z[target_idx];
    double t_h  = d_half_width[target_idx];

    int l1_count = 0;
    int l2_count = 0;
    int l1_offset = target_idx * MAX_LIST1;
    int l2_offset = target_idx * MAX_LIST2;

    int p_l1_count = d_list1_count[p_idx];
    int p_l1_offset = p_idx * MAX_LIST1;

    for (int i = 0; i < p_l1_count; i++) {
        int p_neighbor = d_list1[p_l1_offset + i];
        int pn_first_child = d_child_idx[p_neighbor];

        if (pn_first_child == -1) continue;

        for (int c = 0; c < 8; c++) {
            int source_idx = pn_first_child + c;

            double s_cx = d_center_x[source_idx];
            double s_cy = d_center_y[source_idx];
            double s_cz = d_center_z[source_idx];

            double dx = fabs(t_cx - s_cx);
            double dy = fabs(t_cy - s_cy);
            double dz = fabs(t_cz - s_cz);

            bool touching = fmax(dx, fmax(dy, dz)) <= (2.0 * t_h + 1e-10);

            if (touching) {
                if (l1_count < MAX_LIST1) {
                    d_list1[l1_offset + l1_count] = source_idx;
                    l1_count++;
                }
            } else {
                if (l2_count < MAX_LIST2) {
                    d_list2[l2_offset + l2_count] = source_idx;
                    l2_count++;
                }
            }
        }
    }

    d_list1_count[target_idx] = l1_count;
    d_list2_count[target_idx] = l2_count;
}

__global__ void m2lKernel(
    int level_start,
    int num_nodes_in_level,
    int p,
    const int* __restrict__ d_list2_count,
    const int* __restrict__ d_list2,
    const double* __restrict__ d_center_x,
    const double* __restrict__ d_center_y,
    const double* __restrict__ d_center_z,
    const double* __restrict__ d_M_real,
    const double* __restrict__ d_M_imag,
    double* __restrict__ d_L_real,
    double* __restrict__ d_L_imag
) {
    int target_idx = level_start + blockIdx.x;
    if (blockIdx.x >= num_nodes_in_level) return;

    int tid = threadIdx.x;
    int num_terms = (p + 1) * (p + 2) / 2;

    double t_cx = d_center_x[target_idx];
    double t_cy = d_center_y[target_idx];
    double t_cz = d_center_z[target_idx];

    __shared__ double s_L_real[MAX_TERMS];
    __shared__ double s_L_imag[MAX_TERMS];
    __shared__ double s_source_M_real[MAX_TERMS];
    __shared__ double s_source_M_imag[MAX_TERMS];

    for (int i = tid; i < num_terms; i += blockDim.x) {
        s_L_real[i] = 0.0;
        s_L_imag[i] = 0.0;
    }

    __syncthreads();

    int num_sources = d_list2_count[target_idx];
    int list2_offset = target_idx * MAX_LIST2;

    for (int i = 0; i < num_sources; i++) {
        int source_idx = d_list2[list2_offset + i];

        double s_cx = d_center_x[source_idx];
        double s_cy = d_center_y[source_idx];
        double s_cz = d_center_z[source_idx];

        int source_mem_offset = source_idx * num_terms;

        for (int j = tid; j < num_terms; j += blockDim.x) {
            s_source_M_real[j] = d_M_real[source_mem_offset + j];
            s_source_M_imag[j] = d_M_imag[source_mem_offset + j];
        }

        __syncthreads();

        // ======================================================
        // MATH PLACEHOLDER
        // ----------------

        double dx = s_cx - t_cx;
        double dy = s_cy - t_cy;
        double dz = s_cz - t_cz;
        double r2 = dx*dx + dy*dy + dz*dz;

        __shared__ double s_I_real[MAX_2P_TERMS];
        __shared__ double s_I_imag[MAX_2P_TERMS];

        if (tid == 0) {
            double inv_r = rsqrt(r2);
            double inv_r2 = inv_r * inv_r;

            s_I_real[0] = inv_r;
            s_I_imag[0] = 0.0;

            for (int l = 1; l <= 2 * p; l++) {
                int idx_diag = l * (l + 1) / 2 + l;
                int idx_prev = l * (l - 1) / 2 + (l - 1);

                double factor = (2.0 * l - 1) * inv_r2;

                double Ip_r = s_I_real[idx_prev];
                double Ip_i = s_I_imag[idx_prev];

                s_I_real[idx_diag] = factor * (dx * Ip_r - dy * Ip_i);
                s_I_imag[idx_diag] = factor * (dx * Ip_i + dy * Ip_r);

                int idx_off_diag = l * (l + 1) / 2 + (l - 1);
                s_I_real[idx_off_diag] = factor * dz * Ip_r;
                s_I_imag[idx_off_diag] = factor * dz * Ip_i;

                if (l >= 2) {
                    for (int m = l - 2; m >= 0; m--) {
                        int idx_lm = l * (l + 1) / 2 + m;
                        int prev1 = l * (l - 1) / 2 + m;
                        int prev2 = (l - 2) * (l - 1) / 2 + m;
                        
                        double term1_r = (2.0 * l - 1.0) * dz * s_I_real[prev1];
                        double term2_r = (l - 1.0 + m) * (l - 1.0 - m) * s_I_real[prev2];
                        s_I_real[idx_lm] = (term1_r - term2_r) * inv_r2;
                        
                        double term1_i = (2.0 * l - 1.0) * dz * s_I_imag[prev1];
                        double term2_i = (l - 1.0 + m) * (l - 1.0 - m) * s_I_imag[prev2];
                        s_I_imag[idx_lm] = (term1_i - term2_i) * inv_r2;
                    }
                }
            }
        }

        __syncthreads();

        if (tid < num_terms) {
            int j = d_l_map[tid];
            int k = d_m_map[tid];

            double my_L_term_real = 0.0;
            double my_L_term_imag = 0.0;

            for (int l = 0; l <= p; l++) {
                for (int m = 0; m <= l; m++) {

                    int lm_idx = l * (l + 1) / 2 + m;
                    double M_real = s_source_M_real[lm_idx];
                    double M_imag = s_source_M_imag[lm_idx];

                    int I_l = j + l;
                    int I_m = k - m;
                    int abs_I_m = abs(I_m);

                    int I_idx = I_l * (I_l + 1) / 2 + abs_I_m;

                    double Ir = s_I_real[I_idx];
                    double Ii = s_I_imag[I_idx];

                    if (I_m < 0) {
                        Ii = -Ii;
                        if (abs_I_m % 2 != 0) {
                            Ir = -Ir;
                            Ii = -Ii;
                        }
                    }

                    my_L_term_real += (M_real * Ir - M_imag * Ii);
                    my_L_term_imag += (M_real * Ii + M_imag * Ir);
                }
            }

            s_L_real[tid] += my_L_term_real;
            s_L_imag[tid] += my_L_term_imag;
        }
        // ======================================================

        __syncthreads();
    }

    int target_mem_offset = target_idx * num_terms;
    for (int i = tid; i < num_terms; i += blockDim.x) {
        atomicAdd(&d_L_real[target_mem_offset + i], s_L_real[i]);
        atomicAdd(&d_L_imag[target_mem_offset + i], s_L_imag[i]);
    }
}

__host__ void computeM2LCUDA(
    int total_nodes, int p,
    const int* d_child_idx, const int* d_node_parent,
    const double* d_center_x, const double* d_center_y,
    const double* d_center_z, const double* d_half_width,
    const double* d_M_real, const double* d_M_imag,
    double* d_L_real, double* d_L_imag,
    const std::vector<int>& level_bounds
) {
    if (level_bounds.size() <= 2) return;

    int* d_list1_count, *d_list1, *d_list2_count, *d_list2;
    CUDA_CHECK(cudaMalloc((void**)&d_list1_count, total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_list1, total_nodes * MAX_LIST1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_list2_count, total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_list2, total_nodes * MAX_LIST2 * sizeof(int)));

    int root_count = 1;
    int root_neighbor = 0;
    CUDA_CHECK(cudaMemcpy(&d_list1_count[0], &root_count, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&d_list1[0], &root_neighbor, sizeof(int), cudaMemcpyHostToDevice));

    int num_levels = level_bounds.size() - 1;

    for (int current_level = 1; current_level < num_levels; current_level++) {
        int level_start = level_bounds[current_level];
        int next_level_start = level_bounds[current_level + 1];
        int num_nodes_in_level = next_level_start - level_start;

        if (num_nodes_in_level == 0) continue;

        dim3 build_block(256);
        int num_build_blocks = (num_nodes_in_level + build_block.x - 1) / build_block.x;
        dim3 build_grid(num_build_blocks);

        buildHierarchicalListsKernel<<<build_grid, build_block>>>(
            level_start, num_nodes_in_level,
            d_node_parent, d_child_idx,
            d_center_x, d_center_y,
            d_center_z, d_half_width,
            d_list1_count, d_list1,
            d_list2_count, d_list2
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (current_level >= 2) {
            dim3 m2l_block(128);
            dim3 m2l_grid(num_nodes_in_level);

            m2lKernel<<<m2l_grid, m2l_block>>>(
                level_start, num_nodes_in_level,
                p, d_list2_count, d_list2,
                d_center_x, d_center_y, d_center_z,
                d_M_real, d_M_imag, d_L_real, d_L_imag
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaFree(d_list1_count));
    CUDA_CHECK(cudaFree(d_list1));
    CUDA_CHECK(cudaFree(d_list2_count));
    CUDA_CHECK(cudaFree(d_list2));
}