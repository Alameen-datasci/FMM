#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <iostream>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "FMM_CUDA.hpp"
#include "cuda_utils.h"


__device__ inline uint64_t BitExpansionDevice(uint32_t v) {
        uint64_t x = v & 0x1FFFFF;
        x = (x | (x << 32)) & 0x1F00000000FFFFULL;
        x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
        x = (x | (x << 8)) & 0x100F00F00F00F00FULL;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3ULL;
        x = (x | (x << 2)) & 0x1249249249249249ULL;
        return x;
}

__device__ inline uint64_t MortonCodeDevice(uint32_t x, uint32_t y, uint32_t z) {
        return BitExpansionDevice(x) | (BitExpansionDevice(y) << 1) | (BitExpansionDevice(z) << 2);
}

__global__ void generateMortonCodesKernel(
    const double* __restrict__ d_pos_x,
    const double* __restrict__ d_pos_y,
    const double* __restrict__ d_pos_z,
    double xmin, double ymin, double zmin,
    double side, uint32_t resolution,
    size_t num_particles,
    uint64_t* __restrict__ d_morton,
    uint32_t* __restrict__ d_particle_id
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    double inv_side = (side > 0.0) ? (1.0 / side) : 0.0;
    uint32_t max_val = resolution - 1;

    for (size_t i = idx; i < num_particles; i += stride) {
        double nx = (d_pos_x[i] - xmin) * inv_side * resolution;
        double ny = (d_pos_y[i] - ymin) * inv_side * resolution;
        double nz = (d_pos_z[i] - zmin) * inv_side * resolution;

        uint32_t ux = static_cast<uint32_t>(fmin(fmax(nx, 0.0), static_cast<double>(max_val)));
        uint32_t uy = static_cast<uint32_t>(fmin(fmax(ny, 0.0), static_cast<double>(max_val)));
        uint32_t uz = static_cast<uint32_t>(fmin(fmax(nz, 0.0), static_cast<double>(max_val)));

        d_morton[i] = MortonCodeDevice(ux, uy, uz);
        d_particle_id[i] = static_cast<uint32_t>(i);
    }
}

__host__ void computeMortonCodesCUDA(
    const double* d_pos_x, const double* d_pos_y,
    const double* d_pos_z, double xmin, double ymin,
    double zmin, double side, uint32_t resolution,
    size_t num_particles, uint64_t* d_morton,
    uint32_t* d_particle_id
) {
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
}