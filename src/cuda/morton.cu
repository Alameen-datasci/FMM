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