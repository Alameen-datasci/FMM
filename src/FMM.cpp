#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "../FMM_CUDA.hpp"
#include "../FMM.hpp"
#include "../cuda_utils.h"

namespace {

constexpr int kMaxExpansionOrder = 10;
constexpr int kMaxTreeDepth = 21;

template <typename T>
void freeDevice(T*& pointer) {
    if (pointer) {
        CUDA_CHECK(cudaFree(pointer));
        pointer = nullptr;
    }
}

} // namespace

FMM::FMM(
    size_t num_particles_, double* masses_,
    double* pos_x_, double* pos_y_, double* pos_z_,
    double* vel_x_, double* vel_y_, double* vel_z_,
    double* acc_x_, double* acc_y_, double* acc_z_
    ) : num_particles(num_particles_), masses(masses_),
    pos_x(pos_x_), pos_y(pos_y_), pos_z(pos_z_),
    vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_),
    acc_x(acc_x_), acc_y(acc_y_), acc_z(acc_z_) {}

FMM::~FMM() {
    freeDevice(d_masses);
    freeDevice(d_pos_x);
    freeDevice(d_pos_y);
    freeDevice(d_pos_z);

    freeDevice(d_morton);
    freeDevice(d_particle_id);
}

void FMM::build(int p, int N_crit) {
    if (num_particles == 0) return;

    if (!masses || !pos_x || !pos_y || !pos_z) {
        throw std::invalid_argument("Particle masses and positions are required.");
    }

    if (p < 0 || p > kMaxExpansionOrder) {
        throw std::invalid_argument("p must be in between 0 and 10.");
    }

    if (N_crit <= 0) {
        throw std::invalid_argument("N_crit must be positive.");
    }

    if (num_particles > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("num_particles exeeds the current int-based tree limit.");
    }

    // Allocating space for masses and positions in Device
    if (!d_masses) CUDA_CHECK(cudaMalloc((void**)&d_masses, num_particles * sizeof(double)));
    if (!d_pos_x) CUDA_CHECK(cudaMalloc((void**)&d_pos_x, num_particles * sizeof(double)));
    if (!d_pos_y) CUDA_CHECK(cudaMalloc((void**)&d_pos_y, num_particles * sizeof(double)));
    if (!d_pos_z) CUDA_CHECK(cudaMalloc((void**)&d_pos_z, num_particles * sizeof(double)));

    // Copying data to Device
    CUDA_CHECK(cudaMemcpy(d_masses, masses, num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_x, pos_x, num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, pos_y, num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, pos_z, num_particles * sizeof(double), cudaMemcpyHostToDevice));

    // Calcualting the bounding box
    double xmin = pos_x[0], ymin = pos_y[0], zmin = pos_z[0];
    double xmax = pos_x[0], ymax = pos_y[0], zmax = pos_z[0];

    for (size_t i = 1; i < num_particles; i++) {
        xmin = std::min(xmin, pos_x[i]);
        ymin = std::min(ymin, pos_y[i]);
        zmin = std::min(zmin, pos_z[i]);
        xmax = std::max(xmax, pos_x[i]);
        ymax = std::max(ymax, pos_y[i]);
        zmax = std::max(zmax, pos_z[i]);
    }

    const double side = std::max({xmax - xmin, ymax - ymin, zmax - zmin});
    const double root_cx = 0.5 * (xmax + xmin);
    const double root_cy = 0.5 * (ymax + ymin);
    const double root_cz = 0.5 * (zmax + zmin);

    xmin = root_cx - 0.5 * side;
    ymin = root_cy - 0.5 * side;
    zmin = root_cz - 0.5 * side;

    if (!d_morton) {
        CUDA_CHECK(cudaMalloc((void**)&d_morton, num_particles * sizeof(uint64_t)));
    }

    if (!d_particle_id) {
        CUDA_CHECK(cudaMalloc((void**)&d_particle_id, num_particles * sizeof(uint32_t)));
    }

    // pass masses and positions to the gpu to do the work (morton generation)
    computeMortonCodesCUDA(
        d_pos_x, d_pos_y, d_pos_z,
        xmin, ymin, zmin,
        side, resolution,
        num_particles,
        d_morton,
        d_particle_id
    );
}