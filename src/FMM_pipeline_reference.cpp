#include <algorithm>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>

#include "../FMM.hpp"
#include "../FMM_CUDA.hpp"
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

template <typename T>
void freePinnedHost(T*& pointer) {
    if (pointer) {
        CUDA_CHECK(cudaFreeHost(pointer));
        pointer = nullptr;
    }
}

} // namespace

FMM::FMM(
        size_t num_particles_, const double* masses_,
        const double* pos_x_, const double* pos_y_, const double* pos_z_,
        const double* vel_x_, const double* vel_y_, const double* vel_z_,
        double* acc_x_, double* acc_y_, double* acc_z_
        ) : num_particles(num_particles_), masses(masses_),
        pos_x(pos_x_), pos_y(pos_y_), pos_z(pos_z_),
        vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_),
        acc_x(acc_x_), acc_y(acc_y_), acc_z(acc_z_) {}

FMM::~FMM() {
    freeDevice(d_M_real);
    freeDevice(d_M_imag);

    freeDevice(d_center_x);
    freeDevice(d_center_y);
    freeDevice(d_center_z);
    freeDevice(d_half_width);

    freeDevice(d_node_start);
    freeDevice(d_node_count);
    freeDevice(d_child_idx);
    freeDevice(d_node_parent);
    freeDevice(d_node_level);

    freeDevice(d_masses);
    freeDevice(d_pos_x);
    freeDevice(d_pos_y);
    freeDevice(d_pos_z);
    freeDevice(d_particle_id);

    freePinnedHost(morton);
    freePinnedHost(particle_id);
}

void FMM::build(int p, int n_crit) {
    if (num_particles == 0) return;

    if (!masses || !pos_x || !pos_y || !pos_z) {
        throw std::invalid_argument("Particle masses and positions are required.");
    }

    if (p <= 0 || p > kMaxExpansionOrder) {
        throw std::invalid_argument("p must be between 0 and 10.");
    }

    if (n_crit <= 0) {
        throw std::invalid_argument("n_crit must be positive.");
    }

    if (num_particles > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("num_particles exceeds the current int-based tree limit.");
    }

    if (!morton) {
        CUDA_CHECK(cudaMallocHost((void**)&morton, num_particles * sizeof(uint64_t)));
    }

    if (!particle_id) {
        CUDA_CHECK(cudaMallocHost((void**)&particle_id, num_particles * sizeof(uint32_t)));
    }

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

    computeMortonCodesCUDA(
            pos_x, pos_y, pos_z,
            xmin, ymin, zmin,
            side,
            resolution,
            num_particles,
            morton,
            particle_id
    );

    const size_t max_nonleaf_nodes_per_level =
            (num_particles + static_cast<size_t>(n_crit) - 1) / static_cast<size_t>(n_crit);
    const size_t capacity_factor = 8 * static_cast<size_t>(kMaxTreeDepth);

    if (max_nonleaf_nodes_per_level >
            (std::numeric_limits<size_t>::max() - 1) / capacity_factor) {
        throw std::overflow_error("Tree capacity calculation overflowed.");
    }

    const size_t required_node_capacity =
            1 + capacity_factor * max_nonleaf_nodes_per_level;

    if (required_node_capacity > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Tree capacity exceeds the current int-based node limit.");
    }

    if (required_node_capacity > node_capacity) {
        freeDevice(d_node_start);
        freeDevice(d_node_count);
        freeDevice(d_child_idx);
        freeDevice(d_node_parent);
        freeDevice(d_node_level);
        freeDevice(d_center_x);
        freeDevice(d_center_y);
        freeDevice(d_center_z);
        freeDevice(d_half_width);

        CUDA_CHECK(cudaMalloc((void**)&d_node_start, required_node_capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_node_count, required_node_capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_child_idx, required_node_capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_node_parent, required_node_capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_node_level, required_node_capacity * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void**)&d_center_x, required_node_capacity * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_center_y, required_node_capacity * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_center_z, required_node_capacity * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_half_width, required_node_capacity * sizeof(double)));

        node_capacity = required_node_capacity;
    }

    if (!d_masses) CUDA_CHECK(cudaMalloc((void**)&d_masses, num_particles * sizeof(double)));
    if (!d_pos_x) CUDA_CHECK(cudaMalloc((void**)&d_pos_x, num_particles * sizeof(double)));
    if (!d_pos_y) CUDA_CHECK(cudaMalloc((void**)&d_pos_y, num_particles * sizeof(double)));
    if (!d_pos_z) CUDA_CHECK(cudaMalloc((void**)&d_pos_z, num_particles * sizeof(double)));
    if (!d_particle_id) CUDA_CHECK(cudaMalloc((void**)&d_particle_id, num_particles * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_masses, masses, num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_x, pos_x, num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, pos_y, num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, pos_z, num_particles * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_id, particle_id, num_particles * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint64_t* d_morton = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_morton, num_particles * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_morton, morton, num_particles * sizeof(uint64_t), cudaMemcpyHostToDevice));

    buildTreeCUDA(
            d_morton,
            num_particles,
            n_crit,
            d_node_start,
            d_node_count,
            d_child_idx,
            d_node_parent,
            &total_nodes,
            level_bounds
    );

    CUDA_CHECK(cudaFree(d_morton));

    mapTreeGeometryCUDA(
            total_nodes,
            level_bounds,
            d_child_idx,
            d_node_parent,
            d_node_level,
            d_center_x,
            d_center_y,
            d_center_z,
            d_half_width,
            root_cx,
            root_cy,
            root_cz,
            0.5 * side
    );

    freeDevice(d_M_real);
    freeDevice(d_M_imag);

    computeP2MCUDA(
            total_nodes,
            p,
            d_node_start,
            d_node_count,
            d_child_idx,
            d_particle_id,
            d_pos_x,
            d_pos_y,
            d_pos_z,
            d_masses,
            d_center_x,
            d_center_y,
            d_center_z,
            &d_M_real,
            &d_M_imag
    );
}
