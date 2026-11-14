#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

void computeMortonCodesCUDA(
    const double* pos_x, const double* pos_y, const double* pos_z,
    double xmin, double ymin, double zmin, double side,
    std::uint32_t resolution, std::size_t num_particles,
    std::uint64_t* morton, std::uint32_t* particle_id
);

void buildTreeCUDA(
    const std::uint64_t* morton,
    std::size_t num_particles,
    int particles_per_leaf,
    int* node_start, int* node_count, int* child_idx, int* node_parent,
    int* out_total_nodes,
    std::vector<int>& out_level_bounds
);

void mapTreeGeometryCUDA(
    int total_nodes,
    const std::vector<int>& level_bounds,
    const int* child_idx, const int* node_parent, int* node_level,
    double* center_x, double* center_y, double* center_z,
    double* half_width, double root_cx, double root_cy, double root_cz,
    double root_half_width
);

void computeP2MCUDA(
    int total_nodes, int expansion_order,
    const int* node_start, const int* node_count,
    const int* child_idx, const std::uint32_t* particle_id,
    const double* pos_x, const double* pos_y, const double* pos_z,
    const double* masses, const double* center_x,
    const double* center_y, const double* center_z,
    double** multipole_real, double** multipole_imag
);

void computeM2MCUDA(
    int total_nodes, int expansion_order,
    const std::vector<int>& level_bounds,
    const int* child_idx,
    const double* center_x, const double* center_y, const double* center_z,
    double* multipole_real, double* multipole_imag
);
