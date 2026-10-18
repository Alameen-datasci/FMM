#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

void computeMortonCodesCUDA(
    const double* pos_x, const double* pos_y, const double* pos_z,
    double xmin, double ymin, double zmin, double side,
    uint32_t resolution, size_t num_particles,
    uint64_t* morton, uint32_t* particle_id
);

void buildTreeCUDA(
    const uint64_t* morton,
    size_t num_particles,
    int N_crit,
    int* d_node_start, int* d_node_count, int* d_child_idx, int* d_node_parent,
    int* out_total_nodes,
    std::vector<int>& out_level_bounds
);

void mapTreeGeometryCUDA(
    int total_nodes,
    const std::vector<int>& level_bounds,
    int* d_child_idx, int* d_node_parent, int* d_node_level,
    double* d_center_x, double* d_center_y, double* d_center_z,
    double* d_half_width, double root_cx, double root_cy, double root_cz,
    double root_half_width
);

void computeP2MCUDA(
    int total_nodes, int p,
    const int* d_node_start, const int* d_node_count,
    const int* d_child_idx, const uint32_t* d_particle_id,
    const double* d_pos_x, const double* d_pos_y, const double* d_pos_z,
    const double* d_masses, const double* d_center_x,
    const double* d_center_y, const double* d_center_z,
    double** d_M_real, double** d_M_imag
);
