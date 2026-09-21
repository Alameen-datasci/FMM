#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

constexpr uint32_t resolution = 1u << 21; // 20,97,152


struct Octree {
    int num_nodes = 0;

    int* parent_idx; // Index of Parent node
    int* child_idx;  // Index of Fisrt Child

    int* particle_start;
    int* particle_count; // Number of Particle in this node

    double* center_x;
    double* center_y;
    double* center_z;
    double* half_width;

    int* level; // Node Level
};

class FMM {
    public:
        // Constructor decleration
        FMM(
            size_t num_particles,
            const double* masses_,
            const double* pos_x_, const double* pos_y_, const double* pos_z_,
            const double* vel_x_, const double* vel_y_, const double* vel_z_,
            double* acc_x_, double* acc_y_, double* acc_z_
        );

        // Deconstructor for pinned memory
        ~FMM();

        void build(int p, int n_crit);
    
    private:
        size_t num_particles = 0;    

        const double* masses;

        const double* pos_x;
        const double* pos_y;
        const double* pos_z;

        const double* vel_x;
        const double* vel_y;
        const double* vel_z;

        double* acc_x;
        double* acc_y;
        double* acc_z;

        uint64_t* morton = nullptr;
        uint32_t* particle_id = nullptr;

        double* d_masses = nullptr;
        double* d_pos_x = nullptr;
        double* d_pos_y = nullptr;
        double* d_pos_z = nullptr;
        uint32_t* d_particle_id = nullptr;

        // Tree Topology
        int* d_node_start = nullptr;
        int* d_node_count = nullptr;
        int* d_child_idx = nullptr;
        int* d_node_parent = nullptr;
        int* d_node_level = nullptr;

        double* d_center_x = nullptr;
        double* d_center_y = nullptr;
        double* d_center_z = nullptr;
        double* d_half_width = nullptr;

        double* d_M_real = nullptr;
        double* d_M_imag = nullptr;

        std::vector<int> level_bounds;
        size_t node_capacity = 0;
        int total_nodes = 0;
};
