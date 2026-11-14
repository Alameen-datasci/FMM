#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>


class FMM final {
public:
    static constexpr int max_expansion_order = 10;
    static constexpr std::uint32_t morton_resolution = 1u << 21;

    FMM(
        std::size_t num_particles_,
        const double* masses_,
        const double* pos_x_, const double* pos_y_, const double* pos_z_,
        const double* vel_x_ = nullptr, const double* vel_y_ = nullptr, const double* vel_z_ = nullptr,
        double* acc_x_ = nullptr, double* acc_y_ = nullptr, double* acc_z_ = nullptr
    );

    // Destructor
    ~FMM() noexcept;

    // Disable copy and move operations
    FMM(const FMM&) = delete;
    FMM& operator=(const FMM&) = delete;
    FMM(FMM&) = delete;
    FMM& operator=(FMM&&) = delete;

    // Build
    void build(int expansion_order, int particles_per_leaf);

    [[nodiscard]] std::size_t particle_count() const noexcept { return num_particles; }
    [[nodiscard]] int node_count() const noexcept { return total_nodes; }

private:
    std::size_t num_particles;

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

    double* d_masses = nullptr;
    double* d_pos_x = nullptr;
    double* d_pos_y = nullptr;
    double* d_pos_z = nullptr;
    std::uint64_t* d_morton = nullptr;
    std::uint32_t* d_particle_id = nullptr;

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
    double* d_L_real = nullptr;
    double* d_L_imag = nullptr;

    std::size_t node_capacity = 0;
    int total_nodes = 0;
    std::vector<int> level_bounds;
}