#include "FMM.hpp"
#include "FMM_CUDA.hpp"
#include "cuda_utils.h"

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
        if (morton) CUDA_CHECK(cudaFreeHost(morton));
        if (particle_id) CUDA_CHECK(cudaFreeHost(particle_id));
}

void FMM::build() {
        if (num_particles == 0) return;

        if (!morton) {
                CUDA_CHECK(cudaMallocHost((void**)&morton, num_particles * sizeof(uint64_t)));
        }

        if (!particle_id) {
                CUDA_CHECK(cudaMallocHost((void**)&particle_id, num_particles * sizeof(uint32_t)));
        }

        // Get the bounding box
        double xmin = pos_x[0], ymin = pos_y[0], zmin = pos_z[0];
        double xmax = pos_x[0], ymax = pos_y[0], zmax = pos_z[0];

        for (size_t i = 0; i < num_particles; i++) {
                if (pos_x[i] < xmin) xmin = pos_x[i];
                if (pos_y[i] < ymin) ymin = pos_y[i];
                if (pos_z[i] < zmin) zmin = pos_z[i];

                if (pos_x[i] > xmax) xmax = pos_x[i];
                if (pos_y[i] > ymax) ymax = pos_y[i];
                if (pos_z[i] > zmax) zmax = pos_z[i];
        }

        double dx = xmax - xmin;
        double dy = ymax - ymin;
        double dz = zmax - zmin;

        double side = std::max({dx, dy, dz});

        // Expand the cube, keeping the center fixed
        double cx = 0.5 * (xmax + xmin);
        double cy = 0.5 * (ymax + ymin);
        double cz = 0.5 * (zmax + zmin);

        xmin = cx - 0.5 * side;
        ymin = cy - 0.5 * side;
        zmin = cz - 0.5 * side;

        xmax = cx + 0.5 * side;
        ymax = cy + 0.5 * side;
        zmax = cz + 0.5 * side;

        computeMortonCodesCUDA(
                pos_x, pos_y, pos_z,
                xmin, ymin, zmin,
                side,
                resolution,
                num_particles,
                morton,
                particle_id
        );

}