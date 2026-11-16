#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <limits>
#include "FMM_CUDA.hpp"


class MortonCudaTest : public ::testing::Test {
protected:
    double* d_pos_x = nullptr;
    double* d_pos_y = nullptr;
    double* d_pos_z = nullptr;
    std::uint64_t* d_morton = nullptr;
    std::uint32_t* d_particle_id = nullptr;

    void SetUp() override {
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_pos_x), sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_pos_y), sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_pos_z), sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_morton), sizeof(std::uint64_t)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_particle_id), sizeof(std::uint32_t)), cudaSuccess);
    }

    void TearDown() override {
        EXPECT_EQ(cudaFree(d_pos_x), cudaSuccess);
        EXPECT_EQ(cudaFree(d_pos_y), cudaSuccess);
        EXPECT_EQ(cudaFree(d_pos_z), cudaSuccess);
        EXPECT_EQ(cudaFree(d_morton), cudaSuccess);
        EXPECT_EQ(cudaFree(d_particle_id), cudaSuccess);
    }
};

TEST_F(MortonCudaTest, RejectsZeroResolution) {
    EXPECT_THROW(
        computeMortonCodesCUDA(
            d_pos_x, d_pos_y, d_pos_z,
            0.0, 0.0, 0.0, 1.0, 0,
            1, d_morton, d_particle_id
        ),
        std::invalid_argument
    );
}

TEST_F(MortonCudaTest, ZeroParticle) {
    EXPECT_NO_THROW(
        computeMortonCodesCUDA(
            nullptr, nullptr, nullptr,
            0.0, 0.0, 0.0, 1.0, 1, 0,
            nullptr, nullptr
        )
    );
}

TEST_F(MortonCudaTest, OneParticleAtLowerBounds) {
    const double x = 0.0, y = 0.0, z = 0.0;
    ASSERT_EQ(cudaMemcpy(d_pos_x, &x, sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_pos_y, &y, sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_pos_z, &z, sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);

    computeMortonCodesCUDA(
        d_pos_x, d_pos_y, d_pos_z,
        0.0, 0.0, 0.0, 1.0, 2, 1,
        d_morton, d_particle_id
    );

    std::uint64_t morton = 99; // 99 is a placeholder, initializing with value make debugging easy.
    std::uint32_t particle_id = 99; // 99 is a placeholder, initializing with value make debugging easy.
    ASSERT_EQ(cudaMemcpy(&morton, d_morton, sizeof(morton), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&particle_id, d_particle_id, sizeof(particle_id), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(morton, 0u);
    EXPECT_EQ(particle_id, 0u);
}

TEST_F(MortonCudaTest, RejectsNonFiniteBounds) {
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(
        computeMortonCodesCUDA(
            d_pos_x, d_pos_y, d_pos_z,
            0.0, nan, 0.0,              // Here ymin = nan
            1.0, 1, 1, d_morton, d_particle_id
        ),
        std::invalid_argument
    );
}

TEST_F(MortonCudaTest, RejectsInvalidResolution) {
    const uint32_t res = 1u << 25;

    EXPECT_THROW(
        computeMortonCodesCUDA(
            d_pos_x, d_pos_y, d_pos_z,
            0.0, 0.0, 0.0, 1.0,
            res, 2, d_morton, d_particle_id
        ),
        std::invalid_argument
    );
}

TEST_F(MortonCudaTest, RejectsNegativeSide) {
    EXPECT_THROW(
        computeMortonCodesCUDA(
            d_pos_x, d_pos_y, d_pos_z,
            0.0, 0.0, 0.0, -2.0,
            1, 2, d_morton, d_particle_id
        ),
        std::invalid_argument
    );
}

TEST_F(MortonCudaTest, RejectsTooManyParticles) {
    const std::size_t num_part = 
        static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;

    EXPECT_THROW(
        computeMortonCodesCUDA(
            d_pos_x, d_pos_y, d_pos_z,
            0.0, 0.0, 0.0, 1.0, 1, num_part,
            d_morton, d_particle_id
        ),
        std::overflow_error
    );
}

TEST_F(MortonCudaTest, RejectsNullPointers) {
    EXPECT_THROW(
        computeMortonCodesCUDA(
            d_pos_x, d_pos_y, nullptr,
            0.0, 0.0, 0.0, 1.0, 1, 25,
            d_morton, d_particle_id
        ),
        std::invalid_argument
    );
}

TEST_F(MortonCudaTest, Sorts25ParticlesAndPreservesIds) {
    // =================== Pre-defined Arrays ===================
    const double x[25] = {
    0.625, 0.375, 0.125, 0.375, 0.125,
    0.375, 0.125, 0.375, 0.125, 0.875,
    0.625, 0.875, 0.625, 0.875, 0.625,
    0.875, 0.625, 0.375, 0.125, 0.375,
    0.125, 0.375, 0.125, 0.375, 0.125
    };

    const double y[25] = {
        0.625, 0.875, 0.875, 0.625, 0.625,
        0.875, 0.875, 0.625, 0.625, 0.375,
        0.375, 0.125, 0.125, 0.375, 0.375,
        0.125, 0.125, 0.375, 0.375, 0.125,
        0.125, 0.375, 0.375, 0.125, 0.125
    };

    const double z[25] = {
        0.125, 0.375, 0.375, 0.375, 0.375,
        0.125, 0.125, 0.125, 0.125, 0.375,
        0.375, 0.375, 0.375, 0.125, 0.125,
        0.125, 0.125, 0.375, 0.375, 0.375,
        0.375, 0.125, 0.125, 0.125, 0.125
    };

    // Expected output after sorting by Morton code:
    const std::uint64_t expected_morton[25] = {
        0,  1,  2,  3,  4,
        5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    };

    const std::uint32_t expected_particle_id[25] = {
        24, 23, 22, 21, 20,
        19, 18, 17, 16, 15,
        14, 13, 12, 11, 10,
        9,  8,  7,  6,  5,
        4,  3,  2,  1,  0
    };
    // ==========================================================

    ASSERT_EQ(cudaMemcpy(d_pos_x, x, 25 * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_pos_y, y, 25 * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_pos_z, z, 25 * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);

    computeMortonCodesCUDA(
        d_pos_x, d_pos_y, d_pos_z,
        0.0, 0.0, 0.0, 1.0, 4, 25,
        d_morton, d_particle_id
    );

    std::uint64_t actual_morton[25];
    std::uint32_t actual_particle_id[25];

    ASSERT_EQ(
        cudaMemcpy(actual_morton, d_morton, 25 * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
        cudaSuccess
    );
    ASSERT_EQ(
        cudaMemcpy(actual_particle_id, d_particle_id, 25 * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
        cudaSuccess
    );

    for (std::size_t i = 0; i < 25; i++) {
        EXPECT_EQ(actual_morton[i], expected_morton[i]) << "at sorted index " << i;
        EXPECT_EQ(actual_particle_id[i], expected_particle_id[i]) << "at sorted index " << i;
    }
}