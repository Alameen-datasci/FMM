#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <stdexcept>
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