add_executable(fmm_cuda_tests
    uints/test_cuda_morton.cpp
)
target_link_libraries(fmm_cuda_tests PRIVATE FMM::fmm GTest::gtest_main)
target_compile_features(fmm_cuda_tests PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(fmm_cuda_tests)
