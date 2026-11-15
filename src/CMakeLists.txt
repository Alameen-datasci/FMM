add_library(fmm
    FMM.cpp
    cuda/morton.cu
    cuda/geometry.cu
    cuda/octree.cu
    cuda/p2m.cu
    cuda/m2m.cu
)
add_library(FMM::fmm ALIAS fmm)

set_target_properties(fmm PROPERTIES
    EXPORT_NAME fmm
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_EXTENSIONS NO
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Headers currently live in the repository root. Keep build-tree includes
# private to this source layout while installed consumers use <FMM.hpp>.
target_include_directories(fmm
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${PROJECT_SOURCE_DIR}"
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(fmm PUBLIC CUDA::cudart)
target_compile_features(fmm PUBLIC cxx_std_17 cuda_std_17)

if(FMM_ENABLE_WARNINGS)
    include("${PROJECT_SOURCE_DIR}/cmake/CompilerOptions.cmake")
    fmm_enable_warnings(fmm)
endif()

install(TARGETS fmm
    EXPORT FMMTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(FILES
    "${PROJECT_SOURCE_DIR}/FMM.hpp"
    "${PROJECT_SOURCE_DIR}/FMM_CUDA.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/FMM"
)
