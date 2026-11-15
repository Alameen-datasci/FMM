cmake_minimum_required(VERSION 3.24)

project(FMM
    VERSION 0.1.0
    DESCRIPTION "CUDA-accelerated Fast Multipole Method library"
    LANGUAGES CXX CUDA
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

option(FMM_BUILD_TESTING "Build FMM tests (requires GoogleTest)" OFF)
option(FMM_ENABLE_WARNINGS "Enable useful compiler warnings for FMM targets" ON)

add_subdirectory(src)

if(FMM_BUILD_TESTING)
    enable_testing()
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        find_package(GTest REQUIRED)
    endif()
    add_subdirectory(tests)
endif()

install(EXPORT FMMTargets
    FILE FMMTargets.cmake
    NAMESPACE FMM::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FMM"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FMMConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/FMMConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FMM"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FMMConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FMMConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FMMConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FMM"
)
