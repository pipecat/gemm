cmake_minimum_required(VERSION 3.10)

project(mm_benchmarks CXX)

set(CMAKE_CXX_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Check for CUDA
# First try to find CUDA using CMake's built-in support
find_package(CUDA QUIET)

# If CUDA not found, try to locate CUDA toolkit manually
if(NOT CUDA_FOUND)
  # Look for CUDA in common installation paths
  set(CUDA_SEARCH_PATHS
    "/usr/local/cuda"
    "/opt/cuda"
    "$ENV{CUDA_PATH}"
    "$ENV{CUDA_HOME}"
  )
  
  find_path(CUDA_TOOLKIT_ROOT_DIR
    NAMES include/cuda_runtime.h
    PATHS ${CUDA_SEARCH_PATHS}
    DOC "Path to CUDA Toolkit"
  )
  
  if(CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_FOUND TRUE)
    message(STATUS "Found CUDA Toolkit at: ${CUDA_TOOLKIT_ROOT_DIR}")
  endif()
endif()

if(CUDA_FOUND)
  # CUDA environment - compile only gemm.cu
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  
  # Find CUDA
  find_package(CUDA REQUIRED)
  
  # Add CUDA executables
  cuda_add_executable(gemm-kernel-benchmark gemm.cu gemm.h)
  target_compile_options(gemm-kernel-benchmark PRIVATE -O3)
  
  # Add flops benchmark for CUDA
  cuda_add_executable(flops-benchmark flops.cu)
  target_compile_options(flops-benchmark PRIVATE -O3)
  target_link_libraries(flops-benchmark ${CUDA_LIBRARIES})
  
else()
  # ARM CPU environment - compile original files
  set(EIGEN3_INCLUDE_DIR "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3")
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Please set the environment variable EIGEN3_INCLUDE_DIR")
  endif()

  # Add OpenBLAS CMake config path
  list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/openblas/lib/cmake/openblas")

  # Find OpenBLAS library
  find_package(OpenBLAS REQUIRED)
  include_directories(${OpenBLAS_INCLUDE_DIRS})

  add_executable(gemm-kernel-benchmark gemm.cc gemm.h)
  target_include_directories(gemm-kernel-benchmark PUBLIC "${EIGEN3_INCLUDE_DIR}")
  target_compile_options(gemm-kernel-benchmark PRIVATE -O3 -fno-unroll-loops -fno-vectorize)
  target_link_libraries(gemm-kernel-benchmark ${OpenBLAS_LIBRARIES})

  # Add flops benchmark for ARM
  add_executable(flops-benchmark flops.c)
  target_compile_options(flops-benchmark PRIVATE -O3)
endif()
