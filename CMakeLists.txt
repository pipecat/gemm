cmake_minimum_required(VERSION 3.10)

project(mm_benchmarks CXX)

set (CMAKE_CXX_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set (EIGEN3_INCLUDE_DIR "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3" )
IF(NOT EIGEN3_INCLUDE_DIR )
    message( FATAL_ERROR "Please set the environment variable EIGEN3_INCLUDE_DIR")
ENDIF()

add_executable(gemm-kernel-benchmark gemm.cc gemm.h)
target_include_directories(gemm-kernel-benchmark PUBLIC "${EIGEN3_INCLUDE_DIR}")


