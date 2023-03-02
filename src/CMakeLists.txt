cmake_minimum_required(VERSION 3.5.1)
project(dvsfire)

set(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native") # enable avx

# OpenBlas
find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

# Pybind11
find_package(pybind11 REQUIRED)

# Subdirectory
add_subdirectory(_filter)
