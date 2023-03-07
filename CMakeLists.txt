cmake_minimum_required(VERSION 3.5.1)
project(dvsfire)

# Configurable features
set(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native") # enable avx
 
# Find OpenBlas supports
find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

# Find Pybind11 supports
find_package(pybind11 REQUIRED)

# Find OpenCV supports
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find dv-runtime supports
find_package(dv REQUIRED)

# Inclue header files.
include_directories(${PROJECT_SOURCE_DIR}/include)

# Compile modules
add_subdirectory(src)
