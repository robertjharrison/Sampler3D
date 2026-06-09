cmake_minimum_required(VERSION 3.10)
project(Sampler3D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

find_package(OpenMP REQUIRED)

include_directories(include)

add_executable(sampler_demo src/main.cpp src/octree.cpp src/alias_method.cpp src/sampler.cpp)
target_link_libraries(sampler_demo PRIVATE OpenMP::OpenMP_CXX)

# Tests
add_executable(test_alias tests/test_alias.cpp src/alias_method.cpp)
target_link_libraries(test_alias PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_pcg64 tests/test_pcg64.cpp)
target_link_libraries(test_pcg64 PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_octree tests/test_octree.cpp src/octree.cpp)
target_link_libraries(test_octree PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_sampler tests/test_sampler.cpp src/sampler.cpp src/octree.cpp src/alias_method.cpp)
target_link_libraries(test_sampler PRIVATE OpenMP::OpenMP_CXX)
