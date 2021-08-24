cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS " -Wall -funroll-loops -O3 -march=native ")

add_executable(main main.cpp )
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
