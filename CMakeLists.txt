cmake_minimum_required(VERSION 3.28)
project(alphadev_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(alphadev_env
    src/env/assembly_env.cpp
    src/env/bindings.cpp
)

target_include_directories(alphadev_env PRIVATE src/env)

target_compile_options(alphadev_env PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
)