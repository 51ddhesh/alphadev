cmake_minimum_required(VERSION 3.28)
project(alphadev_env LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Python_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 3. Use that path to find the package
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)

pybind11_add_module(alphadev_env
    src/assembly_env/assembly_env.cpp
    src/assembly_env/bindings.cpp
)

target_include_directories(alphadev_env PRIVATE src/env)

target_compile_options(alphadev_env PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
)
