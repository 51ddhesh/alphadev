cmake_minimum_required(VERSION 3.20)
project(alphadev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 REQUIRED)

pybind11_add_module(alphadev src/assembly_env.cpp src/bindings.cpp)

if(MSVC)
    target_compile_options(alphadev PRIVATE /O2 /fp:fast)
else()
    target_compile_options(alphadev PRIVATE -O3 -march=native)
endif()