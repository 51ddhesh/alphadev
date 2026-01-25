cmake_minimum_required(VERSION 3.20)
project(alphadev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(MSVC)
    add_compile_options(/O2 /fp:fast)
else()
    add_compile_options(-O2 -march=native -ffast-math)
endif()


add_executable(ad main.cpp)