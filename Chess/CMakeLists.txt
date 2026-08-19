cmake_minimum_required(VERSION 3.15)
project(alphaz0_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 (installed via pip)
find_package(pybind11 REQUIRED)

# Build the Python extension module
pybind11_add_module(alphaz0_cpp
    cpp/bindings.cpp
)

target_include_directories(alphaz0_cpp PRIVATE cpp/)

# Optimization flags
if(MSVC)
    target_compile_options(alphaz0_cpp PRIVATE /O2 /GL)
    target_link_options(alphaz0_cpp PRIVATE /LTCG)
else()
    target_compile_options(alphaz0_cpp PRIVATE -O3 -march=native -flto)
    target_link_options(alphaz0_cpp PRIVATE -flto)
endif()

# Install the module into the Chess directory
install(TARGETS alphaz0_cpp LIBRARY DESTINATION .)
