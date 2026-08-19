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
    target_compile_options(alphaz0_cpp PRIVATE /O2)
else()
    # Do NOT use -march=native in Docker — the build and runtime containers
    # can run on different CPU hardware (causes SIGILL / exit 132 on Render).
    # -msse4.1 is a safe baseline supported by all x86-64 hosts since ~2008.
    target_compile_options(alphaz0_cpp PRIVATE -O2 -msse4.1)
endif()

# Install the module into the Chess directory
install(TARGETS alphaz0_cpp LIBRARY DESTINATION .)
