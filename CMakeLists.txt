cmake_minimum_required(VERSION 3.18)
project(Richter LANGUAGES CXX CUDA)

# ─── C++ / CUDA standards ───────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture (default to Ampere)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ─── Source files ────────────────────────────────────────────────────
set(KERNEL_SOURCES
    src/kernels/hello_naive.cu
    src/kernels/hello_shmem.cu
    src/kernels/hello_register.cu
    src/boundary.cu
)

set(CORE_SOURCES
    src/model.cpp
    src/wavelet.cpp
)

# ─── Static library: richter_core ────────────────────────────────────
add_library(richter_core STATIC
    ${KERNEL_SOURCES}
    ${CORE_SOURCES}
)

target_include_directories(richter_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(richter_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ─── PyBind11 module (optional, only if pybind11 found) ──────────────
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(richter_py python/bindings.cpp)
    target_link_libraries(richter_py PRIVATE richter_core)
    message(STATUS "PyBind11 found — building Python module")
else()
    message(STATUS "PyBind11 not found — skipping Python module")
endif()

# ─── Test executable ─────────────────────────────────────────────────
add_executable(runtests tests/runtests.cpp)
target_link_libraries(runtests PRIVATE richter_core)

# ─── Benchmark executable ───────────────────────────────────────────
add_executable(benchmark benchmarks/benchmark.cpp)
target_link_libraries(benchmark PRIVATE richter_core)

# ─── Snapshot tool (visualization) ──────────────────────────────────
add_executable(snapshot tools/snapshot.cpp)
target_link_libraries(snapshot PRIVATE richter_core)
