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
    src/kernels/hello_hybrid.cu
    src/boundary.cu
    src/receiver.cu
    src/imaging.cu
    src/fwi_kernels.cu
    src/fwi_2d_kernels.cu
)

set(CORE_SOURCES
    src/model.cpp
    src/wavelet.cpp
    src/rtm.cpp
    src/rtm_cpu.cpp
    src/fwi.cpp
    src/fwi_cpu.cpp
    src/fwi_2d.cpp
    src/kernels/hello_cpu.cpp
)

# ─── OpenMP ─────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ─── Static library: richter_core ────────────────────────────────────
add_library(richter_core STATIC
    ${KERNEL_SOURCES}
    ${CORE_SOURCES}
)

target_include_directories(richter_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(richter_core PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(richter_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 --use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX2 /O2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-mavx2 -mfma -O3>
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

# ─── RTM demo ───────────────────────────────────────────────────────
add_executable(rtm_demo tools/rtm_demo.cpp)
target_link_libraries(rtm_demo PRIVATE richter_core)

# ─── RTM CPU demo (no GPU required) ────────────────────────────────
add_executable(rtm_cpu_demo tools/rtm_cpu_demo.cpp)
target_link_libraries(rtm_cpu_demo PRIVATE richter_core)

# ─── FWI demo ──────────────────────────────────────────────────────
add_executable(fwi_demo tools/fwi_demo.cpp)
target_link_libraries(fwi_demo PRIVATE richter_core)

# ─── 2D FWI demo ──────────────────────────────────────────────────
add_executable(fwi_2d_demo tools/fwi_2d_demo.cpp)
target_link_libraries(fwi_2d_demo PRIVATE richter_core)

# ─── Profiling target (standalone for NCU) ──────────────────────────
add_executable(profile_register tools/profile_kernel.cu)
target_compile_options(profile_register PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)
target_include_directories(profile_register PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
