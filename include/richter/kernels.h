#pragma once
#include "config.h"

// ─── Kernel Launch Wrappers ─────────────────────────────────────────
// Each wraps its own __global__ kernel and handles grid/block config.

/// Naive kernel — one thread per grid point, all reads from global memory.
void launch_kernel_naive(const float* u_prev, const float* u_curr,
                         float* u_next, const float* vel,
                         int nx, int ny, int nz);

/// Shared-memory 2.5D tiling — XY plane loaded into shmem, Z streamed.
void launch_kernel_shmem(const float* u_prev, const float* u_curr,
                         float* u_next, const float* vel,
                         int nx, int ny, int nz);

/// Register rotation — sliding window of registers along Z axis.
void launch_kernel_register(const float* u_prev, const float* u_curr,
                            float* u_next, const float* vel,
                            int nx, int ny, int nz);

/// Hybrid — shared memory XY tiling + register rotation Z.
void launch_kernel_hybrid(const float* u_prev, const float* u_curr,
                          float* u_next, const float* vel,
                          int nx, int ny, int nz);
