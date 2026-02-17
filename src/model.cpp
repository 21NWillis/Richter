// ─── Richter: Simulation Orchestrator ───────────────────────────────
// Manages device memory, time-stepping loop, and kernel dispatch.

#include "richter/model.h"
#include "richter/kernels.h"
#include "richter/wavelet.h"
#include "richter/boundary.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "[Richter] CUDA error at %s:%d — %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

void richter_init(const Grid& grid, const Source& src, DeviceState& state) {
    size_t bytes = grid.bytes();

    CUDA_CHECK(cudaMalloc(&state.d_u_prev, bytes));
    CUDA_CHECK(cudaMalloc(&state.d_u_curr, bytes));
    CUDA_CHECK(cudaMalloc(&state.d_u_next, bytes));
    CUDA_CHECK(cudaMalloc(&state.d_vel,    bytes));

    CUDA_CHECK(cudaMemset(state.d_u_prev, 0, bytes));
    CUDA_CHECK(cudaMemset(state.d_u_curr, 0, bytes));
    CUDA_CHECK(cudaMemset(state.d_u_next, 0, bytes));

    // Upload wavelet
    CUDA_CHECK(cudaMalloc(&state.d_wavelet, grid.nt * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_wavelet, src.wavelet,
                          grid.nt * sizeof(float), cudaMemcpyHostToDevice));
}

/// Inject source at a single grid point via host round-trip.
/// (For Day 1 simplicity — will be moved to a device kernel later.)
static void inject_source(const Grid& grid, const Source& src,
                           DeviceState& state, int t)
{
    int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
    float val;
    CUDA_CHECK(cudaMemcpy(&val, state.d_u_curr + idx,
                          sizeof(float), cudaMemcpyDeviceToHost));
    val += src.wavelet[t];
    CUDA_CHECK(cudaMemcpy(state.d_u_curr + idx, &val,
                          sizeof(float), cudaMemcpyHostToDevice));
}

void richter_forward(const Grid& grid, const Source& src,
                     DeviceState& state, KernelType kernel)
{
    for (int t = 0; t < grid.nt; t++) {
        // 1. Inject source energy
        inject_source(grid, src, state, t);

        // 2. Dispatch stencil kernel
        switch (kernel) {
            case KernelType::NAIVE:
                launch_kernel_naive(state.d_u_prev, state.d_u_curr,
                                    state.d_u_next, state.d_vel,
                                    grid.nx, grid.ny, grid.nz);
                break;
            case KernelType::SHARED_MEMORY:
                launch_kernel_shmem(state.d_u_prev, state.d_u_curr,
                                    state.d_u_next, state.d_vel,
                                    grid.nx, grid.ny, grid.nz);
                break;
            case KernelType::REGISTER_ROT:
                launch_kernel_register(state.d_u_prev, state.d_u_curr,
                                       state.d_u_next, state.d_vel,
                                       grid.nx, grid.ny, grid.nz);
                break;
        }

        // 3. Apply absorbing boundary condition
        apply_sponge_boundary(state.d_u_next, grid.nx, grid.ny, grid.nz,
                              20, 0.015f);

        // 4. Rotate buffers: prev ← curr, curr ← next
        float* tmp     = state.d_u_prev;
        state.d_u_prev = state.d_u_curr;
        state.d_u_curr = state.d_u_next;
        state.d_u_next = tmp;
    }
}

void richter_snapshot(const Grid& grid, const DeviceState& state,
                      float* h_output)
{
    CUDA_CHECK(cudaMemcpy(h_output, state.d_u_curr,
                          grid.bytes(), cudaMemcpyDeviceToHost));
}

void richter_cleanup(DeviceState& state) {
    cudaFree(state.d_u_prev);
    cudaFree(state.d_u_curr);
    cudaFree(state.d_u_next);
    cudaFree(state.d_vel);
    cudaFree(state.d_wavelet);
    memset(&state, 0, sizeof(state));
}
