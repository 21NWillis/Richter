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
    size_t count = grid.total_points();

    state.d_u_prev = CudaBuffer<float>(count);
    state.d_u_curr = CudaBuffer<float>(count);
    state.d_u_next = CudaBuffer<float>(count);
    state.d_vel    = CudaBuffer<float>(count);

    state.d_u_prev.zero();
    state.d_u_curr.zero();
    state.d_u_next.zero();

    // Upload wavelet
    state.d_wavelet = CudaBuffer<float>(grid.nt);
    state.d_wavelet.copyFromHost(src.wavelet, grid.nt);
}

/// Inject source at a single grid point via host round-trip.
/// (For Day 1 simplicity — will be moved to a device kernel later.)
static void inject_source(const Grid& grid, const Source& src,
                           DeviceState& state, int t)
{
    int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
    float val;
    

    CUDA_CHECK(cudaMemcpy(&val, state.d_u_curr.data() + idx,
                          sizeof(float), cudaMemcpyDeviceToHost));
    val += src.wavelet[t];
    CUDA_CHECK(cudaMemcpy(state.d_u_curr.data() + idx, &val,
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
                launch_kernel_naive(state.d_u_prev.data(), state.d_u_curr.data(),
                                    state.d_u_next.data(), state.d_vel.data(),
                                    grid.nx, grid.ny, grid.nz);
                break;
            case KernelType::SHARED_MEMORY:
                launch_kernel_shmem(state.d_u_prev.data(), state.d_u_curr.data(),
                                    state.d_u_next.data(), state.d_vel.data(),
                                    grid.nx, grid.ny, grid.nz);
                break;
            case KernelType::REGISTER_ROT:
                launch_kernel_register(state.d_u_prev.data(), state.d_u_curr.data(),
                                       state.d_u_next.data(), state.d_vel.data(),
                                       grid.nx, grid.ny, grid.nz);
                break;
            case KernelType::HYBRID:
                launch_kernel_hybrid(state.d_u_prev.data(), state.d_u_curr.data(),
                                     state.d_u_next.data(), state.d_vel.data(),
                                     grid.nx, grid.ny, grid.nz);
                break;
        }

        // 3. Apply absorbing boundary condition
        apply_sponge_boundary(state.d_u_next.data(), grid.nx, grid.ny, grid.nz,
                              20, 0.015f);

        // 4. Rotate buffers: prev ← curr, curr ← next

        state.d_u_prev.swap(state.d_u_curr);
        state.d_u_curr.swap(state.d_u_next);
    }
}

void richter_snapshot(const Grid& grid, const DeviceState& state,
                      float* h_output)
{
    state.d_u_curr.copyToHost(h_output, grid.total_points());
}

void richter_cleanup(DeviceState& state) {

    state.d_u_prev = CudaBuffer<float>();
    state.d_u_curr = CudaBuffer<float>();
    state.d_u_next = CudaBuffer<float>();
    state.d_vel    = CudaBuffer<float>();
    state.d_wavelet = CudaBuffer<float>();
}
