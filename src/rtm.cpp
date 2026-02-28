// RTM Orchestrator
// Reverse Time Migration: forward propagation with receiver recording,
// backward propagation with receiver injection, and imaging condition.
//
// Strategy: save source wavefield snapshot to host at EVERY timestep.
// This uses more host memory but guarantees correct time alignment.

#include "richter/rtm.h"
#include "richter/kernels.h"
#include "richter/wavelet.h"
#include "richter/boundary.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "[Richter RTM] CUDA error at %s:%d — %s\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// Helper: dispatch stencil kernel by type
static void dispatch_stencil(const float* u_prev, const float* u_curr,
                             float* u_next, const float* vel,
                             int nx, int ny, int nz, KernelType kernel)
{
    switch (kernel) {
        case KernelType::NAIVE:
            launch_kernel_naive(u_prev, u_curr, u_next, vel, nx, ny, nz);
            break;
        case KernelType::SHARED_MEMORY:
            launch_kernel_shmem(u_prev, u_curr, u_next, vel, nx, ny, nz);
            break;
        case KernelType::REGISTER_ROT:
            launch_kernel_register(u_prev, u_curr, u_next, vel, nx, ny, nz);
            break;
        case KernelType::HYBRID:
            launch_kernel_hybrid(u_prev, u_curr, u_next, vel, nx, ny, nz);
            break;
    }
}

// richter_rtm
void richter_rtm(const Grid& grid, const Source& src, const ReceiverSet& rec,
                 DeviceState& state, float* h_image, KernelType kernel,
                 int /* checkpoint_interval */,
                 const float* h_vel_background)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);

    // Allocate device receiver buffers
    CudaBuffer<int> d_rx(rec.num_receivers);
    CudaBuffer<int> d_ry(rec.num_receivers);
    CudaBuffer<int> d_rz(rec.num_receivers);
    CudaBuffer<float> d_traces((size_t)rec.num_receivers * grid.nt);

    d_rx.copyFromHost(rec.rx, rec.num_receivers);
    d_ry.copyFromHost(rec.ry, rec.num_receivers);
    d_rz.copyFromHost(rec.rz, rec.num_receivers);
    d_traces.zero();

    // Allocate image buffer on device
    CudaBuffer<float> d_image(N);
    CudaBuffer<float> d_illum(N);
    d_image.zero();
    d_illum.zero();

    printf("[RTM] Saving all %d source snapshots to host (%.1f MB total)\n",
           grid.nt, (float)grid.nt * bytes / (1024.0f * 1024.0f));

    // PHASE 1: FORWARD PROPAGATION — save every source snapshot
    printf("[RTM] Forward propagation (%d steps)...\n", grid.nt);

    std::vector<std::vector<float>> src_snapshots(grid.nt);

    state.d_u_prev.zero();
    state.d_u_curr.zero();
    state.d_u_next.zero();

    for (int t = 0; t < grid.nt; t++) {
        // Inject source at time t
        int src_idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
        float val;
        CUDA_CHECK(cudaMemcpy(&val, state.d_u_curr.data() + src_idx,
                              sizeof(float), cudaMemcpyDeviceToHost));
        val += src.wavelet[t];
        CUDA_CHECK(cudaMemcpy(state.d_u_curr.data() + src_idx, &val,
                              sizeof(float), cudaMemcpyHostToDevice));

        // Save source wavefield AFTER source injection, BEFORE stencil
        src_snapshots[t].resize(N);
        CUDA_CHECK(cudaMemcpy(src_snapshots[t].data(),
                              state.d_u_curr.data(), bytes,
                              cudaMemcpyDeviceToHost));

        // Stencil
        dispatch_stencil(state.d_u_prev.data(), state.d_u_curr.data(),
                         state.d_u_next.data(), state.d_vel.data(),
                         grid.nx, grid.ny, grid.nz, kernel);

        // Absorbing boundary
        int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;
        apply_sponge_boundary(state.d_u_next.data(), grid.nx, grid.ny, grid.nz,
                              sponge_width, 0.015f);

        // Rotate: prev←curr, curr←next
        state.d_u_prev.swap(state.d_u_curr);
        state.d_u_curr.swap(state.d_u_next);

        // Record receivers (from curr after stencil & rotate)
        launch_record_receivers(state.d_u_curr.data(), d_traces.data(),
                                d_rx.data(), d_ry.data(), d_rz.data(),
                                rec.num_receivers, grid.nx, grid.ny,
                                t, grid.nt);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[RTM] Forward pass complete.\n");

    // Copy traces to host
    CUDA_CHECK(cudaMemcpy(rec.traces, d_traces.data(),
                          (size_t)rec.num_receivers * grid.nt * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_trace = 0.0f;
    for (int i = 0; i < rec.num_receivers * grid.nt; i++) {
        float v = rec.traces[i]; if (v < 0) v = -v;
        if (v > max_trace) max_trace = v;
    }
    printf("[RTM] Max receiver trace amplitude: %.6e\n", max_trace);

    // PHASE 1b: DIRECT-ARRIVAL SUBTRACTION (if background velocity provided)
    if (h_vel_background) {
        printf("[RTM] Running direct-arrival simulation (homogeneous model)...\n");

        // Save the actual velocity model, load background velocity
        CudaBuffer<float> d_vel_actual(N);
        CUDA_CHECK(cudaMemcpy(d_vel_actual.data(), state.d_vel.data(), bytes,
                              cudaMemcpyDeviceToDevice));
        state.d_vel.copyFromHost(h_vel_background, N);

        // Allocate buffer for direct-arrival traces
        CudaBuffer<float> d_traces_direct((size_t)rec.num_receivers * grid.nt);
        d_traces_direct.zero();

        // Re-run forward simulation with homogeneous velocity
        state.d_u_prev.zero();
        state.d_u_curr.zero();
        state.d_u_next.zero();

        int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;
        for (int t = 0; t < grid.nt; t++) {
            // Inject source
            int src_idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
            float val;
            CUDA_CHECK(cudaMemcpy(&val, state.d_u_curr.data() + src_idx,
                                  sizeof(float), cudaMemcpyDeviceToHost));
            val += src.wavelet[t];
            CUDA_CHECK(cudaMemcpy(state.d_u_curr.data() + src_idx, &val,
                                  sizeof(float), cudaMemcpyHostToDevice));

            // Stencil + sponge + rotate
            dispatch_stencil(state.d_u_prev.data(), state.d_u_curr.data(),
                             state.d_u_next.data(), state.d_vel.data(),
                             grid.nx, grid.ny, grid.nz, kernel);
            apply_sponge_boundary(state.d_u_next.data(), grid.nx, grid.ny, grid.nz,
                                  sponge_width, 0.015f);
            state.d_u_prev.swap(state.d_u_curr);
            state.d_u_curr.swap(state.d_u_next);

            // Record at receivers
            launch_record_receivers(state.d_u_curr.data(), d_traces_direct.data(),
                                    d_rx.data(), d_ry.data(), d_rz.data(),
                                    rec.num_receivers, grid.nx, grid.ny,
                                    t, grid.nt);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Subtract direct-arrival traces from actual traces on device:
        //   d_traces = d_traces - d_traces_direct  (reflection-only)
        size_t trace_count = (size_t)rec.num_receivers * grid.nt;
        std::vector<float> h_actual(trace_count), h_direct(trace_count);
        CUDA_CHECK(cudaMemcpy(h_actual.data(), d_traces.data(),
                              trace_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_direct.data(), d_traces_direct.data(),
                              trace_count * sizeof(float), cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        for (size_t i = 0; i < trace_count; i++) {
            h_actual[i] -= h_direct[i];
            float v = h_actual[i]; if (v < 0) v = -v;
            if (v > max_diff) max_diff = v;
        }
        CUDA_CHECK(cudaMemcpy(d_traces.data(), h_actual.data(),
                              trace_count * sizeof(float), cudaMemcpyHostToDevice));

        printf("[RTM] Direct-arrival subtracted. Reflection-only max=%.6e\n", max_diff);

        // Restore actual velocity model
        CUDA_CHECK(cudaMemcpy(state.d_vel.data(), d_vel_actual.data(), bytes,
                              cudaMemcpyDeviceToDevice));
    }

    // PHASE 2: BACKWARD PROPAGATION + IMAGING
    printf("[RTM] Backward propagation + imaging...\n");

    DeviceState rcv_state;
    rcv_state.d_u_prev = CudaBuffer<float>(N);
    rcv_state.d_u_curr = CudaBuffer<float>(N);
    rcv_state.d_u_next = CudaBuffer<float>(N);
    rcv_state.d_vel    = CudaBuffer<float>(N);

    rcv_state.d_u_prev.zero();
    rcv_state.d_u_curr.zero();
    rcv_state.d_u_next.zero();

    CUDA_CHECK(cudaMemcpy(rcv_state.d_vel.data(), state.d_vel.data(),
                          bytes, cudaMemcpyDeviceToDevice));

    // Walk backward: t = nt-1, nt-2, ..., 0
    for (int t = grid.nt - 1; t >= 0; t--) {
        // 1. Inject receiver trace at time t (time-reversed adjoint source)
        launch_inject_receivers(rcv_state.d_u_curr.data(), d_traces.data(),
                                d_rx.data(), d_ry.data(), d_rz.data(),
                                rec.num_receivers, grid.nx, grid.ny,
                                t, grid.nt);

        // 2. Load source snapshot at forward time t (BEFORE stencil — matches
        //    forward pass where snapshots were saved post-inject, pre-stencil)
        CUDA_CHECK(cudaMemcpy(state.d_u_curr.data(),
                              src_snapshots[t].data(), bytes,
                              cudaMemcpyHostToDevice));

        // 3. Imaging condition BEFORE stencil: image += src(t) * rcv(t)
        //    Both wavefields are post-inject, pre-stencil = same physical time
        apply_imaging_condition(state.d_u_curr.data(),
                                rcv_state.d_u_curr.data(),
                                d_image.data(), N);

        // 4. Stencil: propagate backward wavefield
        dispatch_stencil(rcv_state.d_u_prev.data(), rcv_state.d_u_curr.data(),
                         rcv_state.d_u_next.data(), rcv_state.d_vel.data(),
                         grid.nx, grid.ny, grid.nz, kernel);

        // 5. Absorbing boundary
        int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;
        apply_sponge_boundary(rcv_state.d_u_next.data(), grid.nx, grid.ny, grid.nz,
                              sponge_width, 0.015f);

        // 6. Rotate backward buffers
        rcv_state.d_u_prev.swap(rcv_state.d_u_curr);
        rcv_state.d_u_curr.swap(rcv_state.d_u_next);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[RTM] Imaging complete.\n");

    // Copy raw cross-correlation image to host (Laplacian filter applied in viewer)
    CUDA_CHECK(cudaMemcpy(h_image, d_image.data(), bytes,
                          cudaMemcpyDeviceToHost));

    // Cleanup
    rcv_state.d_u_prev = CudaBuffer<float>();
    rcv_state.d_u_curr = CudaBuffer<float>();
    rcv_state.d_u_next = CudaBuffer<float>();
    rcv_state.d_vel    = CudaBuffer<float>();
}
