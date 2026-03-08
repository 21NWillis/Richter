// RTM Orchestrator
// Reverse Time Migration: forward propagation with receiver recording,
// backward propagation with receiver injection, and imaging condition.
//
// Strategy: checkpoint-based. Save (u_prev, u_curr) every cp_interval
// steps during forward pass. During backward pass, process one segment
// at a time: restore checkpoint, re-propagate to regenerate source
// snapshots, then walk backward applying the imaging condition.

#include "richter/rtm.h"
#include "richter/kernels.h"
#include "richter/wavelet.h"
#include "richter/boundary.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

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

// Helper: inject source (single-point additive). Host round-trip for simplicity.
static void inject_source(CudaBuffer<float>& d_u, const Source& src,
                           const Grid& grid, int t)
{
    int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
    float val;
    CUDA_CHECK(cudaMemcpy(&val, d_u.data() + idx,
                          sizeof(float), cudaMemcpyDeviceToHost));
    val += src.wavelet[t];
    CUDA_CHECK(cudaMemcpy(d_u.data() + idx, &val,
                          sizeof(float), cudaMemcpyHostToDevice));
}

// Helper: one forward step (inject, stencil, sponge, rotate, record receivers)
static void forward_step(DeviceState& state, const Grid& grid,
                          const Source& src, int t, KernelType kernel,
                          int sponge_width,
                          CudaBuffer<float>* d_traces,
                          CudaBuffer<int>* d_rx, CudaBuffer<int>* d_ry,
                          CudaBuffer<int>* d_rz, int num_receivers)
{
    inject_source(state.d_u_curr, src, grid, t);

    dispatch_stencil(state.d_u_prev.data(), state.d_u_curr.data(),
                     state.d_u_next.data(), state.d_vel.data(),
                     grid.nx, grid.ny, grid.nz, kernel);
    apply_sponge_boundary(state.d_u_next.data(), grid.nx, grid.ny, grid.nz,
                          sponge_width, 0.015f);

    state.d_u_prev.swap(state.d_u_curr);
    state.d_u_curr.swap(state.d_u_next);

    if (d_traces) {
        launch_record_receivers(state.d_u_curr.data(), d_traces->data(),
                                d_rx->data(), d_ry->data(), d_rz->data(),
                                num_receivers, grid.nx, grid.ny,
                                t, grid.nt);
    }
}

// Checkpoint: stores (u_prev, u_curr) at a given time step
struct Checkpoint {
    std::vector<float> h_u_prev;
    std::vector<float> h_u_curr;
    int time_step;
};

// richter_rtm
void richter_rtm(const Grid& grid, const Source& src, const ReceiverSet& rec,
                 DeviceState& state, float* h_image, KernelType kernel,
                 int checkpoint_interval,
                 const float* h_vel_background,
                 bool raw_output,
                 float* h_illum_out)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);
    int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;
    int cp_interval = std::max(checkpoint_interval, 1);

    // Auto-scale checkpoint interval to keep memory bounded.
    // Checkpoints live on host (2 fields each). Snapshots live on GPU if possible
    // (GPU pool), otherwise on host. Strategy: prefer a cp_interval that keeps
    // the GPU snapshot pool feasible, so only checkpoints consume host memory.
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        // Estimate VRAM consumed by allocations between now and snapshot pool
        size_t future_gpu = (size_t)rec.num_receivers * grid.nt * sizeof(float)  // d_traces
                          + 6 * bytes  // d_image + d_illum + rcv_state (4 fields)
                          + (size_t)rec.num_receivers * 3 * sizeof(int);  // d_rx/ry/rz
        size_t est_pool_free = (free_mem > future_gpu) ? free_mem - future_gpu : 0;

        // Max cp where GPU snapshot pool fits (pool must be < 75% of free VRAM)
        int max_gpu_cp = (est_pool_free > 0) ? (int)(est_pool_free * 3 / (4 * bytes)) : 0;

        // Check if checkpoint memory exceeds host budget (6 GB)
        size_t cp_budget = (size_t)6 * 1024 * 1024 * 1024;
        int max_cps = std::max(3, (int)(cp_budget / (2 * bytes)));
        int needed_cps = (grid.nt + cp_interval - 1) / cp_interval + 1;

        if (needed_cps > max_cps) {
            int new_cp = (grid.nt + max_cps - 2) / (max_cps - 1);

            // Clamp to max_gpu_cp if possible — keeps GPU pool feasible,
            // avoiding massive host snapshot allocation
            if (new_cp > max_gpu_cp && max_gpu_cp > cp_interval) {
                new_cp = max_gpu_cp;
            }

            int new_cps = (grid.nt + new_cp - 1) / new_cp + 1;
            printf("[RTM] Auto-scaling checkpoint interval: %d -> %d "
                   "(checkpoints: %d -> %d, %.1f GB -> %.1f GB)\n",
                   cp_interval, new_cp, needed_cps, new_cps,
                   (float)needed_cps * 2 * bytes / (1024.0f * 1024.0f * 1024.0f),
                   (float)new_cps * 2 * bytes / (1024.0f * 1024.0f * 1024.0f));
            if (new_cp <= max_gpu_cp)
                printf("[RTM] GPU snapshot pool should fit (est. %.0f MB free)\n",
                       est_pool_free / (1024.0f * 1024.0f));
            cp_interval = new_cp;
        }
    }

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

    int num_checkpoints = (grid.nt + cp_interval - 1) / cp_interval + 1;

    printf("[RTM] Checkpoint interval=%d, checkpoints=%d (%.1f MB total)\n",
           cp_interval, num_checkpoints,
           (float)num_checkpoints * 2 * bytes / (1024.0f * 1024.0f));

    // PHASE 1: FORWARD PROPAGATION with checkpoint saving
    printf("[RTM] Forward propagation (%d steps)...\n", grid.nt);

    std::vector<Checkpoint> checkpoints(num_checkpoints);

    state.d_u_prev.zero();
    state.d_u_curr.zero();
    state.d_u_next.zero();

    // Save initial checkpoint (t=0, before any steps)
    checkpoints[0].h_u_prev.resize(N);
    checkpoints[0].h_u_curr.resize(N);
    checkpoints[0].time_step = 0;
    CUDA_CHECK(cudaMemcpy(checkpoints[0].h_u_prev.data(), state.d_u_prev.data(),
                          bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(checkpoints[0].h_u_curr.data(), state.d_u_curr.data(),
                          bytes, cudaMemcpyDeviceToHost));

    for (int t = 0; t < grid.nt; t++) {
        forward_step(state, grid, src, t, kernel, sponge_width,
                     &d_traces, &d_rx, &d_ry, &d_rz, rec.num_receivers);

        // Save checkpoint after every cp_interval steps
        if ((t + 1) % cp_interval == 0 || t == grid.nt - 1) {
            int cp_idx = (t + 1) / cp_interval;
            if (t == grid.nt - 1 && (t + 1) % cp_interval != 0)
                cp_idx = num_checkpoints - 1;

            checkpoints[cp_idx].h_u_prev.resize(N);
            checkpoints[cp_idx].h_u_curr.resize(N);
            checkpoints[cp_idx].time_step = t + 1;
            CUDA_CHECK(cudaMemcpy(checkpoints[cp_idx].h_u_prev.data(),
                                  state.d_u_prev.data(), bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(checkpoints[cp_idx].h_u_curr.data(),
                                  state.d_u_curr.data(), bytes, cudaMemcpyDeviceToHost));
        }
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

        // Save actual velocity, load background
        CudaBuffer<float> d_vel_actual(N);
        CUDA_CHECK(cudaMemcpy(d_vel_actual.data(), state.d_vel.data(), bytes,
                              cudaMemcpyDeviceToDevice));
        state.d_vel.copyFromHost(h_vel_background, N);

        CudaBuffer<float> d_traces_direct((size_t)rec.num_receivers * grid.nt);
        d_traces_direct.zero();

        state.d_u_prev.zero();
        state.d_u_curr.zero();
        state.d_u_next.zero();

        for (int t = 0; t < grid.nt; t++) {
            forward_step(state, grid, src, t, kernel, sponge_width,
                         &d_traces_direct, &d_rx, &d_ry, &d_rz, rec.num_receivers);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

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

        // Restore actual velocity
        CUDA_CHECK(cudaMemcpy(state.d_vel.data(), d_vel_actual.data(), bytes,
                              cudaMemcpyDeviceToDevice));
    }

    // PHASE 2: BACKWARD PROPAGATION + IMAGING (checkpoint-based segments)
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


    // Decide snapshot storage: GPU pool (fast) or host fallback with sub-segments.
    // GPU pool: single contiguous VRAM allocation, one chunk per segment.
    // Host fallback: cap snapshot memory at 2 GB, split segments into chunks
    // and re-propagate from checkpoint for each chunk (trades compute for memory).
    CudaBuffer<float> d_snapshot_pool;
    std::vector<std::vector<float>> h_snapshots;
    bool use_gpu_snapshots = false;
    int max_snap_slots = cp_interval;  // default: whole segment fits

    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t pool_elems = (size_t)cp_interval * N;
        size_t pool_bytes = pool_elems * sizeof(float);

        // Only attempt GPU pool if it's small enough to likely succeed.
        // Large failed cudaMalloc calls can corrupt the CUDA driver context on
        // some systems (especially WSL2), causing crashes on subsequent shots.
        size_t max_pool_attempt = (size_t)2 * 1024 * 1024 * 1024;  // 2 GB
        if (pool_bytes <= max_pool_attempt && pool_bytes < free_mem * 3 / 4) {
            use_gpu_snapshots = CudaBuffer<float>::tryAlloc(d_snapshot_pool, pool_elems);
            if (!use_gpu_snapshots) {
                printf("[RTM] GPU snapshot pool alloc failed, falling back to host\n");
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        if (use_gpu_snapshots) {
            printf("[RTM] Using GPU snapshot pool (%.1f MB, single allocation)\n",
                   pool_bytes / (1024.0f * 1024.0f));
        } else {
            // Cap host snapshot memory at 2 GB
            size_t snap_budget = (size_t)2 * 1024 * 1024 * 1024;
            max_snap_slots = std::max(1, std::min(cp_interval, (int)(snap_budget / bytes)));
            h_snapshots.resize(max_snap_slots);

            if (max_snap_slots < cp_interval) {
                int num_chunks_est = (cp_interval + max_snap_slots - 1) / max_snap_slots;
                printf("[RTM] Using host sub-segment fallback: %d snapshot slots, "
                       "~%d chunks/segment (%.1f MB snapshots, %.1f MB VRAM free)\n",
                       max_snap_slots, num_chunks_est,
                       (float)max_snap_slots * bytes / (1024.0f * 1024.0f),
                       free_mem / (1024.0f * 1024.0f));
            } else {
                printf("[RTM] Using host snapshot fallback (%.1f MB)\n",
                       pool_bytes / (1024.0f * 1024.0f));
            }
        }
    }

    // Process segments in reverse order
    int num_segments = (grid.nt + cp_interval - 1) / cp_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * cp_interval;
        int t_end   = std::min(t_start + cp_interval, grid.nt);
        int seg_len = t_end - t_start;

        // Determine chunking for this segment
        int chunk_size = std::min(seg_len, max_snap_slots);
        int num_chunks = (seg_len + chunk_size - 1) / chunk_size;

        // Process chunks in reverse (receiver backward walk is continuous)
        for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
            int c_start = chunk * chunk_size;  // relative to segment start
            int c_end   = std::min(c_start + chunk_size, seg_len);
            int c_len   = c_end - c_start;

            // Restore checkpoint from host (kept alive for all chunks)
            CUDA_CHECK(cudaMemcpy(state.d_u_prev.data(),
                                  checkpoints[seg].h_u_prev.data(),
                                  bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state.d_u_curr.data(),
                                  checkpoints[seg].h_u_curr.data(),
                                  bytes, cudaMemcpyHostToDevice));

            // Fast-forward from segment start to chunk start (no snapshots needed)
            for (int i = 0; i < c_start; i++) {
                int t = t_start + i;
                inject_source(state.d_u_curr, src, grid, t);
                dispatch_stencil(state.d_u_prev.data(), state.d_u_curr.data(),
                                 state.d_u_next.data(), state.d_vel.data(),
                                 grid.nx, grid.ny, grid.nz, kernel);
                apply_sponge_boundary(state.d_u_next.data(), grid.nx, grid.ny, grid.nz,
                                      sponge_width, 0.015f);
                state.d_u_prev.swap(state.d_u_curr);
                state.d_u_curr.swap(state.d_u_next);
            }

            // Save snapshots for this chunk
            for (int i = c_start; i < c_end; i++) {
                int t = t_start + i;
                int snap_idx = i - c_start;

                inject_source(state.d_u_curr, src, grid, t);

                if (use_gpu_snapshots) {
                    CUDA_CHECK(cudaMemcpy(d_snapshot_pool.data() + (size_t)snap_idx * N,
                                          state.d_u_curr.data(),
                                          bytes, cudaMemcpyDeviceToDevice));
                } else {
                    h_snapshots[snap_idx].resize(N);
                    CUDA_CHECK(cudaMemcpy(h_snapshots[snap_idx].data(),
                                          state.d_u_curr.data(),
                                          bytes, cudaMemcpyDeviceToHost));
                }

                dispatch_stencil(state.d_u_prev.data(), state.d_u_curr.data(),
                                 state.d_u_next.data(), state.d_vel.data(),
                                 grid.nx, grid.ny, grid.nz, kernel);
                apply_sponge_boundary(state.d_u_next.data(), grid.nx, grid.ny, grid.nz,
                                      sponge_width, 0.015f);
                state.d_u_prev.swap(state.d_u_curr);
                state.d_u_curr.swap(state.d_u_next);
            }

            // Walk backward through this chunk
            for (int t = t_start + c_end - 1; t >= t_start + c_start; t--) {
                int snap_idx = t - (t_start + c_start);

                launch_inject_receivers(rcv_state.d_u_curr.data(), d_traces.data(),
                                        d_rx.data(), d_ry.data(), d_rz.data(),
                                        rec.num_receivers, grid.nx, grid.ny,
                                        t, grid.nt);

                if (use_gpu_snapshots) {
                    float* snap_ptr = d_snapshot_pool.data() + (size_t)snap_idx * N;
                    accumulate_source_illumination(snap_ptr, d_illum.data(), N);
                    apply_imaging_condition(snap_ptr,
                                            rcv_state.d_u_curr.data(),
                                            d_image.data(), N);
                } else {
                    CUDA_CHECK(cudaMemcpy(state.d_u_curr.data(),
                                          h_snapshots[snap_idx].data(),
                                          bytes, cudaMemcpyHostToDevice));
                    accumulate_source_illumination(state.d_u_curr.data(),
                                                   d_illum.data(), N);
                    apply_imaging_condition(state.d_u_curr.data(),
                                            rcv_state.d_u_curr.data(),
                                            d_image.data(), N);
                }

                dispatch_stencil(rcv_state.d_u_prev.data(), rcv_state.d_u_curr.data(),
                                 rcv_state.d_u_next.data(), rcv_state.d_vel.data(),
                                 grid.nx, grid.ny, grid.nz, kernel);
                apply_sponge_boundary(rcv_state.d_u_next.data(), grid.nx, grid.ny, grid.nz,
                                      sponge_width, 0.015f);
                rcv_state.d_u_prev.swap(rcv_state.d_u_curr);
                rcv_state.d_u_curr.swap(rcv_state.d_u_next);
            }
        }

        // Free checkpoint after all chunks of this segment are processed
        std::vector<float>().swap(checkpoints[seg].h_u_prev);
        std::vector<float>().swap(checkpoints[seg].h_u_curr);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[RTM] Imaging complete.\n");

    // Copy raw cross-correlation image to host
    CUDA_CHECK(cudaMemcpy(h_image, d_image.data(), bytes,
                          cudaMemcpyDeviceToHost));

    // Download illumination map
    std::vector<float> h_illum(N);
    CUDA_CHECK(cudaMemcpy(h_illum.data(), d_illum.data(), bytes,
                          cudaMemcpyDeviceToHost));

    // Copy raw illumination to caller if requested
    if (h_illum_out) {
        memcpy(h_illum_out, h_illum.data(), bytes);
    }

    if (!raw_output) {
        // Find max illumination
        float max_illum = 0.0f;
        for (size_t i = 0; i < N; i++) {
            if (h_illum[i] > max_illum) max_illum = h_illum[i];
        }
        float epsilon = 0.01f * max_illum;
        if (epsilon < 1e-12f) epsilon = 1e-12f;

        // Apply illumination normalization and source muting
        int mute_start = src.sz;
        int mute_end   = src.sz + 15;
        for (int z = 0; z < grid.nz; z++) {
            float mute = 1.0f;
            if (z <= mute_start) mute = 0.0f;
            else if (z < mute_end) mute = (float)(z - mute_start) / (mute_end - mute_start);

            for (int y = 0; y < grid.ny; y++) {
                for (int x = 0; x < grid.nx; x++) {
                    size_t idx = z * (size_t)grid.nx * grid.ny + y * grid.nx + x;
                    h_image[idx] = (h_image[idx] / (h_illum[idx] + epsilon)) * mute;
                }
            }
        }
    }

    // Cleanup
    rcv_state.d_u_prev = CudaBuffer<float>();
    rcv_state.d_u_curr = CudaBuffer<float>();
    rcv_state.d_u_next = CudaBuffer<float>();
    rcv_state.d_vel    = CudaBuffer<float>();
}

// richter_rtm_multishot
// Run RTM for multiple shots and stack the results. Accumulates raw
// cross-correlation images and illumination, then normalizes once.
void richter_rtm_multishot(const Grid& grid,
                           const Source* sources, int num_shots,
                           const ReceiverSet& rec,
                           DeviceState& state, float* h_image,
                           KernelType kernel,
                           int checkpoint_interval,
                           const float* h_vel_background)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);

    // Host accumulators for stacking
    std::vector<float> stacked_image(N, 0.0f);
    std::vector<float> stacked_illum(N, 0.0f);

    // Per-shot temporary buffers
    std::vector<float> shot_image(N);
    std::vector<float> shot_illum(N);

    printf("[RTM Multi-Shot] Stacking %d shots\n", num_shots);

    for (int s = 0; s < num_shots; s++) {
        printf("\n[RTM Multi-Shot] ═══ Shot %d/%d  src=(%d,%d,%d) ═══\n",
               s + 1, num_shots, sources[s].sx, sources[s].sy, sources[s].sz);

        // Zero per-shot buffers
        memset(shot_image.data(), 0, bytes);
        memset(shot_illum.data(), 0, bytes);

        // Run single-shot RTM in raw mode
        richter_rtm(grid, sources[s], rec, state, shot_image.data(), kernel,
                    checkpoint_interval, h_vel_background,
                    /*raw_output=*/true, /*h_illum_out=*/shot_illum.data());

        // Accumulate into stacked buffers
        for (size_t i = 0; i < N; i++) {
            stacked_image[i] += shot_image[i];
            stacked_illum[i] += shot_illum[i];
        }
    }

    printf("\n[RTM Multi-Shot] All shots complete. Normalizing stacked image...\n");

    // Find max cumulative illumination for epsilon
    float max_illum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        if (stacked_illum[i] > max_illum) max_illum = stacked_illum[i];
    }
    float epsilon = 0.01f * max_illum;
    if (epsilon < 1e-12f) epsilon = 1e-12f;

    // Source muting: use shallowest source depth across all shots
    int mute_src_z = sources[0].sz;
    for (int s = 1; s < num_shots; s++) {
        if (sources[s].sz < mute_src_z) mute_src_z = sources[s].sz;
    }
    int mute_start = mute_src_z;
    int mute_end   = mute_src_z + 15;

    // Apply illumination normalization and source muting
    for (int z = 0; z < grid.nz; z++) {
        float mute = 1.0f;
        if (z <= mute_start) mute = 0.0f;
        else if (z < mute_end) mute = (float)(z - mute_start) / (mute_end - mute_start);

        for (int y = 0; y < grid.ny; y++) {
            for (int x = 0; x < grid.nx; x++) {
                size_t idx = z * (size_t)grid.nx * grid.ny + y * grid.nx + x;
                h_image[idx] = (stacked_image[idx] / (stacked_illum[idx] + epsilon)) * mute;
            }
        }
    }

    printf("[RTM Multi-Shot] Stacking complete (%d shots).\n", num_shots);
}
