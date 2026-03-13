// ─── Richter: 2D FWI Orchestrator ───────────────────────────────────
// Full Waveform Inversion in 2D with:
// - Laplacian-based imaging condition (proper FWI gradient)
// - Checkpointed forward/backward propagation
// - Backtracking line search
// - Multi-scale frequency staging
// - Illumination preconditioning

#include "richter/fwi_2d.h"
#include "richter/fwi.h"      // reuse flat-array kernels
#include "richter/rtm.h"      // normalize_by_illumination
#include "richter/wavelet.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "[Richter FWI2D] CUDA error at %s:%d — %s\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// Checkpoint: stores (u_prev, u_curr) at a given time step
struct Checkpoint2D {
    std::vector<float> h_u_prev;
    std::vector<float> h_u_curr;
    int time_step;
};

// ─── Forward-only propagation (for line search) ─────────────────────

static void forward_only_2d(const Grid2D& grid, const Source2D& src,
                             DeviceState2D& state,
                             CudaBuffer<float>& d_syn_traces,
                             CudaBuffer<int>& d_rx, CudaBuffer<int>& d_rz,
                             int num_receivers)
{
    int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;

    state.d_u_prev.zero();
    state.d_u_curr.zero();
    state.d_u_next.zero();
    d_syn_traces.zero();

    for (int t = 0; t < grid.nt; t++) {
        // Inject source
        inject_source_2d(state.d_u_curr.data(), src.sx, src.sz,
                         src.wavelet[t], grid.nx, grid.nz);

        // Stencil
        launch_stencil_2d(state.d_u_prev.data(), state.d_u_curr.data(),
                          state.d_u_next.data(), state.d_vel.data(),
                          grid.nx, grid.nz);

        // Sponge
        apply_sponge_2d(state.d_u_next.data(), grid.nx, grid.nz,
                        sponge_width, 0.015f);

        // Rotate
        state.d_u_prev.swap(state.d_u_curr);
        state.d_u_curr.swap(state.d_u_next);

        // Record
        record_receivers_2d(state.d_u_curr.data(), d_syn_traces.data(),
                            d_rx.data(), d_rz.data(),
                            num_receivers, grid.nx, t, grid.nt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─── Compute total misfit across all shots ──────────────────────────

static float compute_total_misfit_2d(const Grid2D& grid,
                                      const Source2D* sources, int num_shots,
                                      DeviceState2D& state,
                                      const std::vector<CudaBuffer<float>>& d_obs_traces,
                                      CudaBuffer<float>& d_syn_traces,
                                      CudaBuffer<int>& d_rx, CudaBuffer<int>& d_rz,
                                      int num_receivers)
{
    float total_misfit = 0.0f;
    for (int s = 0; s < num_shots; s++) {
        forward_only_2d(grid, sources[s], state,
                        d_syn_traces, d_rx, d_rz, num_receivers);

        total_misfit += compute_misfit_only(d_syn_traces.data(),
                                            d_obs_traces[s].data(),
                                            num_receivers, grid.nt);
    }
    return total_misfit;
}

// ─── Gradient computation for a single shot ─────────────────────────

static float compute_gradient_2d(const Grid2D& grid, const Source2D& src,
                                  DeviceState2D& state,
                                  CudaBuffer<float>& d_obs_traces,
                                  CudaBuffer<float>& d_residual,
                                  CudaBuffer<float>& d_syn_traces,
                                  CudaBuffer<int>& d_rx, CudaBuffer<int>& d_rz,
                                  int num_receivers,
                                  CudaBuffer<float>& d_gradient,
                                  CudaBuffer<float>& d_illum,
                                  const FWIConfig2D& config)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);
    int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;
    int cp_interval = std::max(config.checkpoint_interval, 1);

    // Auto-scale checkpoint interval based on memory budget
    {
        size_t cp_budget = (size_t)2 * 1024 * 1024 * 1024;  // 2 GB for 2D is plenty
        int max_cps = std::max(3, (int)(cp_budget / (2 * bytes)));
        int needed_cps = (grid.nt + cp_interval - 1) / cp_interval + 1;
        if (needed_cps > max_cps) {
            cp_interval = (grid.nt + max_cps - 2) / (max_cps - 1);
        }
    }

    int num_checkpoints = (grid.nt + cp_interval - 1) / cp_interval + 1;

    // PHASE 1: FORWARD PROPAGATION with checkpointing + synthetic trace recording
    std::vector<Checkpoint2D> checkpoints(num_checkpoints);

    state.d_u_prev.zero();
    state.d_u_curr.zero();
    state.d_u_next.zero();
    d_syn_traces.zero();

    checkpoints[0].h_u_prev.resize(N);
    checkpoints[0].h_u_curr.resize(N);
    checkpoints[0].time_step = 0;
    CUDA_CHECK(cudaMemcpy(checkpoints[0].h_u_prev.data(), state.d_u_prev.data(),
                          bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(checkpoints[0].h_u_curr.data(), state.d_u_curr.data(),
                          bytes, cudaMemcpyDeviceToHost));

    for (int t = 0; t < grid.nt; t++) {
        inject_source_2d(state.d_u_curr.data(), src.sx, src.sz,
                         src.wavelet[t], grid.nx, grid.nz);

        launch_stencil_2d(state.d_u_prev.data(), state.d_u_curr.data(),
                          state.d_u_next.data(), state.d_vel.data(),
                          grid.nx, grid.nz);
        apply_sponge_2d(state.d_u_next.data(), grid.nx, grid.nz,
                        sponge_width, 0.015f);

        state.d_u_prev.swap(state.d_u_curr);
        state.d_u_curr.swap(state.d_u_next);

        // Record synthetic traces
        record_receivers_2d(state.d_u_curr.data(), d_syn_traces.data(),
                            d_rx.data(), d_rz.data(),
                            num_receivers, grid.nx, t, grid.nt);

        // Save checkpoint
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

    // PHASE 1.5: COMPUTE RESIDUAL (syn - obs) and L2 misfit
    float shot_misfit = compute_residual_and_misfit(
        d_syn_traces.data(), d_obs_traces.data(),
        d_residual.data(), num_receivers, grid.nt);

    // Mute direct arrivals in the residual before backward propagation.
    // This removes surface-dominated energy that doesn't carry deep-structure info.
    if (config.mute_direct_v > 0.0f) {
        mute_direct_arrivals_2d(d_residual.data(),
                                 d_rx.data(), d_rz.data(),
                                 src.sx, src.sz,
                                 num_receivers, grid.nt,
                                 grid.dt, grid.dx,
                                 config.mute_direct_v,
                                 config.mute_taper_samples);
    }

    // PHASE 2: BACKWARD PROPAGATION + FWI IMAGING
    DeviceState2D adj_state;
    adj_state.d_u_prev = CudaBuffer<float>(N);
    adj_state.d_u_curr = CudaBuffer<float>(N);
    adj_state.d_u_next = CudaBuffer<float>(N);
    adj_state.d_vel    = CudaBuffer<float>(N);

    adj_state.d_u_prev.zero();
    adj_state.d_u_curr.zero();
    adj_state.d_u_next.zero();
    CUDA_CHECK(cudaMemcpy(adj_state.d_vel.data(), state.d_vel.data(),
                          bytes, cudaMemcpyDeviceToDevice));

    // For 2D grids, snapshots easily fit on GPU
    CudaBuffer<float> d_snapshot_pool;
    bool use_gpu_snapshots = false;
    size_t pool_elems = (size_t)cp_interval * N;
    use_gpu_snapshots = CudaBuffer<float>::tryAlloc(d_snapshot_pool, pool_elems);

    std::vector<std::vector<float>> h_snapshots;
    int max_snap_slots = cp_interval;
    if (!use_gpu_snapshots) {
        size_t snap_budget = (size_t)2 * 1024 * 1024 * 1024;
        max_snap_slots = std::max(1, std::min(cp_interval, (int)(snap_budget / bytes)));
        h_snapshots.resize(max_snap_slots);
    }

    int num_segments = (grid.nt + cp_interval - 1) / cp_interval;

    for (int seg = num_segments - 1; seg >= 0; seg--) {
        int t_start = seg * cp_interval;
        int t_end   = std::min(t_start + cp_interval, grid.nt);
        int seg_len = t_end - t_start;

        int chunk_size = std::min(seg_len, max_snap_slots);
        int num_chunks = (seg_len + chunk_size - 1) / chunk_size;

        for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
            int c_start = chunk * chunk_size;
            int c_end   = std::min(c_start + chunk_size, seg_len);
            int c_len   = c_end - c_start;

            // Restore checkpoint
            CUDA_CHECK(cudaMemcpy(state.d_u_prev.data(),
                                  checkpoints[seg].h_u_prev.data(),
                                  bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state.d_u_curr.data(),
                                  checkpoints[seg].h_u_curr.data(),
                                  bytes, cudaMemcpyHostToDevice));

            // Fast-forward to chunk start
            for (int i = 0; i < c_start; i++) {
                int t = t_start + i;
                inject_source_2d(state.d_u_curr.data(), src.sx, src.sz,
                                 src.wavelet[t], grid.nx, grid.nz);
                launch_stencil_2d(state.d_u_prev.data(), state.d_u_curr.data(),
                                  state.d_u_next.data(), state.d_vel.data(),
                                  grid.nx, grid.nz);
                apply_sponge_2d(state.d_u_next.data(), grid.nx, grid.nz,
                                sponge_width, 0.015f);
                state.d_u_prev.swap(state.d_u_curr);
                state.d_u_curr.swap(state.d_u_next);
            }

            // Save snapshots for this chunk
            for (int i = c_start; i < c_end; i++) {
                int t = t_start + i;
                int snap_idx = i - c_start;

                inject_source_2d(state.d_u_curr.data(), src.sx, src.sz,
                                 src.wavelet[t], grid.nx, grid.nz);

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

                launch_stencil_2d(state.d_u_prev.data(), state.d_u_curr.data(),
                                  state.d_u_next.data(), state.d_vel.data(),
                                  grid.nx, grid.nz);
                apply_sponge_2d(state.d_u_next.data(), grid.nx, grid.nz,
                                sponge_width, 0.015f);
                state.d_u_prev.swap(state.d_u_curr);
                state.d_u_curr.swap(state.d_u_next);
            }

            // Walk backward through chunk — inject residuals, apply FWI imaging
            for (int t = t_start + c_end - 1; t >= t_start + c_start; t--) {
                int snap_idx = t - (t_start + c_start);

                inject_receivers_2d(adj_state.d_u_curr.data(), d_residual.data(),
                                    d_rx.data(), d_rz.data(),
                                    num_receivers, grid.nx, t, grid.nt);

                // FWI imaging: Laplacian(source) * adjoint
                if (use_gpu_snapshots) {
                    float* snap_ptr = d_snapshot_pool.data() + (size_t)snap_idx * N;
                    apply_fwi_imaging_2d(snap_ptr, adj_state.d_u_curr.data(),
                                         d_gradient.data(), d_illum.data(),
                                         grid.nx, grid.nz);
                } else {
                    CUDA_CHECK(cudaMemcpy(state.d_u_curr.data(),
                                          h_snapshots[snap_idx].data(),
                                          bytes, cudaMemcpyHostToDevice));
                    apply_fwi_imaging_2d(state.d_u_curr.data(),
                                         adj_state.d_u_curr.data(),
                                         d_gradient.data(), d_illum.data(),
                                         grid.nx, grid.nz);
                }

                // Backward adjoint stencil
                launch_stencil_2d(adj_state.d_u_prev.data(), adj_state.d_u_curr.data(),
                                  adj_state.d_u_next.data(), adj_state.d_vel.data(),
                                  grid.nx, grid.nz);
                apply_sponge_2d(adj_state.d_u_next.data(), grid.nx, grid.nz,
                                sponge_width, 0.015f);
                adj_state.d_u_prev.swap(adj_state.d_u_curr);
                adj_state.d_u_curr.swap(adj_state.d_u_next);
            }
        }

        // Free checkpoint memory for this segment
        std::vector<float>().swap(checkpoints[seg].h_u_prev);
        std::vector<float>().swap(checkpoints[seg].h_u_curr);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup adjoint state
    adj_state.d_u_prev = CudaBuffer<float>();
    adj_state.d_u_curr = CudaBuffer<float>();
    adj_state.d_u_next = CudaBuffer<float>();
    adj_state.d_vel    = CudaBuffer<float>();

    return shot_misfit;
}

// ─── Main 2D FWI Orchestrator ───────────────────────────────────────

void richter_fwi_2d(const Grid2D& grid,
                    const Source2D* sources, int num_shots,
                    const ReceiverSet2D& rec,
                    const float* const* h_obs_traces,
                    const float* h_vel_initial,
                    float* h_vel_output,
                    const FWIConfig2D& config,
                    FWIResult2D* result)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);
    size_t trace_size = (size_t)rec.num_receivers * grid.nt;

    printf("═══════════════════════════════════════\n");
    printf("  Richter — 2D Full Waveform Inversion\n");
    printf("  Grid: %dx%d  Shots: %d  Steps: %d\n", grid.nx, grid.nz, num_shots, grid.nt);
    printf("  Velocity bounds: %.0f - %.0f m/s\n", config.v_min, config.v_max);
    printf("═══════════════════════════════════════\n\n");

    // Initialize device state
    DeviceState2D state;
    state.d_u_prev = CudaBuffer<float>(N);
    state.d_u_curr = CudaBuffer<float>(N);
    state.d_u_next = CudaBuffer<float>(N);
    state.d_vel    = CudaBuffer<float>(N);
    state.d_wavelet = CudaBuffer<float>(grid.nt);

    // Convert initial velocity to coefficient and upload
    {
        std::vector<float> h_coeff(N);
        float dt2_dx2 = (grid.dt * grid.dt) / (grid.dx * grid.dx);
        for (size_t i = 0; i < N; i++) {
            h_coeff[i] = h_vel_initial[i] * h_vel_initial[i] * dt2_dx2;
        }
        state.d_vel.copyFromHost(h_coeff.data(), N);
    }

    // Upload observed traces to device (one buffer per shot)
    std::vector<CudaBuffer<float>> d_obs_traces(num_shots);
    for (int s = 0; s < num_shots; s++) {
        d_obs_traces[s] = CudaBuffer<float>(trace_size);
        d_obs_traces[s].copyFromHost(h_obs_traces[s], trace_size);
    }

    // Upload receiver positions
    CudaBuffer<int> d_rx(rec.num_receivers);
    CudaBuffer<int> d_rz(rec.num_receivers);
    d_rx.copyFromHost(rec.rx, rec.num_receivers);
    d_rz.copyFromHost(rec.rz, rec.num_receivers);

    // Allocate working buffers
    CudaBuffer<float> d_gradient(N);
    CudaBuffer<float> d_illum(N);
    CudaBuffer<float> d_smooth_temp(N);
    CudaBuffer<float> d_vel_backup(N);
    CudaBuffer<float> d_syn_traces(trace_size);
    CudaBuffer<float> d_residual(trace_size);

    // Result tracking
    if (result) {
        result->iterations_completed = 0;
        result->misfit_history.clear();
        result->step_size_history.clear();
    }

    // Determine frequency stages
    int num_stages = config.num_frequency_stages;
    if (num_stages <= 0) num_stages = 1;

    std::vector<float> h_wavelet(grid.nt);
    std::vector<Source2D> stage_sources(num_shots);

    int total_iterations = 0;

    for (int stage = 0; stage < num_stages; stage++) {
        float stage_freq;
        int stage_iters;

        if (config.num_frequency_stages > 1 && config.frequency_stages && config.iterations_per_stage) {
            stage_freq = config.frequency_stages[stage];
            stage_iters = config.iterations_per_stage[stage];
        } else {
            stage_freq = sources[0].peak_freq;
            stage_iters = config.max_iterations;
        }

        printf("\n[FWI2D] ═══ Frequency Stage %d/%d  f=%.1f Hz  iters=%d ═══\n",
               stage + 1, num_stages, stage_freq, stage_iters);

        // Generate wavelet for this frequency stage
        generate_ricker_wavelet(h_wavelet.data(), grid.nt, grid.dt, stage_freq);
        state.d_wavelet.copyFromHost(h_wavelet.data(), grid.nt);

        // Create source configurations for this stage
        for (int s = 0; s < num_shots; s++) {
            stage_sources[s] = sources[s];
            stage_sources[s].peak_freq = stage_freq;
            stage_sources[s].wavelet = h_wavelet.data();
        }

        float prev_misfit = -1.0f;
        float step_size = config.initial_step_size;

        // L-BFGS state
        const int lbfgs_m = 7;  // number of correction pairs to store
        std::vector<std::vector<float>> lbfgs_s(lbfgs_m, std::vector<float>(N, 0.0f));  // s_k = x_{k+1} - x_k
        std::vector<std::vector<float>> lbfgs_y(lbfgs_m, std::vector<float>(N, 0.0f));  // y_k = g_{k+1} - g_k
        std::vector<float> lbfgs_rho(lbfgs_m, 0.0f);
        int lbfgs_count = 0;     // number of stored pairs
        int lbfgs_oldest = 0;    // ring buffer position
        std::vector<float> h_prev_grad(N, 0.0f);
        std::vector<float> h_prev_coeff(N, 0.0f);
        std::vector<float> h_search_dir(N, 0.0f);

        CudaBuffer<float> d_search_dir(N);

        for (int iter = 0; iter < stage_iters; iter++) {
            total_iterations++;

            d_gradient.zero();
            d_illum.zero();

            float total_misfit = 0.0f;

            for (int s = 0; s < num_shots; s++) {
                float shot_misfit = compute_gradient_2d(
                    grid, stage_sources[s], state,
                    d_obs_traces[s], d_residual, d_syn_traces,
                    d_rx, d_rz, rec.num_receivers,
                    d_gradient, d_illum,
                    config);
                total_misfit += shot_misfit;
            }

            // Dump raw gradient (before any preconditioning) at iter 1
            if (total_iterations == 1) {
                std::vector<float> h_raw(N);
                CUDA_CHECK(cudaMemcpy(h_raw.data(), d_gradient.data(), bytes,
                                      cudaMemcpyDeviceToHost));
                FILE* rf = fopen("fwi_2d_gradient_raw.npy", "wb");
                if (rf) {
                    char hdr[128];
                    int hl = snprintf(hdr, sizeof(hdr),
                        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",
                        grid.nz, grid.nx);
                    int total_hdr = 10 + hl + 1;
                    int pad = 64 - (total_hdr % 64);
                    if (pad == 64) pad = 0;
                    int padded = hl + pad + 1;
                    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
                    fwrite(magic, 1, 6, rf);
                    unsigned char ver[] = {1, 0};
                    fwrite(ver, 1, 2, rf);
                    unsigned short phl = (unsigned short)padded;
                    fwrite(&phl, 2, 1, rf);
                    fwrite(hdr, 1, hl, rf);
                    for (int p = 0; p < pad; p++) fputc(' ', rf);
                    fputc('\n', rf);
                    fwrite(h_raw.data(), sizeof(float), N, rf);
                    fclose(rf);
                    printf("       [DEBUG] Saved RAW gradient to fwi_2d_gradient_raw.npy\n");
                }
                // Also dump illumination
                std::vector<float> h_illum_dump(N);
                CUDA_CHECK(cudaMemcpy(h_illum_dump.data(), d_illum.data(), bytes,
                                      cudaMemcpyDeviceToHost));
                FILE* ilf = fopen("fwi_2d_illumination.npy", "wb");
                if (ilf) {
                    char hdr[128];
                    int hl = snprintf(hdr, sizeof(hdr),
                        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",
                        grid.nz, grid.nx);
                    int total_hdr = 10 + hl + 1;
                    int pad = 64 - (total_hdr % 64);
                    if (pad == 64) pad = 0;
                    int padded = hl + pad + 1;
                    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
                    fwrite(magic, 1, 6, ilf);
                    unsigned char ver[] = {1, 0};
                    fwrite(ver, 1, 2, ilf);
                    unsigned short phl = (unsigned short)padded;
                    fwrite(&phl, 2, 1, ilf);
                    fwrite(hdr, 1, hl, ilf);
                    for (int p = 0; p < pad; p++) fputc(' ', ilf);
                    fputc('\n', ilf);
                    fwrite(h_illum_dump.data(), sizeof(float), N, ilf);
                    fclose(ilf);
                    printf("       [DEBUG] Saved illumination to fwi_2d_illumination.npy\n");
                }
            }

            // Smooth gradient
            if (config.gradient_smooth_sigma > 0.0f) {
                smooth_gradient_2d(d_gradient.data(), d_smooth_temp.data(),
                                   grid.nx, grid.nz,
                                   config.gradient_smooth_sigma);
            }

            // Illumination normalization (pseudo-Hessian preconditioning)
            // Fixes lateral bias: boosts center relative to near-source regions
            {
                std::vector<float> h_illum(N);
                CUDA_CHECK(cudaMemcpy(h_illum.data(), d_illum.data(), bytes,
                                      cudaMemcpyDeviceToHost));
                float max_illum = 0.0f;
                for (size_t i = 0; i < N; i++) {
                    if (h_illum[i] > max_illum) max_illum = h_illum[i];
                }
                float epsilon = 1e-2f * max_illum;
                normalize_by_illumination(d_gradient.data(), d_illum.data(), N, epsilon);
            }

            // Water mask
            if (config.water_depth > 0) {
                apply_water_mask_2d(d_gradient.data(), grid.nx, grid.nz,
                                    config.water_depth);
            }

            // Sponge zone gradient mask: zero gradient at left/right/bottom boundaries
            {
                int sponge_w = (grid.nx < 64) ? grid.nx / 6 : 20;
                apply_sponge_gradient_mask_2d(d_gradient.data(), grid.nx, grid.nz,
                                              sponge_w);
            }

            // Depth scaling: compensate for geometric spreading decay with depth
            if (config.depth_scale_power > 0.0f) {
                apply_depth_scaling_2d(d_gradient.data(), grid.nx, grid.nz,
                                       config.water_depth, config.depth_scale_power);
            }

            // Layer stripping: freeze shallow velocities after initial convergence
            // to force the optimizer to modify deeper features (e.g. the lens)
            bool layer_strip_active = (config.layer_strip_iter > 0 &&
                                       total_iterations > config.layer_strip_iter);
            if (layer_strip_active) {
                if (total_iterations == config.layer_strip_iter + 1) {
                    printf("       [LAYER STRIP] Activated: freezing z < %d (taper=%d)\n",
                           config.layer_strip_depth, config.layer_strip_taper);
                    // Reset L-BFGS state: old curvature history is invalid
                    std::fill(h_search_dir.begin(), h_search_dir.end(), 0.0f);
                    lbfgs_count = 0;
                    lbfgs_oldest = 0;
                    step_size = config.initial_step_size;
                }
                apply_shallow_freeze_2d(d_gradient.data(), grid.nx, grid.nz,
                                         config.layer_strip_depth,
                                         config.layer_strip_taper);
            }

            // Download preconditioned gradient + current model
            std::vector<float> h_grad(N);
            std::vector<float> h_vel(N);
            CUDA_CHECK(cudaMemcpy(h_grad.data(), d_gradient.data(), bytes,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_vel.data(), state.d_vel.data(), bytes,
                                  cudaMemcpyDeviceToHost));

            // Dump gradient at early iterations for diagnostic visualization
            if (total_iterations <= 3 || total_iterations == 10 || total_iterations == 50) {
                char fname[64];
                snprintf(fname, sizeof(fname), "fwi_2d_gradient_iter%d.npy", total_iterations);
                FILE* gf = fopen(fname, "wb");
                if (gf) {
                    char hdr[128];
                    int hl = snprintf(hdr, sizeof(hdr),
                        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",
                        grid.nz, grid.nx);
                    int total_hdr = 10 + hl + 1;
                    int pad = 64 - (total_hdr % 64);
                    if (pad == 64) pad = 0;
                    int padded = hl + pad + 1;
                    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
                    fwrite(magic, 1, 6, gf);
                    unsigned char ver[] = {1, 0};
                    fwrite(ver, 1, 2, gf);
                    unsigned short phl = (unsigned short)padded;
                    fwrite(&phl, 2, 1, gf);
                    fwrite(hdr, 1, hl, gf);
                    for (int p = 0; p < pad; p++) fputc(' ', gf);
                    fputc('\n', gf);
                    fwrite(h_grad.data(), sizeof(float), N, gf);
                    fclose(gf);
                    printf("       [DEBUG] Saved gradient to %s\n", fname);
                }
            }

            float max_coeff = 0.0f;
            for (size_t i = 0; i < N; i++) {
                if (h_vel[i] > max_coeff) max_coeff = h_vel[i];
            }

            // ─── L-BFGS: update correction pairs from previous iteration ───
            if (iter > 0) {
                // s = x_curr - x_prev (model difference)
                // y = g_curr - g_prev (gradient difference)
                int slot = lbfgs_oldest;
                float ys = 0.0f;
                for (size_t i = 0; i < N; i++) {
                    lbfgs_s[slot][i] = h_vel[i] - h_prev_coeff[i];
                    lbfgs_y[slot][i] = h_grad[i] - h_prev_grad[i];
                    ys += lbfgs_s[slot][i] * lbfgs_y[slot][i];
                }

                if (ys > 1e-30f) {
                    lbfgs_rho[slot] = 1.0f / ys;
                    lbfgs_oldest = (lbfgs_oldest + 1) % lbfgs_m;
                    if (lbfgs_count < lbfgs_m) lbfgs_count++;
                } else {
                    // Skip this pair — curvature condition not satisfied
                    printf("       [L-BFGS] Skipping pair (ys=%.2e)\n", ys);
                }
            }

            // Save current gradient & model for next iteration's s/y computation
            h_prev_grad = h_grad;
            h_prev_coeff = h_vel;

            // ─── L-BFGS two-loop recursion ───
            if (lbfgs_count == 0) {
                // First iteration: steepest descent
                for (size_t i = 0; i < N; i++) h_search_dir[i] = h_grad[i];
                printf("       SD (L-BFGS cold start)\n");
            } else {
                // q = gradient
                std::vector<float> q(h_grad);
                std::vector<float> alpha(lbfgs_m, 0.0f);

                // First loop: newest to oldest
                for (int j = lbfgs_count - 1; j >= 0; j--) {
                    int idx = (lbfgs_oldest - 1 - (lbfgs_count - 1 - j) + lbfgs_m * 2) % lbfgs_m;
                    float dot = 0.0f;
                    for (size_t i = 0; i < N; i++) dot += lbfgs_s[idx][i] * q[i];
                    alpha[j] = lbfgs_rho[idx] * dot;
                    for (size_t i = 0; i < N; i++) q[i] -= alpha[j] * lbfgs_y[idx][i];
                }

                // Initial Hessian scaling: gamma = s^T y / y^T y (most recent pair)
                int newest = (lbfgs_oldest - 1 + lbfgs_m) % lbfgs_m;
                float yy = 0.0f, sy = 0.0f;
                for (size_t i = 0; i < N; i++) {
                    yy += lbfgs_y[newest][i] * lbfgs_y[newest][i];
                    sy += lbfgs_s[newest][i] * lbfgs_y[newest][i];
                }
                float gamma = (yy > 1e-30f) ? sy / yy : 1.0f;

                // r = gamma * q
                std::vector<float>& r = h_search_dir;
                for (size_t i = 0; i < N; i++) r[i] = gamma * q[i];

                // Second loop: oldest to newest
                for (int j = 0; j < lbfgs_count; j++) {
                    int idx = (lbfgs_oldest - lbfgs_count + j + lbfgs_m * 2) % lbfgs_m;
                    float dot = 0.0f;
                    for (size_t i = 0; i < N; i++) dot += lbfgs_y[idx][i] * r[i];
                    float beta_j = lbfgs_rho[idx] * dot;
                    for (size_t i = 0; i < N; i++) r[i] += (alpha[j] - beta_j) * lbfgs_s[idx][i];
                }

                printf("       L-BFGS (m=%d, gamma=%.3e)\n", lbfgs_count, gamma);
            }

            // Descent check — fall back to steepest descent if needed
            float descent_check = 0.0f;
            for (size_t i = 0; i < N; i++) {
                descent_check += h_grad[i] * h_search_dir[i];
            }
            if (descent_check <= 0.0f) {
                for (size_t i = 0; i < N; i++) h_search_dir[i] = h_grad[i];
                printf("       [L-BFGS] Non-descent direction, falling back to SD\n");
                // Reset L-BFGS history
                lbfgs_count = 0;
                lbfgs_oldest = 0;
            }

            // Direction normalization
            // For SD: scale to max_coeff so step_size is a fraction of model magnitude
            // For L-BFGS: scale to max_coeff but preserve relative structure
            // (L-BFGS already incorporates curvature via gamma and the two-loop recursion)
            float max_abs_dir = 0.0f;
            for (size_t i = 0; i < N; i++) {
                float a = fabsf(h_search_dir[i]);
                if (a > max_abs_dir) max_abs_dir = a;
            }

            int wd = config.water_depth > 0 ? config.water_depth : 0;
            int cx = grid.nx / 2;
            int cz = wd + (grid.nz - wd) / 3;
            size_t center_idx = (size_t)cz * grid.nx + cx;
            float dir_center = (center_idx < N) ? fabsf(h_search_dir[center_idx]) : 0.0f;
            printf("       |dir|: max=%.3e  lens=%.3e  ratio=%.4f\n",
                   max_abs_dir, dir_center,
                   max_abs_dir > 0 ? dir_center / max_abs_dir : 0.0f);

            if (max_abs_dir > 0.0f) {
                float scale = max_coeff / max_abs_dir;
                for (size_t i = 0; i < N; i++) h_search_dir[i] *= scale;
            }

            // Reset step size each iteration — prevents shrinkage death spiral
            // where step shrinks to 1e-8 and takes 20+ iterations to recover
            step_size = config.initial_step_size;

            CUDA_CHECK(cudaMemcpy(d_search_dir.data(), h_search_dir.data(), bytes,
                                  cudaMemcpyHostToDevice));

            // Backtracking line search
            CUDA_CHECK(cudaMemcpy(d_vel_backup.data(), state.d_vel.data(),
                                  bytes, cudaMemcpyDeviceToDevice));

            bool accepted = false;
            float accepted_misfit = total_misfit;
            float accepted_step = 0.0f;

            for (int ls = 0; ls < config.max_line_search_steps; ls++) {
                CUDA_CHECK(cudaMemcpy(state.d_vel.data(), d_vel_backup.data(),
                                      bytes, cudaMemcpyDeviceToDevice));
                apply_velocity_update(state.d_vel.data(), d_search_dir.data(),
                                       step_size, grid.dt, grid.dx,
                                       config.v_min, config.v_max, N);

                float trial_misfit = compute_total_misfit_2d(
                    grid, stage_sources.data(), num_shots, state,
                    d_obs_traces, d_syn_traces,
                    d_rx, d_rz, rec.num_receivers);

                if (trial_misfit < total_misfit) {
                    accepted = true;
                    accepted_misfit = trial_misfit;
                    accepted_step = step_size;
                    break;
                }
                step_size *= config.step_size_reduction;
            }

            if (!accepted) {
                CUDA_CHECK(cudaMemcpy(state.d_vel.data(), d_vel_backup.data(),
                                      bytes, cudaMemcpyDeviceToDevice));
                std::fill(h_search_dir.begin(), h_search_dir.end(), 0.0f);
                // Reset L-BFGS history on line search failure
                lbfgs_count = 0;
                lbfgs_oldest = 0;
                printf("[FWI2D] Iter %d: line search failed. misfit=%.6e (reset L-BFGS)\n",
                       total_iterations, total_misfit);
            } else {
                printf("[FWI2D] Iter %d: misfit=%.6e -> %.6e  step=%.2e\n",
                       total_iterations, total_misfit, accepted_misfit, accepted_step);
            }

            float current_misfit = accepted ? accepted_misfit : total_misfit;
            if (result) {
                result->misfit_history.push_back(current_misfit);
                result->step_size_history.push_back(accepted ? accepted_step : 0.0f);
                result->iterations_completed = total_iterations;
            }
            prev_misfit = current_misfit;
        }
    }

    // Convert final velocity coefficient -> physical velocity and download
    {
        CudaBuffer<float> d_vel_phys(N);
        coefficient_to_velocity(state.d_vel.data(), d_vel_phys.data(),
                                 grid.dt, grid.dx, N);
        CUDA_CHECK(cudaMemcpy(h_vel_output, d_vel_phys.data(), bytes,
                              cudaMemcpyDeviceToHost));
    }

    printf("\n[FWI2D] Complete. %d total iterations.\n", total_iterations);

    // Cleanup
    state.d_u_prev = CudaBuffer<float>();
    state.d_u_curr = CudaBuffer<float>();
    state.d_u_next = CudaBuffer<float>();
    state.d_vel    = CudaBuffer<float>();
    state.d_wavelet = CudaBuffer<float>();
}
