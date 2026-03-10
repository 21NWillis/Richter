// ─── Richter: CPU-Only RTM Pipeline ─────────────────────────────────
// Full Reverse Time Migration running entirely on host memory using
// the AVX2+FMA+OpenMP stencil kernel.  No CUDA dependency.

#include "richter/rtm_cpu.h"
#include "richter/kernels.h"   // launch_kernel_cpu_avx
#include "richter/wavelet.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════
//  CPU Helper Operations
// ═══════════════════════════════════════════════════════════════════

void apply_sponge_boundary_cpu(float* u, int nx, int ny, int nz,
                               int sponge_width, float damping_factor)
{
    if (sponge_width <= 0) return;
    float alpha = damping_factor * 40.0f;
    float sw = (float)sponge_width;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            int dist_z = std::min(z, nz - 1 - z);
            int dist_y = std::min(y, ny - 1 - y);

            // Skip rows entirely inside the sponge-free interior
            if (dist_z >= sponge_width && dist_y >= sponge_width) {
                // Still need to check x boundaries
                for (int x = 0; x < nx; x++) {
                    int dist_x = std::min(x, nx - 1 - x);
                    if (dist_x >= sponge_width) continue;

                    float normalized = (sw - (float)dist_x) / sw;
                    float damping = expf(-alpha * normalized * normalized);
                    int idx = z * nx * ny + y * nx + x;
                    u[idx] *= damping;
                }
                continue;
            }

            for (int x = 0; x < nx; x++) {
                int dist_x = std::min(x, nx - 1 - x);

                if (dist_x >= sponge_width && dist_y >= sponge_width && dist_z >= sponge_width)
                    continue;

                float damping = 1.0f;

                if (dist_x < sponge_width) {
                    float normalized = (sw - (float)dist_x) / sw;
                    damping *= expf(-alpha * normalized * normalized);
                }
                if (dist_y < sponge_width) {
                    float normalized = (sw - (float)dist_y) / sw;
                    damping *= expf(-alpha * normalized * normalized);
                }
                if (dist_z < sponge_width) {
                    float normalized = (sw - (float)dist_z) / sw;
                    damping *= expf(-alpha * normalized * normalized);
                }

                int idx = z * nx * ny + y * nx + x;
                u[idx] *= damping;
            }
        }
    }
}

void record_receivers_cpu(const float* u, float* traces,
                          const int* rx, const int* ry, const int* rz,
                          int num_receivers, int nx, int ny,
                          int t, int nt)
{
    for (int i = 0; i < num_receivers; i++) {
        int idx = rz[i] * nx * ny + ry[i] * nx + rx[i];
        traces[i * nt + t] = u[idx];
    }
}

void inject_receivers_cpu(float* u, const float* traces,
                          const int* rx, const int* ry, const int* rz,
                          int num_receivers, int nx, int ny,
                          int t, int nt)
{
    for (int i = 0; i < num_receivers; i++) {
        int idx = rz[i] * nx * ny + ry[i] * nx + rx[i];
        u[idx] += traces[i * nt + t];
    }
}

void apply_imaging_condition_cpu(const float* src_field,
                                 const float* rcv_field,
                                 float* image, size_t n)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        image[i] += src_field[i] * rcv_field[i];
    }
}

void accumulate_source_illumination_cpu(const float* src_field,
                                        float* illum, size_t n)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        float s = src_field[i];
        illum[i] += s * s;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  CPU Model Functions
// ═══════════════════════════════════════════════════════════════════

void richter_init_cpu(const Grid& grid, HostState& state)
{
    size_t N = grid.total_points();
    state.u_prev.assign(N, 0.0f);
    state.u_curr.assign(N, 0.0f);
    state.u_next.assign(N, 0.0f);
    state.vel.resize(N);
}

void richter_forward_cpu(const Grid& grid, const Source& src,
                         HostState& state)
{
    int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;

    for (int t = 0; t < grid.nt; t++) {
        // Inject source
        int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
        state.u_curr[idx] += src.wavelet[t];

        // Stencil
        launch_kernel_cpu_avx(state.u_prev.data(), state.u_curr.data(),
                              state.u_next.data(), state.vel.data(),
                              grid.nx, grid.ny, grid.nz);

        // Boundary
        apply_sponge_boundary_cpu(state.u_next.data(), grid.nx, grid.ny, grid.nz,
                                  sponge_width, 0.015f);

        // Rotate
        std::swap(state.u_prev, state.u_curr);
        std::swap(state.u_curr, state.u_next);
    }
}

void richter_snapshot_cpu(const Grid& grid, const HostState& state,
                          float* h_output)
{
    memcpy(h_output, state.u_curr.data(), grid.total_points() * sizeof(float));
}

void richter_cleanup_cpu(HostState& state)
{
    std::vector<float>().swap(state.u_prev);
    std::vector<float>().swap(state.u_curr);
    std::vector<float>().swap(state.u_next);
    std::vector<float>().swap(state.vel);
}

// ═══════════════════════════════════════════════════════════════════
//  CPU RTM Helpers
// ═══════════════════════════════════════════════════════════════════

struct Checkpoint_CPU {
    std::vector<float> h_u_prev;
    std::vector<float> h_u_curr;
    int time_step;
};

// One forward step: inject source, stencil, boundary, rotate, optionally record receivers
static void forward_step_cpu(HostState& state, const Grid& grid,
                              const Source& src, int t, int sponge_width,
                              float* traces,
                              const int* rx, const int* ry, const int* rz,
                              int num_receivers)
{
    // Inject source
    int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
    state.u_curr[idx] += src.wavelet[t];

    // Stencil
    launch_kernel_cpu_avx(state.u_prev.data(), state.u_curr.data(),
                          state.u_next.data(), state.vel.data(),
                          grid.nx, grid.ny, grid.nz);

    // Boundary
    apply_sponge_boundary_cpu(state.u_next.data(), grid.nx, grid.ny, grid.nz,
                              sponge_width, 0.015f);

    // Rotate
    std::swap(state.u_prev, state.u_curr);
    std::swap(state.u_curr, state.u_next);

    // Record receivers
    if (traces) {
        record_receivers_cpu(state.u_curr.data(), traces,
                             rx, ry, rz,
                             num_receivers, grid.nx, grid.ny,
                             t, grid.nt);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  richter_rtm_cpu — Single-shot CPU RTM
// ═══════════════════════════════════════════════════════════════════

void richter_rtm_cpu(const Grid& grid, const Source& src,
                     const ReceiverSet& rec,
                     HostState& state, float* h_image,
                     int checkpoint_interval,
                     const float* h_vel_background,
                     bool raw_output,
                     float* h_illum_out)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);
    int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;
    int cp_interval = std::max(checkpoint_interval, 1);

    // Auto-scale checkpoint interval to keep host memory bounded (6 GB budget)
    {
        size_t cp_budget = (size_t)6 * 1024 * 1024 * 1024;
        int max_cps = std::max(3, (int)(cp_budget / (2 * bytes)));
        int needed_cps = (grid.nt + cp_interval - 1) / cp_interval + 1;

        if (needed_cps > max_cps) {
            int new_cp = (grid.nt + max_cps - 2) / (max_cps - 1);
            int new_cps = (grid.nt + new_cp - 1) / new_cp + 1;
            printf("[RTM-CPU] Auto-scaling checkpoint interval: %d -> %d "
                   "(checkpoints: %d -> %d, %.1f GB -> %.1f GB)\n",
                   cp_interval, new_cp, needed_cps, new_cps,
                   (float)needed_cps * 2 * bytes / (1024.0f * 1024.0f * 1024.0f),
                   (float)new_cps * 2 * bytes / (1024.0f * 1024.0f * 1024.0f));
            cp_interval = new_cp;
        }
    }

    // Allocate receiver traces on host
    size_t trace_count = (size_t)rec.num_receivers * grid.nt;
    std::vector<float> h_traces(trace_count, 0.0f);

    // Image and illumination accumulators
    std::vector<float> h_img(N, 0.0f);
    std::vector<float> h_illum(N, 0.0f);

    int num_checkpoints = (grid.nt + cp_interval - 1) / cp_interval + 1;

    printf("[RTM-CPU] Checkpoint interval=%d, checkpoints=%d (%.1f MB total)\n",
           cp_interval, num_checkpoints,
           (float)num_checkpoints * 2 * bytes / (1024.0f * 1024.0f));

    // ── PHASE 1: FORWARD PROPAGATION with checkpoint saving ──
    printf("[RTM-CPU] Forward propagation (%d steps)...\n", grid.nt);

    std::vector<Checkpoint_CPU> checkpoints(num_checkpoints);

    // Zero state
    std::fill(state.u_prev.begin(), state.u_prev.end(), 0.0f);
    std::fill(state.u_curr.begin(), state.u_curr.end(), 0.0f);
    std::fill(state.u_next.begin(), state.u_next.end(), 0.0f);

    // Save initial checkpoint (t=0)
    checkpoints[0].h_u_prev = state.u_prev;
    checkpoints[0].h_u_curr = state.u_curr;
    checkpoints[0].time_step = 0;

    for (int t = 0; t < grid.nt; t++) {
        forward_step_cpu(state, grid, src, t, sponge_width,
                         h_traces.data(), rec.rx, rec.ry, rec.rz,
                         rec.num_receivers);

        if ((t + 1) % cp_interval == 0 || t == grid.nt - 1) {
            int cp_idx = (t + 1) / cp_interval;
            if (t == grid.nt - 1 && (t + 1) % cp_interval != 0)
                cp_idx = num_checkpoints - 1;

            checkpoints[cp_idx].h_u_prev = state.u_prev;
            checkpoints[cp_idx].h_u_curr = state.u_curr;
            checkpoints[cp_idx].time_step = t + 1;
        }
    }

    printf("[RTM-CPU] Forward pass complete.\n");

    // Copy traces to receiver output buffer
    memcpy(rec.traces, h_traces.data(), trace_count * sizeof(float));

    float max_trace = 0.0f;
    for (size_t i = 0; i < trace_count; i++) {
        float v = rec.traces[i]; if (v < 0) v = -v;
        if (v > max_trace) max_trace = v;
    }
    printf("[RTM-CPU] Max receiver trace amplitude: %.6e\n", max_trace);

    // ── PHASE 1b: DIRECT-ARRIVAL SUBTRACTION ──
    if (h_vel_background) {
        printf("[RTM-CPU] Running direct-arrival simulation (homogeneous model)...\n");

        // Save actual velocity, load background
        std::vector<float> vel_actual = state.vel;
        memcpy(state.vel.data(), h_vel_background, bytes);

        std::vector<float> h_traces_direct(trace_count, 0.0f);

        std::fill(state.u_prev.begin(), state.u_prev.end(), 0.0f);
        std::fill(state.u_curr.begin(), state.u_curr.end(), 0.0f);
        std::fill(state.u_next.begin(), state.u_next.end(), 0.0f);

        for (int t = 0; t < grid.nt; t++) {
            forward_step_cpu(state, grid, src, t, sponge_width,
                             h_traces_direct.data(), rec.rx, rec.ry, rec.rz,
                             rec.num_receivers);
        }

        // Subtract direct arrivals
        float max_diff = 0.0f;
        for (size_t i = 0; i < trace_count; i++) {
            h_traces[i] -= h_traces_direct[i];
            float v = h_traces[i]; if (v < 0) v = -v;
            if (v > max_diff) max_diff = v;
        }

        printf("[RTM-CPU] Direct-arrival subtracted. Reflection-only max=%.6e\n", max_diff);

        // Restore actual velocity
        state.vel = vel_actual;
    }

    // ── PHASE 2: BACKWARD PROPAGATION + IMAGING ──
    printf("[RTM-CPU] Backward propagation + imaging...\n");

    // Receiver state for backward propagation
    HostState rcv_state;
    rcv_state.u_prev.assign(N, 0.0f);
    rcv_state.u_curr.assign(N, 0.0f);
    rcv_state.u_next.assign(N, 0.0f);
    rcv_state.vel = state.vel;

    // Snapshot memory budget: cap at 2 GB
    int max_snap_slots = cp_interval;
    {
        size_t snap_budget = (size_t)2 * 1024 * 1024 * 1024;
        max_snap_slots = std::max(1, std::min(cp_interval, (int)(snap_budget / bytes)));

        if (max_snap_slots < cp_interval) {
            int num_chunks_est = (cp_interval + max_snap_slots - 1) / max_snap_slots;
            printf("[RTM-CPU] Sub-segment mode: %d snapshot slots, "
                   "~%d chunks/segment (%.1f MB snapshots)\n",
                   max_snap_slots, num_chunks_est,
                   (float)max_snap_slots * bytes / (1024.0f * 1024.0f));
        } else {
            printf("[RTM-CPU] Snapshot storage: %.1f MB per segment\n",
                   (float)cp_interval * bytes / (1024.0f * 1024.0f));
        }
    }
    std::vector<std::vector<float>> snapshots(max_snap_slots);

    // Process segments in reverse order
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
            state.u_prev = checkpoints[seg].h_u_prev;
            state.u_curr = checkpoints[seg].h_u_curr;

            // Fast-forward from segment start to chunk start
            for (int i = 0; i < c_start; i++) {
                int t = t_start + i;
                int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
                state.u_curr[idx] += src.wavelet[t];

                launch_kernel_cpu_avx(state.u_prev.data(), state.u_curr.data(),
                                      state.u_next.data(), state.vel.data(),
                                      grid.nx, grid.ny, grid.nz);
                apply_sponge_boundary_cpu(state.u_next.data(), grid.nx, grid.ny, grid.nz,
                                          sponge_width, 0.015f);
                std::swap(state.u_prev, state.u_curr);
                std::swap(state.u_curr, state.u_next);
            }

            // Save snapshots for this chunk
            for (int i = c_start; i < c_end; i++) {
                int t = t_start + i;
                int snap_idx = i - c_start;

                int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
                state.u_curr[idx] += src.wavelet[t];

                // Save snapshot (just a vector copy — all host memory)
                snapshots[snap_idx] = state.u_curr;

                launch_kernel_cpu_avx(state.u_prev.data(), state.u_curr.data(),
                                      state.u_next.data(), state.vel.data(),
                                      grid.nx, grid.ny, grid.nz);
                apply_sponge_boundary_cpu(state.u_next.data(), grid.nx, grid.ny, grid.nz,
                                          sponge_width, 0.015f);
                std::swap(state.u_prev, state.u_curr);
                std::swap(state.u_curr, state.u_next);
            }

            // Walk backward through this chunk
            for (int t = t_start + c_end - 1; t >= t_start + c_start; t--) {
                int snap_idx = t - (t_start + c_start);

                inject_receivers_cpu(rcv_state.u_curr.data(), h_traces.data(),
                                     rec.rx, rec.ry, rec.rz,
                                     rec.num_receivers, grid.nx, grid.ny,
                                     t, grid.nt);

                accumulate_source_illumination_cpu(snapshots[snap_idx].data(),
                                                    h_illum.data(), N);
                apply_imaging_condition_cpu(snapshots[snap_idx].data(),
                                            rcv_state.u_curr.data(),
                                            h_img.data(), N);

                launch_kernel_cpu_avx(rcv_state.u_prev.data(), rcv_state.u_curr.data(),
                                      rcv_state.u_next.data(), rcv_state.vel.data(),
                                      grid.nx, grid.ny, grid.nz);
                apply_sponge_boundary_cpu(rcv_state.u_next.data(), grid.nx, grid.ny, grid.nz,
                                          sponge_width, 0.015f);
                std::swap(rcv_state.u_prev, rcv_state.u_curr);
                std::swap(rcv_state.u_curr, rcv_state.u_next);
            }
        }

        // Free checkpoint after all chunks processed
        std::vector<float>().swap(checkpoints[seg].h_u_prev);
        std::vector<float>().swap(checkpoints[seg].h_u_curr);
    }

    printf("[RTM-CPU] Imaging complete.\n");

    // Copy raw image to output
    memcpy(h_image, h_img.data(), bytes);

    if (h_illum_out) {
        memcpy(h_illum_out, h_illum.data(), bytes);
    }

    if (!raw_output) {
        // Illumination normalization + source muting
        float max_illum = 0.0f;
        for (size_t i = 0; i < N; i++) {
            if (h_illum[i] > max_illum) max_illum = h_illum[i];
        }
        float epsilon = 0.01f * max_illum;
        if (epsilon < 1e-12f) epsilon = 1e-12f;

        int mute_start = src.sz;
        int mute_end   = src.sz + 15;
        for (int z = 0; z < grid.nz; z++) {
            float mute = 1.0f;
            if (z <= mute_start) mute = 0.0f;
            else if (z < mute_end) mute = (float)(z - mute_start) / (mute_end - mute_start);

            for (int y = 0; y < grid.ny; y++) {
                for (int x = 0; x < grid.nx; x++) {
                    size_t idx = z * (size_t)grid.nx * grid.ny + y * grid.nx + x;
                    h_image[idx] = (h_img[idx] / (h_illum[idx] + epsilon)) * mute;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  richter_rtm_multishot_cpu — Multi-shot stacking
// ═══════════════════════════════════════════════════════════════════

void richter_rtm_multishot_cpu(const Grid& grid,
                               const Source* sources, int num_shots,
                               const ReceiverSet& rec,
                               HostState& state, float* h_image,
                               int checkpoint_interval,
                               const float* h_vel_background)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);

    std::vector<float> stacked_image(N, 0.0f);
    std::vector<float> stacked_illum(N, 0.0f);
    std::vector<float> shot_image(N);
    std::vector<float> shot_illum(N);

    printf("[RTM-CPU Multi-Shot] Stacking %d shots\n", num_shots);

    for (int s = 0; s < num_shots; s++) {
        printf("\n[RTM-CPU Multi-Shot] === Shot %d/%d  src=(%d,%d,%d) ===\n",
               s + 1, num_shots, sources[s].sx, sources[s].sy, sources[s].sz);

        memset(shot_image.data(), 0, bytes);
        memset(shot_illum.data(), 0, bytes);

        richter_rtm_cpu(grid, sources[s], rec, state, shot_image.data(),
                        checkpoint_interval, h_vel_background,
                        /*raw_output=*/true, /*h_illum_out=*/shot_illum.data());

        for (size_t i = 0; i < N; i++) {
            stacked_image[i] += shot_image[i];
            stacked_illum[i] += shot_illum[i];
        }
    }

    printf("\n[RTM-CPU Multi-Shot] All shots complete. Normalizing stacked image...\n");

    float max_illum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        if (stacked_illum[i] > max_illum) max_illum = stacked_illum[i];
    }
    float epsilon = 0.01f * max_illum;
    if (epsilon < 1e-12f) epsilon = 1e-12f;

    int mute_src_z = sources[0].sz;
    for (int s = 1; s < num_shots; s++) {
        if (sources[s].sz < mute_src_z) mute_src_z = sources[s].sz;
    }
    int mute_start = mute_src_z;
    int mute_end   = mute_src_z + 15;

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

    printf("[RTM-CPU Multi-Shot] Stacking complete (%d shots).\n", num_shots);
}
