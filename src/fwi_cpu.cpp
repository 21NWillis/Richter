// ─── Richter: CPU-Only FWI Pipeline ─────────────────────────────────
// Full Waveform Inversion running entirely on host memory using the
// AVX2+FMA+OpenMP stencil kernel. No CUDA dependency.

#include "richter/fwi_cpu.h"
#include "richter/kernels.h"
#include "richter/wavelet.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═══════════════════════════════════════════════════════════════════
//  CPU FWI Helper Operations
// ═══════════════════════════════════════════════════════════════════

float compute_residual_and_misfit_cpu(const float* syn, const float* obs,
                                      float* residual, int num_receivers, int nt)
{
    int n = num_receivers * nt;
    float misfit = 0.0f;

    #pragma omp parallel for reduction(+:misfit) schedule(static)
    for (int i = 0; i < n; i++) {
        float r = syn[i] - obs[i];
        residual[i] = r;
        misfit += r * r;
    }

    return 0.5f * misfit;
}

float compute_misfit_only_cpu(const float* syn, const float* obs,
                               int num_receivers, int nt)
{
    int n = num_receivers * nt;
    float misfit = 0.0f;

    #pragma omp parallel for reduction(+:misfit) schedule(static)
    for (int i = 0; i < n; i++) {
        float r = syn[i] - obs[i];
        misfit += r * r;
    }

    return 0.5f * misfit;
}

void velocity_to_coefficient_cpu(const float* vel_phys, float* vel_coeff,
                                  float dt, float dx, size_t n)
{
    float dt2_dx2 = (dt * dt) / (dx * dx);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        vel_coeff[i] = vel_phys[i] * vel_phys[i] * dt2_dx2;
    }
}

void coefficient_to_velocity_cpu(const float* vel_coeff, float* vel_phys,
                                  float dt, float dx, size_t n)
{
    float dx_dt = dx / dt;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        vel_phys[i] = sqrtf(vel_coeff[i]) * dx_dt;
    }
}

void apply_velocity_update_cpu(float* vel_coeff, const float* gradient,
                                float step_size, float dt, float dx,
                                float v_min, float v_max, size_t n)
{
    float dt2_dx2 = (dt * dt) / (dx * dx);
    float c_min = v_min * v_min * dt2_dx2;
    float c_max = v_max * v_max * dt2_dx2;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        float c = vel_coeff[i] - step_size * gradient[i];
        vel_coeff[i] = std::min(std::max(c, c_min), c_max);
    }
}

void smooth_gradient_3d_cpu(float* gradient, float* temp,
                             int nx, int ny, int nz, float sigma)
{
    if (sigma <= 0.0f) return;

    int radius = (int)ceilf(3.0f * sigma);
    if (radius < 1) radius = 1;
    int ksize = 2 * radius + 1;

    // Build Gaussian weights
    std::vector<float> weights(ksize);
    float sum = 0.0f;
    for (int i = 0; i < ksize; i++) {
        float x = (float)(i - radius);
        weights[i] = expf(-0.5f * x * x / (sigma * sigma));
        sum += weights[i];
    }
    for (int i = 0; i < ksize; i++) weights[i] /= sum;

    // X pass: gradient -> temp
    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float s = 0.0f, wsum = 0.0f;
                for (int r = -radius; r <= radius; r++) {
                    int xx = x + r;
                    if (xx >= 0 && xx < nx) {
                        float w = weights[r + radius];
                        s += w * gradient[z * nx * ny + y * nx + xx];
                        wsum += w;
                    }
                }
                temp[z * nx * ny + y * nx + x] = s / wsum;
            }
        }
    }

    // Y pass: temp -> gradient
    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < nz; z++) {
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                float s = 0.0f, wsum = 0.0f;
                for (int r = -radius; r <= radius; r++) {
                    int yy = y + r;
                    if (yy >= 0 && yy < ny) {
                        float w = weights[r + radius];
                        s += w * temp[z * nx * ny + yy * nx + x];
                        wsum += w;
                    }
                }
                gradient[z * nx * ny + y * nx + x] = s / wsum;
            }
        }
    }

    // Z pass: gradient -> temp, then copy back
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int z = 0; z < nz; z++) {
                float s = 0.0f, wsum = 0.0f;
                for (int r = -radius; r <= radius; r++) {
                    int zz = z + r;
                    if (zz >= 0 && zz < nz) {
                        float w = weights[r + radius];
                        s += w * gradient[zz * nx * ny + y * nx + x];
                        wsum += w;
                    }
                }
                temp[z * nx * ny + y * nx + x] = s / wsum;
            }
        }
    }

    size_t bytes = (size_t)nx * ny * nz * sizeof(float);
    memcpy(gradient, temp, bytes);
}

void apply_water_mask_cpu(float* gradient, int nx, int ny, int nz, int water_depth)
{
    if (water_depth <= 0) return;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < water_depth; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                gradient[z * nx * ny + y * nx + x] = 0.0f;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Forward-Only Propagation (for line search)
// ═══════════════════════════════════════════════════════════════════

void richter_forward_only_cpu(const Grid& grid, const Source& src,
                               HostState& state, float* syn_traces,
                               const int* rx, const int* ry, const int* rz,
                               int num_receivers)
{
    int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;

    std::fill(state.u_prev.begin(), state.u_prev.end(), 0.0f);
    std::fill(state.u_curr.begin(), state.u_curr.end(), 0.0f);
    std::fill(state.u_next.begin(), state.u_next.end(), 0.0f);
    memset(syn_traces, 0, (size_t)num_receivers * grid.nt * sizeof(float));

    for (int t = 0; t < grid.nt; t++) {
        int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
        state.u_curr[idx] += src.wavelet[t];

        launch_kernel_cpu_avx(state.u_prev.data(), state.u_curr.data(),
                              state.u_next.data(), state.vel.data(),
                              grid.nx, grid.ny, grid.nz);
        apply_sponge_boundary_cpu(state.u_next.data(), grid.nx, grid.ny, grid.nz,
                                  sponge_width, 0.015f);
        std::swap(state.u_prev, state.u_curr);
        std::swap(state.u_curr, state.u_next);

        record_receivers_cpu(state.u_curr.data(), syn_traces,
                             rx, ry, rz, num_receivers,
                             grid.nx, grid.ny, t, grid.nt);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Gradient Computation (forward + checkpoint + backward)
// ═══════════════════════════════════════════════════════════════════

struct CheckpointFWI {
    std::vector<float> h_u_prev;
    std::vector<float> h_u_curr;
    int time_step;
};

void richter_gradient_cpu(const Grid& grid, const Source& src,
                           HostState& state,
                           const float* traces,
                           float* syn_traces,
                           const int* rx, const int* ry, const int* rz,
                           int num_receivers,
                           float* gradient, float* illum,
                           int checkpoint_interval)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);
    int sponge_width = (grid.nx < 64) ? grid.nx / 6 : 20;
    int cp_interval = std::max(checkpoint_interval, 1);

    // Auto-scale
    {
        size_t cp_budget = (size_t)6 * 1024 * 1024 * 1024;
        int max_cps = std::max(3, (int)(cp_budget / (2 * bytes)));
        int needed_cps = (grid.nt + cp_interval - 1) / cp_interval + 1;
        if (needed_cps > max_cps) {
            cp_interval = (grid.nt + max_cps - 2) / (max_cps - 1);
        }
    }

    int num_checkpoints = (grid.nt + cp_interval - 1) / cp_interval + 1;

    // Forward pass with checkpointing + recording
    std::vector<CheckpointFWI> checkpoints(num_checkpoints);

    std::fill(state.u_prev.begin(), state.u_prev.end(), 0.0f);
    std::fill(state.u_curr.begin(), state.u_curr.end(), 0.0f);
    std::fill(state.u_next.begin(), state.u_next.end(), 0.0f);
    memset(syn_traces, 0, (size_t)num_receivers * grid.nt * sizeof(float));

    checkpoints[0].h_u_prev = state.u_prev;
    checkpoints[0].h_u_curr = state.u_curr;
    checkpoints[0].time_step = 0;

    for (int t = 0; t < grid.nt; t++) {
        int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
        state.u_curr[idx] += src.wavelet[t];

        launch_kernel_cpu_avx(state.u_prev.data(), state.u_curr.data(),
                              state.u_next.data(), state.vel.data(),
                              grid.nx, grid.ny, grid.nz);
        apply_sponge_boundary_cpu(state.u_next.data(), grid.nx, grid.ny, grid.nz,
                                  sponge_width, 0.015f);
        std::swap(state.u_prev, state.u_curr);
        std::swap(state.u_curr, state.u_next);

        record_receivers_cpu(state.u_curr.data(), syn_traces,
                             rx, ry, rz, num_receivers,
                             grid.nx, grid.ny, t, grid.nt);

        if ((t + 1) % cp_interval == 0 || t == grid.nt - 1) {
            int cp_idx = (t + 1) / cp_interval;
            if (t == grid.nt - 1 && (t + 1) % cp_interval != 0)
                cp_idx = num_checkpoints - 1;
            checkpoints[cp_idx].h_u_prev = state.u_prev;
            checkpoints[cp_idx].h_u_curr = state.u_curr;
            checkpoints[cp_idx].time_step = t + 1;
        }
    }

    // Backward pass with imaging
    HostState adj_state;
    adj_state.u_prev.assign(N, 0.0f);
    adj_state.u_curr.assign(N, 0.0f);
    adj_state.u_next.assign(N, 0.0f);
    adj_state.vel = state.vel;

    // Snapshot slots
    size_t snap_budget = (size_t)2 * 1024 * 1024 * 1024;
    int max_snap_slots = std::max(1, std::min(cp_interval, (int)(snap_budget / bytes)));
    std::vector<std::vector<float>> snapshots(max_snap_slots);

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

            // Restore checkpoint
            state.u_prev = checkpoints[seg].h_u_prev;
            state.u_curr = checkpoints[seg].h_u_curr;

            // Fast-forward to chunk start
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

            // Save snapshots
            for (int i = c_start; i < c_end; i++) {
                int t = t_start + i;
                int snap_idx = i - c_start;

                int idx = src.sz * grid.nx * grid.ny + src.sy * grid.nx + src.sx;
                state.u_curr[idx] += src.wavelet[t];

                snapshots[snap_idx] = state.u_curr;

                launch_kernel_cpu_avx(state.u_prev.data(), state.u_curr.data(),
                                      state.u_next.data(), state.vel.data(),
                                      grid.nx, grid.ny, grid.nz);
                apply_sponge_boundary_cpu(state.u_next.data(), grid.nx, grid.ny, grid.nz,
                                          sponge_width, 0.015f);
                std::swap(state.u_prev, state.u_curr);
                std::swap(state.u_curr, state.u_next);
            }

            // Walk backward
            for (int t = t_start + c_end - 1; t >= t_start + c_start; t--) {
                int snap_idx = t - (t_start + c_start);

                inject_receivers_cpu(adj_state.u_curr.data(), traces,
                                     rx, ry, rz, num_receivers,
                                     grid.nx, grid.ny, t, grid.nt);

                accumulate_source_illumination_cpu(snapshots[snap_idx].data(),
                                                    illum, N);
                apply_imaging_condition_cpu(snapshots[snap_idx].data(),
                                            adj_state.u_curr.data(),
                                            gradient, N);

                launch_kernel_cpu_avx(adj_state.u_prev.data(), adj_state.u_curr.data(),
                                      adj_state.u_next.data(), adj_state.vel.data(),
                                      grid.nx, grid.ny, grid.nz);
                apply_sponge_boundary_cpu(adj_state.u_next.data(), grid.nx, grid.ny, grid.nz,
                                          sponge_width, 0.015f);
                std::swap(adj_state.u_prev, adj_state.u_curr);
                std::swap(adj_state.u_curr, adj_state.u_next);
            }
        }

        std::vector<float>().swap(checkpoints[seg].h_u_prev);
        std::vector<float>().swap(checkpoints[seg].h_u_curr);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  richter_fwi_cpu — Main FWI Orchestrator (CPU-only)
// ═══════════════════════════════════════════════════════════════════

static float compute_total_misfit_cpu(const Grid& grid,
                                      const Source* sources, int num_shots,
                                      HostState& state,
                                      const float* const* h_obs_traces,
                                      float* syn_buf,
                                      const int* rx, const int* ry, const int* rz,
                                      int num_receivers)
{
    float total = 0.0f;
    for (int s = 0; s < num_shots; s++) {
        richter_forward_only_cpu(grid, sources[s], state, syn_buf,
                                  rx, ry, rz, num_receivers);
        total += compute_misfit_only_cpu(syn_buf, h_obs_traces[s],
                                          num_receivers, grid.nt);
    }
    return total;
}

void richter_fwi_cpu(const Grid& grid,
                     const Source* sources, int num_shots,
                     const ReceiverSet& rec,
                     const float* const* h_obs_traces,
                     const float* h_vel_initial,
                     float* h_vel_output,
                     const FWIConfigCPU& config,
                     FWIResultCPU* result)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);
    size_t trace_size = (size_t)rec.num_receivers * grid.nt;

    printf("═══════════════════════════════════════\n");
    printf("  Richter — FWI (CPU AVX2+OMP)\n");
    printf("  Grid: %dx%dx%d  Shots: %d\n", grid.nx, grid.ny, grid.nz, num_shots);
    printf("═══════════════════════════════════════\n\n");

    // Init host state
    HostState state;
    richter_init_cpu(grid, state);

    // Convert initial velocity to coefficient
    velocity_to_coefficient_cpu(h_vel_initial, state.vel.data(),
                                 grid.dt, grid.dx, N);

    // Working buffers
    std::vector<float> gradient(N, 0.0f);
    std::vector<float> illum(N, 0.0f);
    std::vector<float> smooth_temp(N, 0.0f);
    std::vector<float> vel_backup(N, 0.0f);
    std::vector<float> syn_traces(trace_size, 0.0f);
    std::vector<float> residual(trace_size, 0.0f);

    if (result) {
        result->iterations_completed = 0;
        result->misfit_history.clear();
        result->step_size_history.clear();
    }

    int num_stages = config.num_frequency_stages;
    if (num_stages <= 0) num_stages = 1;

    std::vector<float> h_wavelet(grid.nt);
    std::vector<Source> stage_sources(num_shots);

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

        printf("\n[FWI-CPU] ═══ Frequency Stage %d/%d  f=%.1f Hz  iters=%d ═══\n",
               stage + 1, num_stages, stage_freq, stage_iters);

        generate_ricker_wavelet(h_wavelet.data(), grid.nt, grid.dt, stage_freq);

        for (int s = 0; s < num_shots; s++) {
            stage_sources[s] = sources[s];
            stage_sources[s].peak_freq = stage_freq;
            stage_sources[s].wavelet = h_wavelet.data();
        }

        float prev_misfit = -1.0f;
        float step_size = config.initial_step_size;

        for (int iter = 0; iter < stage_iters; iter++) {
            total_iterations++;

            std::fill(gradient.begin(), gradient.end(), 0.0f);
            std::fill(illum.begin(), illum.end(), 0.0f);

            float total_misfit = 0.0f;

            for (int s = 0; s < num_shots; s++) {
                richter_gradient_cpu(grid, stage_sources[s], state,
                                      residual.data(), syn_traces.data(),
                                      rec.rx, rec.ry, rec.rz, rec.num_receivers,
                                      gradient.data(), illum.data(),
                                      config.checkpoint_interval);

                float shot_misfit = compute_residual_and_misfit_cpu(
                    syn_traces.data(), h_obs_traces[s],
                    residual.data(), rec.num_receivers, grid.nt);

                total_misfit += shot_misfit;
            }

            // Water mask
            if (config.water_depth > 0) {
                apply_water_mask_cpu(gradient.data(), grid.nx, grid.ny, grid.nz,
                                     config.water_depth);
            }

            // Smooth
            if (config.gradient_smooth_sigma > 0.0f) {
                smooth_gradient_3d_cpu(gradient.data(), smooth_temp.data(),
                                        grid.nx, grid.ny, grid.nz,
                                        config.gradient_smooth_sigma);
            }

            // Illumination normalization
            {
                float max_illum = 0.0f;
                for (size_t i = 0; i < N; i++) {
                    if (illum[i] > max_illum) max_illum = illum[i];
                }
                float epsilon = 0.01f * max_illum;
                if (epsilon < 1e-12f) epsilon = 1e-12f;

                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < N; i++) {
                    gradient[i] /= (illum[i] + epsilon);
                }
            }

            // ML hook
            if (config.gradient_filter) {
                config.gradient_filter(gradient.data(), grid, config.gradient_filter_data);
            }

            // Negate and normalize gradient (same as GPU path)
            {
                float max_abs_grad = 0.0f;
                for (size_t i = 0; i < N; i++) {
                    float a = fabsf(gradient[i]);
                    if (a > max_abs_grad) max_abs_grad = a;
                }
                if (max_abs_grad > 0.0f) {
                    float scale = -1.0f / max_abs_grad;
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < N; i++) gradient[i] *= scale;
                }
            }

            // Backtracking line search
            memcpy(vel_backup.data(), state.vel.data(), bytes);

            bool accepted = false;
            float accepted_misfit = total_misfit;
            float accepted_step = 0.0f;

            for (int ls = 0; ls < config.max_line_search_steps; ls++) {
                memcpy(state.vel.data(), vel_backup.data(), bytes);
                apply_velocity_update_cpu(state.vel.data(), gradient.data(),
                                           step_size, grid.dt, grid.dx,
                                           config.v_min, config.v_max, N);

                float trial_misfit = compute_total_misfit_cpu(
                    grid, stage_sources.data(), num_shots, state,
                    h_obs_traces, syn_traces.data(),
                    rec.rx, rec.ry, rec.rz, rec.num_receivers);

                if (trial_misfit < total_misfit) {
                    accepted = true;
                    accepted_misfit = trial_misfit;
                    accepted_step = step_size;
                    step_size = std::min(step_size / config.step_size_reduction,
                                         config.initial_step_size * 4.0f);
                    break;
                }

                step_size *= config.step_size_reduction;
            }

            if (!accepted) {
                memcpy(state.vel.data(), vel_backup.data(), bytes);
                printf("[FWI-CPU] Iter %d: line search failed. misfit=%.6e\n",
                       total_iterations, total_misfit);
            } else {
                printf("[FWI-CPU] Iter %d: misfit=%.6e -> %.6e  step=%.2e\n",
                       total_iterations, total_misfit, accepted_misfit, accepted_step);
            }

            float current_misfit = accepted ? accepted_misfit : total_misfit;
            if (result) {
                result->misfit_history.push_back(current_misfit);
                result->step_size_history.push_back(accepted ? accepted_step : 0.0f);
                result->iterations_completed = total_iterations;
            }

            if (prev_misfit > 0.0f && current_misfit > 0.0f) {
                float rel_change = fabsf(prev_misfit - current_misfit) / prev_misfit;
                if (rel_change < config.misfit_tolerance) {
                    printf("[FWI-CPU] Converged at iteration %d (rel change %.2e < %.2e)\n",
                           total_iterations, rel_change, config.misfit_tolerance);
                    break;
                }
            }
            prev_misfit = current_misfit;
        }
    }

    // Convert final velocity to physical and output
    coefficient_to_velocity_cpu(state.vel.data(), h_vel_output,
                                 grid.dt, grid.dx, N);

    printf("\n[FWI-CPU] Complete. %d total iterations.\n", total_iterations);

    richter_cleanup_cpu(state);
}
