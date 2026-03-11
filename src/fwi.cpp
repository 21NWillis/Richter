// ─── Richter: FWI Orchestrator ──────────────────────────────────────
// Full Waveform Inversion: iteratively updates the velocity model to
// minimize L2 misfit between synthetic and observed seismic data.
//
// Uses the RTM gradient computation (cross-correlation imaging condition)
// wrapped in an optimization loop with backtracking line search and
// multi-scale frequency staging.

#include "richter/fwi.h"
#include "richter/kernels.h"
#include "richter/wavelet.h"
#include "richter/boundary.h"
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
        fprintf(stderr, "[Richter FWI] CUDA error at %s:%d — %s\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// Compute total misfit across all shots (forward-only, no gradient)
static float compute_total_misfit(const Grid& grid,
                                  const Source* sources, int num_shots,
                                  DeviceState& state,
                                  const std::vector<CudaBuffer<float>>& d_obs_traces,
                                  CudaBuffer<float>& d_syn_traces,
                                  CudaBuffer<int>& d_rx, CudaBuffer<int>& d_ry,
                                  CudaBuffer<int>& d_rz, int num_receivers,
                                  KernelType kernel)
{
    float total_misfit = 0.0f;
    for (int s = 0; s < num_shots; s++) {
        richter_forward_only(grid, sources[s], state,
                              d_syn_traces, d_rx, d_ry, d_rz,
                              num_receivers, kernel);

        total_misfit += compute_misfit_only(d_syn_traces.data(),
                                            d_obs_traces[s].data(),
                                            num_receivers, grid.nt);
    }
    return total_misfit;
}

void richter_fwi(const Grid& grid,
                 const Source* sources, int num_shots,
                 const ReceiverSet& rec,
                 const float* const* h_obs_traces,
                 const float* h_vel_initial,
                 float* h_vel_output,
                 const FWIConfig& config,
                 KernelType kernel,
                 FWIResult* result)
{
    size_t N = grid.total_points();
    size_t bytes = N * sizeof(float);
    size_t trace_size = (size_t)rec.num_receivers * grid.nt;

    printf("═══════════════════════════════════════\n");
    printf("  Richter — Full Waveform Inversion\n");
    printf("  Grid: %dx%dx%d  Shots: %d\n", grid.nx, grid.ny, grid.nz, num_shots);
    printf("  Velocity bounds: %.0f - %.0f m/s\n", config.v_min, config.v_max);
    printf("═══════════════════════════════════════\n\n");

    // Initialize device state
    DeviceState state;
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
    CudaBuffer<int> d_ry(rec.num_receivers);
    CudaBuffer<int> d_rz(rec.num_receivers);
    d_rx.copyFromHost(rec.rx, rec.num_receivers);
    d_ry.copyFromHost(rec.ry, rec.num_receivers);
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

    // Wavelet buffers for multi-scale
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

        printf("\n[FWI] ═══ Frequency Stage %d/%d  f=%.1f Hz  iters=%d ═══\n",
               stage + 1, num_stages, stage_freq, stage_iters);

        // Generate wavelet for this frequency stage
        generate_ricker_wavelet(h_wavelet.data(), grid.nt, grid.dt, stage_freq);

        // Upload wavelet to device
        state.d_wavelet.copyFromHost(h_wavelet.data(), grid.nt);

        // Create source configurations for this stage
        for (int s = 0; s < num_shots; s++) {
            stage_sources[s] = sources[s];
            stage_sources[s].peak_freq = stage_freq;
            stage_sources[s].wavelet = h_wavelet.data();
        }

        float prev_misfit = -1.0f;
        float step_size = config.initial_step_size;

        for (int iter = 0; iter < stage_iters; iter++) {
            total_iterations++;

            // Zero gradient and illumination accumulators
            d_gradient.zero();
            d_illum.zero();

            float total_misfit = 0.0f;

            // Accumulate gradient across all shots
            for (int s = 0; s < num_shots; s++) {
                // Forward propagation, residual computation, backward + gradient
                float shot_misfit = richter_rtm_gradient_gpu(
                    grid, stage_sources[s], state,
                    d_obs_traces[s], d_residual, d_syn_traces,
                    d_rx, d_ry, d_rz, rec.num_receivers,
                    d_gradient, d_illum,
                    kernel, config.checkpoint_interval);

                total_misfit += shot_misfit;
            }

            // Smooth gradient
            if (config.gradient_smooth_sigma > 0.0f) {
                smooth_gradient_3d(d_gradient.data(), d_smooth_temp.data(),
                                    grid.nx, grid.ny, grid.nz,
                                    config.gradient_smooth_sigma);
            }

            // Pseudo-Hessian preconditioning: normalize gradient by source illumination.
            // This equalizes gradient amplitude across depth — crucial for surface-only
            // acquisition where raw gradient is ~1000x stronger near sources.
            {
                std::vector<float> h_illum(N);
                CUDA_CHECK(cudaMemcpy(h_illum.data(), d_illum.data(), bytes,
                                      cudaMemcpyDeviceToHost));
                float max_illum = 0.0f;
                for (size_t i = 0; i < N; i++) {
                    if (h_illum[i] > max_illum) max_illum = h_illum[i];
                }
                // Small epsilon: 0.01% of max so deep regions get proper boost
                float epsilon = 1e-4f * max_illum;
                if (epsilon < 1e-20f) epsilon = 1e-20f;
                normalize_by_illumination(d_gradient.data(), d_illum.data(), N, epsilon);
            }

            // Apply water mask (after smoothing + normalization so it doesn't leak back)
            if (config.water_depth > 0) {
                apply_water_mask(d_gradient.data(), grid.nx, grid.ny, grid.nz,
                                 config.water_depth);
            }

            // ML gradient filter hook
            if (config.gradient_filter) {
                config.gradient_filter(d_gradient.data(), grid, config.gradient_filter_data);
            }

            // Negate and normalize gradient.
            // Cross-correlation gradient (src * adj) needs sign flip for descent:
            // coeff -= step * (-gradient) effectively adds gradient to reduce misfit.
            {
                std::vector<float> h_grad(N);
                CUDA_CHECK(cudaMemcpy(h_grad.data(), d_gradient.data(), bytes,
                                      cudaMemcpyDeviceToHost));
                float max_abs_grad = 0.0f;
                for (size_t i = 0; i < N; i++) {
                    float a = fabsf(h_grad[i]);
                    if (a > max_abs_grad) max_abs_grad = a;
                }

                // Diagnostic: gradient at a point below sources (mid-active region)
                int wd = config.water_depth > 0 ? config.water_depth : 0;
                int cx = grid.nx / 2, cy = grid.ny / 2;
                int cz = wd + (grid.nz - wd) / 3;  // 1/3 into active region
                size_t center_idx = (size_t)cz * grid.nx * grid.ny + cy * grid.nx + cx;
                float grad_center = (center_idx < N) ? h_grad[center_idx] : 0.0f;
                printf("       grad: max=%.3e  center(lens)=%.3e  ratio=%.4f\n",
                       max_abs_grad, fabsf(grad_center),
                       max_abs_grad > 0 ? fabsf(grad_center) / max_abs_grad : 0.0f);

                if (max_abs_grad > 0.0f) {
                    float scale = -1.0f / max_abs_grad;  // negate + normalize
                    for (size_t i = 0; i < N; i++) h_grad[i] *= scale;
                    CUDA_CHECK(cudaMemcpy(d_gradient.data(), h_grad.data(), bytes,
                                          cudaMemcpyHostToDevice));
                }
            }

            // Backtracking line search
            // Save current velocity for rollback
            CUDA_CHECK(cudaMemcpy(d_vel_backup.data(), state.d_vel.data(),
                                  bytes, cudaMemcpyDeviceToDevice));

            bool accepted = false;
            float accepted_misfit = total_misfit;
            float accepted_step = 0.0f;

            for (int ls = 0; ls < config.max_line_search_steps; ls++) {
                // Trial velocity update
                CUDA_CHECK(cudaMemcpy(state.d_vel.data(), d_vel_backup.data(),
                                      bytes, cudaMemcpyDeviceToDevice));
                apply_velocity_update(state.d_vel.data(), d_gradient.data(),
                                       step_size, grid.dt, grid.dx,
                                       config.v_min, config.v_max, N);

                // Compute trial misfit (forward-only for all shots)
                float trial_misfit = compute_total_misfit(
                    grid, stage_sources.data(), num_shots, state,
                    d_obs_traces, d_syn_traces,
                    d_rx, d_ry, d_rz, rec.num_receivers, kernel);

                if (trial_misfit < total_misfit) {
                    accepted = true;
                    accepted_misfit = trial_misfit;
                    accepted_step = step_size;
                    // Increase step for next iteration
                    step_size = std::min(step_size / config.step_size_reduction,
                                         config.initial_step_size);
                    break;
                }

                step_size *= config.step_size_reduction;
            }

            if (!accepted) {
                // Restore original velocity
                CUDA_CHECK(cudaMemcpy(state.d_vel.data(), d_vel_backup.data(),
                                      bytes, cudaMemcpyDeviceToDevice));
                printf("[FWI] Iter %d: line search failed. misfit=%.6e (no update)\n",
                       total_iterations, total_misfit);
            } else {
                printf("[FWI] Iter %d: misfit=%.6e -> %.6e  step=%.2e\n",
                       total_iterations, total_misfit, accepted_misfit, accepted_step);
            }

            // Track results
            float current_misfit = accepted ? accepted_misfit : total_misfit;
            if (result) {
                result->misfit_history.push_back(current_misfit);
                result->step_size_history.push_back(accepted ? accepted_step : 0.0f);
                result->iterations_completed = total_iterations;
            }

            // Convergence check
            if (prev_misfit > 0.0f && current_misfit > 0.0f) {
                float rel_change = fabsf(prev_misfit - current_misfit) / prev_misfit;
                if (rel_change < config.misfit_tolerance) {
                    printf("[FWI] Converged at iteration %d (relative change %.2e < %.2e)\n",
                           total_iterations, rel_change, config.misfit_tolerance);
                    break;
                }
            }
            prev_misfit = current_misfit;
        }
    }

    // Convert final velocity coefficient → physical velocity and download
    {
        CudaBuffer<float> d_vel_phys(N);
        coefficient_to_velocity(state.d_vel.data(), d_vel_phys.data(),
                                 grid.dt, grid.dx, N);
        CUDA_CHECK(cudaMemcpy(h_vel_output, d_vel_phys.data(), bytes,
                              cudaMemcpyDeviceToHost));
    }

    printf("\n[FWI] Complete. %d total iterations.\n", total_iterations);

    // Cleanup
    state.d_u_prev = CudaBuffer<float>();
    state.d_u_curr = CudaBuffer<float>();
    state.d_u_next = CudaBuffer<float>();
    state.d_vel    = CudaBuffer<float>();
    state.d_wavelet = CudaBuffer<float>();
}
