#pragma once
#include "richter/model.h"
#include "richter/rtm.h"
#include "richter/cuda_buffer.h"
#include <vector>

// ─── FWI Configuration ───────────────────────────────────────────

struct FWIConfig {
    int max_iterations;              // max iterations per frequency stage
    float initial_step_size;         // starting step for line search
    float step_size_reduction;       // backtracking factor (e.g. 0.5)
    int max_line_search_steps;       // max attempts per line search
    float misfit_tolerance;          // stop if relative misfit change < this
    float gradient_smooth_sigma;     // Gaussian sigma in grid points (0 = none)
    float v_min, v_max;              // velocity bounds (m/s)
    int water_depth;                 // zero gradient above this depth (grid points)
    int checkpoint_interval;         // checkpoint interval for gradient computation

    // Multi-scale frequency schedule
    int num_frequency_stages;              // 0 or 1 = single-stage (use source's peak_freq)
    const float* frequency_stages;         // ascending peak frequencies (Hz)
    const int* iterations_per_stage;       // iterations at each frequency stage

    // ML gradient filter hook — called after smoothing, before velocity update.
    // The gradient is in device memory. Set to nullptr to skip.
    using GradientFilter = void(*)(float* d_gradient, const Grid& grid, void* user_data);
    GradientFilter gradient_filter;
    void* gradient_filter_data;
};

// ─── FWI Result ──────────────────────────────────────────────────

struct FWIResult {
    int iterations_completed;
    std::vector<float> misfit_history;
    std::vector<float> step_size_history;
};

// ─── Public API ──────────────────────────────────────────────────

/// Run Full Waveform Inversion.
///
/// Starting from an initial velocity model, iteratively updates the velocity
/// to minimize L2 misfit between synthetic and observed seismic data.
///
/// @param grid             simulation domain
/// @param sources          source configurations (length = num_shots).
///                         wavelet pointers are used only for frequency stage 0;
///                         for multi-scale, wavelets are regenerated per stage.
/// @param num_shots        number of shots
/// @param rec              receiver geometry (same for all shots; traces buffer unused)
/// @param h_obs_traces     observed traces per shot [num_shots][num_receivers × nt] (host)
/// @param h_vel_initial    initial velocity model in m/s (host, length = total_points)
/// @param h_vel_output     inverted velocity model in m/s (host output)
/// @param config           FWI parameters
/// @param kernel           stencil kernel to use
/// @param result           optional output for convergence history
void richter_fwi(const Grid& grid,
                 const Source* sources, int num_shots,
                 const ReceiverSet& rec,
                 const float* const* h_obs_traces,
                 const float* h_vel_initial,
                 float* h_vel_output,
                 const FWIConfig& config,
                 KernelType kernel,
                 FWIResult* result = nullptr);

// ─── FWI Kernel Wrappers (defined in fwi_kernels.cu) ─────────────

/// Compute residual and L2 misfit: residual[i] = syn[i] - obs[i]
/// Returns 0.5 * ||residual||^2
float compute_residual_and_misfit(const float* d_synthetic, const float* d_observed,
                                  float* d_residual, int num_receivers, int nt);

/// Forward-only misfit (allocates temp residual buffer internally)
float compute_misfit_only(const float* d_synthetic, const float* d_observed,
                          int num_receivers, int nt);

/// Convert physical velocity (m/s) → stored coefficient (v²dt²/dx²)
void velocity_to_coefficient(const float* d_vel_phys, float* d_vel_coeff,
                              float dt, float dx, size_t n);

/// Convert stored coefficient → physical velocity (m/s)
void coefficient_to_velocity(const float* d_vel_coeff, float* d_vel_phys,
                              float dt, float dx, size_t n);

/// Update velocity: coeff[i] -= step * gradient[i], clamped to [v_min, v_max]
void apply_velocity_update(float* d_vel_coeff, const float* d_gradient,
                            float step_size, float dt, float dx,
                            float v_min, float v_max, size_t n);

/// 3D Gaussian smoothing (separable, in-place using temp buffer)
void smooth_gradient_3d(float* d_gradient, float* d_temp,
                         int nx, int ny, int nz, float sigma);

/// FWI imaging condition: grad += Laplacian(source) * adjoint
/// Uses the proper FWI gradient formulation for better depth penetration.
void apply_fwi_imaging_condition(const float* d_src, const float* d_adj,
                                  float* d_gradient, float* d_illum,
                                  int nx, int ny, int nz);

/// Zero gradient above water depth
void apply_water_mask(float* d_gradient, int nx, int ny, int nz, int water_depth);
