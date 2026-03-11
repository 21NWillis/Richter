#pragma once
#include "richter/rtm_cpu.h"
#include <vector>

// ─── CPU FWI Configuration (mirrors GPU FWIConfig) ───────────────

struct FWIConfigCPU {
    int max_iterations;
    float initial_step_size;
    float step_size_reduction;
    int max_line_search_steps;
    float misfit_tolerance;
    float gradient_smooth_sigma;
    float v_min, v_max;
    int water_depth;
    int checkpoint_interval;

    // Multi-scale frequency schedule
    int num_frequency_stages;
    const float* frequency_stages;
    const int* iterations_per_stage;

    // ML gradient filter hook (operates on host memory)
    using GradientFilter = void(*)(float* gradient, const Grid& grid, void* user_data);
    GradientFilter gradient_filter;
    void* gradient_filter_data;
};

struct FWIResultCPU {
    int iterations_completed;
    std::vector<float> misfit_history;
    std::vector<float> step_size_history;
};

// ─── Public API ──────────────────────────────────────────────────

/// Run Full Waveform Inversion (CPU-only, AVX2+OpenMP).
void richter_fwi_cpu(const Grid& grid,
                     const Source* sources, int num_shots,
                     const ReceiverSet& rec,
                     const float* const* h_obs_traces,
                     const float* h_vel_initial,
                     float* h_vel_output,
                     const FWIConfigCPU& config,
                     FWIResultCPU* result = nullptr);

// ─── CPU FWI Helper Operations ───────────────────────────────────

/// Compute residual and L2 misfit on CPU
float compute_residual_and_misfit_cpu(const float* syn, const float* obs,
                                      float* residual, int num_receivers, int nt);

/// Forward-only misfit (no residual output)
float compute_misfit_only_cpu(const float* syn, const float* obs,
                               int num_receivers, int nt);

/// Convert physical velocity (m/s) → stored coefficient (v²dt²/dx²)
void velocity_to_coefficient_cpu(const float* vel_phys, float* vel_coeff,
                                  float dt, float dx, size_t n);

/// Convert stored coefficient → physical velocity (m/s)
void coefficient_to_velocity_cpu(const float* vel_coeff, float* vel_phys,
                                  float dt, float dx, size_t n);

/// Update velocity: coeff[i] -= step * gradient[i], clamped
void apply_velocity_update_cpu(float* vel_coeff, const float* gradient,
                                float step_size, float dt, float dx,
                                float v_min, float v_max, size_t n);

/// 3D Gaussian smoothing (separable, in-place)
void smooth_gradient_3d_cpu(float* gradient, float* temp,
                             int nx, int ny, int nz, float sigma);

/// Zero gradient above water depth
void apply_water_mask_cpu(float* gradient, int nx, int ny, int nz, int water_depth);

/// Forward-only propagation for a single shot (CPU, for line search)
void richter_forward_only_cpu(const Grid& grid, const Source& src,
                               HostState& state, float* syn_traces,
                               const int* rx, const int* ry, const int* rz,
                               int num_receivers);

/// Gradient computation for a single shot (CPU).
/// Forward with checkpointing, backward with imaging.
/// Accumulates into gradient and illum (not zeroed internally).
void richter_gradient_cpu(const Grid& grid, const Source& src,
                           HostState& state,
                           const float* traces,
                           float* syn_traces,
                           const int* rx, const int* ry, const int* rz,
                           int num_receivers,
                           float* gradient, float* illum,
                           int checkpoint_interval);
