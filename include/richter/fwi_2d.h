#pragma once
#include "richter/config.h"
#include "richter/cuda_buffer.h"
#include <vector>
#include <cstddef>

// ─── 2D Simulation Domain ─────────────────────────────────────────
struct Grid2D {
    int nx, nz;            // Number of grid points per axis
    float dx, dz;          // Grid spacing (meters)
    float dt;              // Time step (seconds)
    int nt;                // Number of time steps

    size_t total_points() const { return (size_t)nx * nz; }
    size_t bytes()        const { return total_points() * sizeof(float); }
};

// ─── 2D Source Configuration ──────────────────────────────────────
struct Source2D {
    int sx, sz;            // Source grid position
    float peak_freq;       // Ricker wavelet peak frequency (Hz)
    float* wavelet;        // Pre-computed wavelet samples (length = nt)
};

// ─── 2D Receiver Configuration ───────────────────────────────────
struct ReceiverSet2D {
    int num_receivers;
    int* rx;               // x-positions (host array, length = num_receivers)
    int* rz;               // z-positions
};

// ─── 2D Device State ─────────────────────────────────────────────
struct DeviceState2D {
    CudaBuffer<float> d_u_prev;       // u(t-1)
    CudaBuffer<float> d_u_curr;       // u(t)
    CudaBuffer<float> d_u_next;       // u(t+1)
    CudaBuffer<float> d_vel;          // velocity coefficient (v²dt²/dx²)
    CudaBuffer<float> d_wavelet;      // source wavelet on device
};

// ─── 2D FWI Configuration ────────────────────────────────────────

struct FWIConfig2D {
    int max_iterations;
    float initial_step_size;
    float step_size_reduction;
    int max_line_search_steps;
    float misfit_tolerance;
    float gradient_smooth_sigma;     // Gaussian sigma in grid points (0 = none)
    float v_min, v_max;              // velocity bounds (m/s)
    int water_depth;                 // zero gradient above this depth (grid points)
    int checkpoint_interval;

    // Direct arrival muting
    float mute_direct_v;             // reference velocity for muting (m/s), 0 = disabled
    float mute_taper_samples;        // cosine taper length in samples

    // Depth gradient scaling
    float depth_scale_power;         // gradient *= (z/active_depth)^power, 0 = disabled

    // Layer stripping: freeze shallow velocities after initial convergence
    int layer_strip_iter;            // iteration to activate shallow freeze (0 = disabled)
    int layer_strip_depth;           // freeze gradient above this depth (grid points)
    int layer_strip_taper;           // cosine taper width at the freeze boundary (grid points)

    // Multi-scale frequency schedule
    int num_frequency_stages;
    const float* frequency_stages;
    const int* iterations_per_stage;
};

// ─── 2D FWI Result ───────────────────────────────────────────────

struct FWIResult2D {
    int iterations_completed;
    std::vector<float> misfit_history;
    std::vector<float> step_size_history;
};

// ─── 2D FWI Public API ──────────────────────────────────────────

void richter_fwi_2d(const Grid2D& grid,
                    const Source2D* sources, int num_shots,
                    const ReceiverSet2D& rec,
                    const float* const* h_obs_traces,
                    const float* h_vel_initial,
                    float* h_vel_output,
                    const FWIConfig2D& config,
                    FWIResult2D* result = nullptr);

// ─── 2D Kernel Wrappers ─────────────────────────────────────────

// Wave propagation: 8th-order FD stencil in 2D (X + Z only)
void launch_stencil_2d(const float* u_prev, const float* u_curr,
                       float* u_next, const float* vel,
                       int nx, int nz);

// Sponge absorbing boundary in 2D
void apply_sponge_2d(float* u, int nx, int nz,
                     int sponge_width, float damping_factor);

// Source injection (single kernel call, no host round-trip)
void inject_source_2d(float* d_u, int sx, int sz, float amplitude,
                      int nx, int nz);

// Record receivers: gather pressure at receiver locations
void record_receivers_2d(const float* d_u, float* d_traces,
                         const int* d_rx, const int* d_rz,
                         int num_receivers, int nx,
                         int t, int nt);

// Inject receivers: scatter trace values into pressure field (adjoint)
void inject_receivers_2d(float* d_u, const float* d_traces,
                         const int* d_rx, const int* d_rz,
                         int num_receivers, int nx,
                         int t, int nt);

// FWI imaging condition: grad += Laplacian(source) * adjoint
void apply_fwi_imaging_2d(const float* d_src, const float* d_adj,
                          float* d_gradient, float* d_illum,
                          int nx, int nz);

// 2D Gaussian gradient smoothing (separable: X pass + Z pass)
void smooth_gradient_2d(float* d_gradient, float* d_temp,
                        int nx, int nz, float sigma);

// Water layer mask in 2D
void apply_water_mask_2d(float* d_gradient, int nx, int nz, int water_depth);

// Sponge zone gradient mask: zeros gradient in left/right/bottom sponge zones
void apply_sponge_gradient_mask_2d(float* d_gradient, int nx, int nz,
                                    int sponge_width);

// Mute direct arrivals in trace data (applied to residual before backward pass).
// Zeros out trace samples before the expected direct arrival time for each
// source-receiver pair, with a cosine taper transition.
// v_direct: reference velocity for direct arrival (m/s), typically v_water or v_top
// taper_samples: number of samples for cosine taper (e.g. 20-50)
void mute_direct_arrivals_2d(float* d_traces,
                              const int* d_rx, const int* d_rz,
                              int sx, int sz,
                              int num_receivers, int nt,
                              float dt, float dx,
                              float v_direct, float taper_samples);

// Per-row gradient normalization: divide each row by its L2 norm
// Equalizes gradient energy across depths so anomalies stand out laterally
void normalize_gradient_per_row_2d(float* d_gradient, int nx, int nz,
                                    int water_depth);

// Depth-weighted gradient scaling: gradient *= ((z - water_depth) / active_depth)^power
// Compensates for adjoint wavefield decay with depth.
// power=1.0 for linear scaling, power=2.0 for quadratic.
void apply_depth_scaling_2d(float* d_gradient, int nx, int nz,
                             int water_depth, float power);

// Layer-stripping shallow freeze: zero gradient above freeze_depth with cosine taper
void apply_shallow_freeze_2d(float* d_gradient, int nx, int nz,
                              int freeze_depth, int taper_width);

// ─── Reused from 3D (flat-array operations) ─────────────────────
// These work unchanged for 2D since they operate on flat arrays:
//   compute_residual_and_misfit()
//   compute_misfit_only()
//   velocity_to_coefficient()
//   coefficient_to_velocity()
//   apply_velocity_update()
//   normalize_by_illumination()
// Declared in richter/fwi.h — just include that header alongside this one.
