#pragma once
#include "richter/types.h"   // Grid, Source, ReceiverSet (no CUDA dependency)
#include <vector>
#include <cstddef>

// ─── Host Buffers (CPU-only, no CUDA dependency) ──────────────────
struct HostState {
    std::vector<float> u_prev;     // u(t-1)
    std::vector<float> u_curr;     // u(t)
    std::vector<float> u_next;     // u(t+1)
    std::vector<float> vel;        // velocity coefficient field
};

// ─── CPU Model API ────────────────────────────────────────────────

/// Allocate host memory and initialize the simulation state.
void richter_init_cpu(const Grid& grid, HostState& state);

/// Run forward propagation for nt time steps on CPU (AVX2+OpenMP).
void richter_forward_cpu(const Grid& grid, const Source& src,
                         HostState& state);

/// Copy the current pressure field to an output buffer.
void richter_snapshot_cpu(const Grid& grid, const HostState& state,
                          float* h_output);

/// Free all host state memory.
void richter_cleanup_cpu(HostState& state);

// ─── CPU RTM API ──────────────────────────────────────────────────

/// Run full Reverse Time Migration for a single shot (CPU-only).
void richter_rtm_cpu(const Grid& grid, const Source& src,
                     const ReceiverSet& rec,
                     HostState& state, float* h_image,
                     int checkpoint_interval = 10,
                     const float* h_vel_background = nullptr,
                     bool raw_output = false,
                     float* h_illum_out = nullptr);

/// Run RTM for multiple shots and stack the results (CPU-only).
void richter_rtm_multishot_cpu(const Grid& grid,
                               const Source* sources, int num_shots,
                               const ReceiverSet& rec,
                               HostState& state, float* h_image,
                               int checkpoint_interval = 50,
                               const float* h_vel_background = nullptr);

// ─── CPU Helper Operations ────────────────────────────────────────

void apply_sponge_boundary_cpu(float* u, int nx, int ny, int nz,
                               int sponge_width, float damping_factor);

void record_receivers_cpu(const float* u, float* traces,
                          const int* rx, const int* ry, const int* rz,
                          int num_receivers, int nx, int ny,
                          int t, int nt);

void inject_receivers_cpu(float* u, const float* traces,
                          const int* rx, const int* ry, const int* rz,
                          int num_receivers, int nx, int ny,
                          int t, int nt);

void apply_imaging_condition_cpu(const float* src_field,
                                 const float* rcv_field,
                                 float* image, size_t n);

void accumulate_source_illumination_cpu(const float* src_field,
                                        float* illum, size_t n);
