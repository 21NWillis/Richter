#pragma once
#include "config.h"
#include "model.h"
#include "cuda_buffer.h"
#include <cstddef>

// ─── Receiver Configuration ────────────────────────────────────────
struct ReceiverSet {
    int num_receivers;
    int* rx;           // x-positions (host array, length = num_receivers)
    int* ry;           // y-positions
    int* rz;           // z-positions
    float* traces;     // recorded data [num_receivers × nt] row-major (host)
};

// ─── Receiver Kernel Wrappers ──────────────────────────────────────

/// Gather pressure at each receiver location and store in trace buffer.
/// @param d_u        current pressure field (device)
/// @param d_traces   trace buffer [num_receivers × nt] (device)
/// @param d_rx/ry/rz receiver positions (device arrays, length = num_receivers)
/// @param t          current time step index
/// @param nt         total number of time steps
void launch_record_receivers(const float* d_u, float* d_traces,
                             const int* d_rx, const int* d_ry, const int* d_rz,
                             int num_receivers, int nx, int ny,
                             int t, int nt);

/// Inject recorded trace values back into pressure field (adjoint).
/// @param d_u        current pressure field (device, modified in-place)
/// @param d_traces   trace buffer [num_receivers × nt] (device)
/// @param d_rx/ry/rz receiver positions (device arrays)
/// @param t          current time step index (in forward time)
/// @param nt         total number of time steps
void launch_inject_receivers(float* d_u, const float* d_traces,
                             const int* d_rx, const int* d_ry, const int* d_rz,
                             int num_receivers, int nx, int ny,
                             int t, int nt);

// ─── Imaging Condition ─────────────────────────────────────────────

/// Accumulate cross-correlation: image[i] += src[i] * rcv[i]
void apply_imaging_condition(const float* d_src_field,
                             const float* d_rcv_field,
                             float* d_image, size_t n);

/// Accumulate source illumination: illum[i] += src[i]²
void accumulate_source_illumination(const float* d_src_field,
                                     float* d_illum, size_t n);

/// Normalize image by source illumination: image[i] /= (illum[i] + epsilon)
void normalize_by_illumination(float* d_image, const float* d_illum,
                                size_t n, float epsilon = 1e-10f);

// ─── RTM Orchestrator ──────────────────────────────────────────────

/// Run full Reverse Time Migration.
/// @param grid       simulation domain
/// @param src        source configuration (with pre-computed wavelet)
/// @param rec        receivers (positions filled in; traces will be recorded)
/// @param state      device state (will be allocated/zeroed internally)
/// @param h_image    output image (host buffer, size = grid.total_points())
/// @param kernel     which stencil kernel to use
/// @param checkpoint_interval save source wavefield every N steps (trades compute for memory)
void richter_rtm(const Grid& grid, const Source& src, const ReceiverSet& rec,
                 DeviceState& state, float* h_image, KernelType kernel,
                 int checkpoint_interval = 10,
                 const float* h_vel_background = nullptr);
