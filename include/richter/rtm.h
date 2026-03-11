#pragma once
#include "config.h"
#include "model.h"
#include "cuda_buffer.h"
#include <cstddef>

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

/// Run full Reverse Time Migration for a single shot.
/// @param grid       simulation domain
/// @param src        source configuration (with pre-computed wavelet)
/// @param rec        receivers (positions filled in; traces will be recorded)
/// @param state      device state (will be allocated/zeroed internally)
/// @param h_image    output image (host buffer, size = grid.total_points())
/// @param kernel     which stencil kernel to use
/// @param checkpoint_interval save source wavefield every N steps (trades compute for memory)
/// @param h_vel_background  optional homogeneous velocity for direct-arrival subtraction
/// @param raw_output if true, skip illumination normalization and source muting (for multi-shot stacking)
/// @param h_illum_out optional host buffer to receive raw illumination map (size = grid.total_points())
void richter_rtm(const Grid& grid, const Source& src, const ReceiverSet& rec,
                 DeviceState& state, float* h_image, KernelType kernel,
                 int checkpoint_interval = 10,
                 const float* h_vel_background = nullptr,
                 bool raw_output = false,
                 float* h_illum_out = nullptr);

/// Run RTM for multiple shots and stack the results.
/// Accumulates raw cross-correlation images and illumination across all shots,
/// then applies a single illumination normalization and source muting pass.
/// @param grid       simulation domain
/// @param sources    array of source configurations (length = num_shots)
/// @param num_shots  number of shots to stack
/// @param rec        receivers (same geometry reused per shot; traces overwritten each shot)
/// @param state      device state (reused across shots)
/// @param h_image    output stacked image (host buffer, size = grid.total_points())
/// @param kernel     which stencil kernel to use
/// @param checkpoint_interval save source wavefield every N steps
/// @param h_vel_background  optional homogeneous velocity for direct-arrival subtraction
void richter_rtm_multishot(const Grid& grid,
                           const Source* sources, int num_shots,
                           const ReceiverSet& rec,
                           DeviceState& state, float* h_image,
                           KernelType kernel,
                           int checkpoint_interval = 50,
                           const float* h_vel_background = nullptr);

// ─── FWI Gradient Helper ─────────────────────────────────────────

/// Compute the FWI gradient for a single shot (device-resident I/O).
/// Performs forward propagation with checkpointing, records synthetic traces,
/// computes residual (syn - obs) and L2 misfit, then backward propagation
/// injecting the residuals and accumulating the cross-correlation gradient.
///
/// Unlike richter_rtm, this function:
/// - Takes traces already on device (no upload/download)
/// - Keeps gradient on device (no D→H copy)
/// - Skips illumination normalization and source muting
/// - Records synthetic traces into d_syn_traces during forward pass
/// - Computes residual internally between forward and backward phases
///
/// @param grid             simulation domain
/// @param src              source configuration
/// @param state            device state (velocity must be set; wavefields will be zeroed)
/// @param d_obs_traces     observed traces on device (input, num_receivers × nt)
/// @param d_residual       residual buffer (device, num_receivers × nt, written internally)
/// @param d_syn_traces     synthetic traces recorded during forward pass (device output, num_receivers × nt)
/// @param d_rx/ry/rz       receiver positions on device
/// @param num_receivers    number of receivers
/// @param d_gradient       accumulated gradient (device, not zeroed internally)
/// @param d_illum          accumulated illumination (device, not zeroed internally)
/// @param kernel           stencil kernel to use
/// @param checkpoint_interval  checkpoint interval for memory management
/// @return                 L2 misfit for this shot (0.5 * ||syn - obs||^2)
float richter_rtm_gradient_gpu(const Grid& grid, const Source& src,
                                DeviceState& state,
                                CudaBuffer<float>& d_obs_traces,
                                CudaBuffer<float>& d_residual,
                                CudaBuffer<float>& d_syn_traces,
                                CudaBuffer<int>& d_rx,
                                CudaBuffer<int>& d_ry,
                                CudaBuffer<int>& d_rz,
                                int num_receivers,
                                CudaBuffer<float>& d_gradient,
                                CudaBuffer<float>& d_illum,
                                KernelType kernel,
                                int checkpoint_interval);

/// Forward-only propagation for a single shot (for line search misfit evaluation).
/// Records synthetic traces on device without computing the gradient.
void richter_forward_only(const Grid& grid, const Source& src,
                           DeviceState& state,
                           CudaBuffer<float>& d_syn_traces,
                           CudaBuffer<int>& d_rx,
                           CudaBuffer<int>& d_ry,
                           CudaBuffer<int>& d_rz,
                           int num_receivers,
                           KernelType kernel);
