#pragma once
#include "config.h"
#include <cstddef>

// ─── Simulation Domain ─────────────────────────────────────────────
struct Grid {
    int nx, ny, nz;        // Number of grid points per axis
    float dx, dy, dz;      // Grid spacing (meters)
    float dt;              // Time step (seconds)
    int nt;                // Number of time steps

    size_t total_points() const { return (size_t)nx * ny * nz; }
    size_t bytes()        const { return total_points() * sizeof(float); }
};

// ─── Source Configuration ───────────────────────────────────────────
struct Source {
    int sx, sy, sz;        // Source grid position
    float peak_freq;       // Ricker wavelet peak frequency (Hz)
    float* wavelet;        // Pre-computed wavelet samples (length = nt)
};

// ─── Device Buffers ─────────────────────────────────────────────────
#include "richter/cuda_buffer.h"

// ─── Device Buffers ─────────────────────────────────────────────────
struct DeviceState {
    CudaBuffer<float> d_u_prev;       // u(t-1)  — previous time step
    CudaBuffer<float> d_u_curr;       // u(t)    — current time step
    CudaBuffer<float> d_u_next;       // u(t+1)  — next time step (output)
    CudaBuffer<float> d_vel;          // velocity model (v^2 * dt^2 / dx^2 precomputed)
    CudaBuffer<float> d_wavelet;      // source wavelet on device
};

// ─── Public API ─────────────────────────────────────────────────────

/// Allocate device memory and initialize the simulation state.
void richter_init(const Grid& grid, const Source& src, DeviceState& state);

/// Run the full forward propagation for `nt` time steps.
/// `kernel` selects which implementation to dispatch.
void richter_forward(const Grid& grid, const Source& src,
                     DeviceState& state, KernelType kernel);

/// Copy the current pressure field back to host.
void richter_snapshot(const Grid& grid, const DeviceState& state,
                      float* h_output);

/// Free all device memory.
void richter_cleanup(DeviceState& state);
