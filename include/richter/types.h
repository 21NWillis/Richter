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

// ─── Receiver Configuration ────────────────────────────────────────
struct ReceiverSet {
    int num_receivers;
    int* rx;           // x-positions (host array, length = num_receivers)
    int* ry;           // y-positions
    int* rz;           // z-positions
    float* traces;     // recorded data [num_receivers x nt] row-major (host)
};
