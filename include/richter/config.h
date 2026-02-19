#pragma once

// ─── Grid & Stencil Configuration ──────────────────────────────────
// Stencil radius: 4 for 8th-order finite differences
constexpr int STENCIL_RADIUS = 4;
constexpr int STENCIL_ORDER  = 2 * STENCIL_RADIUS;  // 8th order

// Default thread-block dimensions for XY tiling
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 16;

// ─── Physics Defaults ───────────────────────────────────────────────
constexpr float DEFAULT_VELOCITY = 2000.0f;  // m/s (water/soft sediment)
constexpr float DEFAULT_DX       = 10.0f;    // grid spacing (meters)
constexpr float DEFAULT_DT       = 0.001f;   // time step (seconds)

// ─── 8th-order FD Coefficients ──────────────────────────────────────
// Central coefficients for the 2nd derivative from Taylor expansion
// d^2u/dx^2 ≈ sum_i c[i] * u(x + i*dx) / dx^2
constexpr float FD_COEFF[STENCIL_RADIUS + 1] = {
    -205.0f / 72.0f,   // c0 (center)
     8.0f  / 5.0f,     // c1 (±1)
    -1.0f  / 5.0f,     // c2 (±2)
     8.0f  / 315.0f,   // c3 (±3)
    -1.0f  / 560.0f    // c4 (±4)
};

// ─── Kernel Selection ───────────────────────────────────────────────
enum class KernelType {
    NAIVE,           // Direct global memory reads
    SHARED_MEMORY,   // 2.5D XY tiling with shared memory
    REGISTER_ROT,    // Register rotation / sliding window on Z
    HYBRID,          // Shared memory XY + register rotation Z
};
