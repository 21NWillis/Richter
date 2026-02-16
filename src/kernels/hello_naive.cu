// ─── Richter: Naive 3D Acoustic Wave Propagation Kernel ─────────────
// Implementation A: Direct global memory reads. One thread per grid point.
// This is the correctness baseline — no optimizations applied.

#include "richter/kernels.h"
#include "richter/config.h"
#include <cstdio>

// 8th-order FD coefficients in constant memory for fast broadcast
__constant__ float d_coeff[STENCIL_RADIUS + 1];

static bool coeffs_loaded = false;

static void ensure_coefficients() {
    if (!coeffs_loaded) {
        cudaMemcpyToSymbol(d_coeff, FD_COEFF, sizeof(FD_COEFF));
        coeffs_loaded = true;
    }
}

__global__ void kernel_naive(const float* __restrict__ u_prev,
                             const float* __restrict__ u_curr,
                             float*       __restrict__ u_next,
                             const float* __restrict__ vel,
                             int nx, int ny, int nz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;

    // Boundary check: skip the halo region where stencil can't reach
    if (x < STENCIL_RADIUS || x >= nx - STENCIL_RADIUS) return;
    if (y < STENCIL_RADIUS || y >= ny - STENCIL_RADIUS) return;
    if (z < STENCIL_RADIUS || z >= nz - STENCIL_RADIUS) return;

    const int stride_y = nx;
    const int stride_z = nx * ny;
    const int idx = z * stride_z + y * stride_y + x;

    // Compute the 3D Laplacian using 8th-order stencil
    float laplacian = 3.0f * d_coeff[0] * u_curr[idx]; // center weight × 3 axes

    #pragma unroll
    for (int r = 1; r <= STENCIL_RADIUS; r++) {
        laplacian += d_coeff[r] * (
            u_curr[idx + r]            + u_curr[idx - r]            +  // X axis
            u_curr[idx + r * stride_y] + u_curr[idx - r * stride_y] +  // Y axis
            u_curr[idx + r * stride_z] + u_curr[idx - r * stride_z]    // Z axis
        );
    }

    // Time-stepping: u(t+1) = 2*u(t) - u(t-1) + v^2 * dt^2 * laplacian
    u_next[idx] = 2.0f * u_curr[idx] - u_prev[idx] + vel[idx] * laplacian;
}

void launch_kernel_naive(const float* u_prev, const float* u_curr,
                         float* u_next, const float* vel,
                         int nx, int ny, int nz)
{
    ensure_coefficients();

    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              nz);

    kernel_naive<<<grid, block>>>(u_prev, u_curr, u_next, vel, nx, ny, nz);
}
