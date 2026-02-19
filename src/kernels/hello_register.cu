// Richter: Register Rotation (Warp Shuffle Optimized)

#include "richter/kernels.h"
#include "richter/config.h"
#include <cstdio>

__constant__ float d_coeff_reg[STENCIL_RADIUS + 1];
static bool coeffs_loaded_reg = false;

static void ensure_coefficients_reg() {
    if (!coeffs_loaded_reg) {
        cudaMemcpyToSymbol(d_coeff_reg, FD_COEFF, sizeof(FD_COEFF));
        coeffs_loaded_reg = true;
    }
}

#define FULL_MASK 0xFFFFFFFF

__global__ void kernel_register(const float* __restrict__ u_prev,
                                const float* __restrict__ u_curr,
                                float*       __restrict__ u_next,
                                const float* __restrict__ vel,
                                int nx, int ny, int nz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    bool active = (x >= STENCIL_RADIUS && x < nx - STENCIL_RADIUS && 
                   y >= STENCIL_RADIUS && y < ny - STENCIL_RADIUS);

    const int stride_y = nx;
    const int stride_z = nx * ny;
    const int xy_idx   = y * stride_y + x;

    float c0 = d_coeff_reg[0];
    float c1 = d_coeff_reg[1];
    float c2 = d_coeff_reg[2];
    float c3 = d_coeff_reg[3];
    float c4 = d_coeff_reg[4];

    // Z-axis stencil window (registers): m=minus, c=center, p=plus
    float z_m4 = __ldg(&u_curr[xy_idx + 0 * stride_z]);
    float z_m3 = __ldg(&u_curr[xy_idx + 1 * stride_z]);
    float z_m2 = __ldg(&u_curr[xy_idx + 2 * stride_z]);
    float z_m1 = __ldg(&u_curr[xy_idx + 3 * stride_z]);
    float z_c  = __ldg(&u_curr[xy_idx + 4 * stride_z]);
    float z_p1 = __ldg(&u_curr[xy_idx + 5 * stride_z]);
    float z_p2 = __ldg(&u_curr[xy_idx + 6 * stride_z]);
    float z_p3 = __ldg(&u_curr[xy_idx + 7 * stride_z]);

    #pragma unroll 4
    for (int z = STENCIL_RADIUS; z < nz - STENCIL_RADIUS; z++) {
        float z_p4 = __ldg(&u_curr[xy_idx + (z + STENCIL_RADIUS) * stride_z]);
        int idx = z * stride_z + xy_idx;

        // Z-axis Laplacian
        float lap = 3.0f * c0 * z_c;
        lap += c1 * (z_m1 + z_p1);
        lap += c2 * (z_m2 + z_p2);
        lap += c3 * (z_m3 + z_p3);
        lap += c4 * (z_m4 + z_p4);

        // X-axis Laplacian (Warp Shuffles)
        float val = z_c;
        
        float r1 = __shfl_down_sync(FULL_MASK, val, 1);
        if (threadIdx.x >= BLOCK_X - 1) r1 = __ldg(&u_curr[idx + 1]); 
        float r2 = __shfl_down_sync(FULL_MASK, val, 2);
        if (threadIdx.x >= BLOCK_X - 2) r2 = __ldg(&u_curr[idx + 2]);
        float r3 = __shfl_down_sync(FULL_MASK, val, 3);
        if (threadIdx.x >= BLOCK_X - 3) r3 = __ldg(&u_curr[idx + 3]);
        float r4 = __shfl_down_sync(FULL_MASK, val, 4);
        if (threadIdx.x >= BLOCK_X - 4) r4 = __ldg(&u_curr[idx + 4]);

        float l1 = __shfl_up_sync(FULL_MASK, val, 1);
        if (threadIdx.x < 1) l1 = __ldg(&u_curr[idx - 1]);
        float l2 = __shfl_up_sync(FULL_MASK, val, 2);
        if (threadIdx.x < 2) l2 = __ldg(&u_curr[idx - 2]);
        float l3 = __shfl_up_sync(FULL_MASK, val, 3);
        if (threadIdx.x < 3) l3 = __ldg(&u_curr[idx - 3]);
        float l4 = __shfl_up_sync(FULL_MASK, val, 4);
        if (threadIdx.x < 4) l4 = __ldg(&u_curr[idx - 4]);

        lap += c1 * (r1 + l1);
        lap += c2 * (r2 + l2);
        lap += c3 * (r3 + l3);
        lap += c4 * (r4 + l4);

        // Y-axis Laplacian (Global Reads)
        lap += c1 * (__ldg(&u_curr[idx + stride_y])   + __ldg(&u_curr[idx - stride_y]));
        lap += c2 * (__ldg(&u_curr[idx + 2*stride_y]) + __ldg(&u_curr[idx - 2*stride_y]));
        lap += c3 * (__ldg(&u_curr[idx + 3*stride_y]) + __ldg(&u_curr[idx - 3*stride_y]));
        lap += c4 * (__ldg(&u_curr[idx + 4*stride_y]) + __ldg(&u_curr[idx - 4*stride_y]));

        if (active) {
            float v_sq = __ldg(&vel[idx]);
            float u_old = __ldg(&u_prev[idx]);
            u_next[idx] = 2.0f * z_c - u_old + v_sq * lap;
        }

        // Rotate
        z_m4 = z_m3; z_m3 = z_m2; z_m2 = z_m1; z_m1 = z_c;
        z_c  = z_p1; z_p1 = z_p2; z_p2 = z_p3; z_p3 = z_p4;
    }
}

void launch_kernel_register(const float* u_prev, const float* u_curr,
                            float* u_next, const float* vel,
                            int nx, int ny, int nz)
{
    ensure_coefficients_reg();
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);
    kernel_register<<<grid, block>>>(u_prev, u_curr, u_next, vel, nx, ny, nz);
}
