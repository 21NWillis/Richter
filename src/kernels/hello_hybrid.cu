// Richter: Hybrid Kernel (Shared Memory XY + Register Rotation Z)

#include "richter/kernels.h"
#include "richter/config.h"
#include <cstdio>

__constant__ float d_coeff_hybrid[STENCIL_RADIUS + 1];
static bool coeffs_loaded_hybrid = false;

static void ensure_coefficients_hybrid() {
    if (!coeffs_loaded_hybrid) {
        cudaMemcpyToSymbol(d_coeff_hybrid, FD_COEFF, sizeof(FD_COEFF));
        coeffs_loaded_hybrid = true;
    }
}

constexpr int HTILE_X = BLOCK_X + 2 * STENCIL_RADIUS;
constexpr int HTILE_Y = BLOCK_Y + 2 * STENCIL_RADIUS;

__global__ void kernel_hybrid(const float* __restrict__ u_prev,
                              const float* __restrict__ u_curr,
                              float*       __restrict__ u_next,
                              const float* __restrict__ vel,
                              int nx, int ny, int nz)
{
    __shared__ float tile[HTILE_Y][HTILE_X];

    const int local_x  = threadIdx.x;
    const int local_y  = threadIdx.y;
    const int global_x = blockIdx.x * BLOCK_X + local_x;
    const int global_y = blockIdx.y * BLOCK_Y + local_y;

    const int tile_x = local_x + STENCIL_RADIUS;
    const int tile_y = local_y + STENCIL_RADIUS;

    const int stride_y = nx;
    const int stride_z = nx * ny;

    const bool in_domain = (global_x >= STENCIL_RADIUS && global_x < nx - STENCIL_RADIUS &&
                            global_y >= STENCIL_RADIUS && global_y < ny - STENCIL_RADIUS);
    const bool in_grid   = (global_x < nx && global_y < ny);

    const int xy_idx = global_y * stride_y + global_x;

    // Prime Z pipeline
    float z_m4 = in_grid ? u_curr[xy_idx + 0 * stride_z] : 0.0f;
    float z_m3 = in_grid ? u_curr[xy_idx + 1 * stride_z] : 0.0f;
    float z_m2 = in_grid ? u_curr[xy_idx + 2 * stride_z] : 0.0f;
    float z_m1 = in_grid ? u_curr[xy_idx + 3 * stride_z] : 0.0f;
    float z_c  = in_grid ? u_curr[xy_idx + 4 * stride_z] : 0.0f;
    float z_p1 = in_grid ? u_curr[xy_idx + 5 * stride_z] : 0.0f;
    float z_p2 = in_grid ? u_curr[xy_idx + 6 * stride_z] : 0.0f;
    float z_p3 = in_grid ? u_curr[xy_idx + 7 * stride_z] : 0.0f;

    for (int z = STENCIL_RADIUS; z < nz - STENCIL_RADIUS; z++) {
        float z_p4 = in_grid ? u_curr[xy_idx + (z + STENCIL_RADIUS) * stride_z] : 0.0f;

        if (in_grid) tile[tile_y][tile_x] = z_c;

        // Load Halos
        if (local_x < STENCIL_RADIUS) {
            int hx = global_x - STENCIL_RADIUS;
            tile[tile_y][local_x] = (hx >= 0 && global_y < ny)
                ? u_curr[z * stride_z + global_y * stride_y + hx] : 0.0f;
        }
        if (local_x >= BLOCK_X - STENCIL_RADIUS) {
            int hx = global_x + STENCIL_RADIUS;
            tile[tile_y][tile_x + STENCIL_RADIUS] = (hx < nx && global_y < ny)
                ? u_curr[z * stride_z + global_y * stride_y + hx] : 0.0f;
        }
        if (local_y < STENCIL_RADIUS) {
            int hy = global_y - STENCIL_RADIUS;
            tile[local_y][tile_x] = (hy >= 0 && global_x < nx)
                ? u_curr[z * stride_z + hy * stride_y + global_x] : 0.0f;
        }
        if (local_y >= BLOCK_Y - STENCIL_RADIUS) {
            int hy = global_y + STENCIL_RADIUS;
            tile[tile_y + STENCIL_RADIUS][tile_x] = (hy < ny && global_x < nx)
                ? u_curr[z * stride_z + hy * stride_y + global_x] : 0.0f;
        }

        // Corner Halos
        if (local_x < STENCIL_RADIUS && local_y < STENCIL_RADIUS) {
             int hx = global_x - STENCIL_RADIUS, hy = global_y - STENCIL_RADIUS;
             tile[local_y][local_x] = (hx >= 0 && hy >= 0) ? u_curr[z * stride_z + hy * stride_y + hx] : 0.0f;
        }
        if (local_x >= BLOCK_X - STENCIL_RADIUS && local_y < STENCIL_RADIUS) {
             int hx = global_x + STENCIL_RADIUS, hy = global_y - STENCIL_RADIUS;
             tile[local_y][tile_x + STENCIL_RADIUS] = (hx < nx && hy >= 0) ? u_curr[z * stride_z + hy * stride_y + hx] : 0.0f;
        }
        if (local_x < STENCIL_RADIUS && local_y >= BLOCK_Y - STENCIL_RADIUS) {
             int hx = global_x - STENCIL_RADIUS, hy = global_y + STENCIL_RADIUS;
             tile[tile_y + STENCIL_RADIUS][local_x] = (hx >= 0 && hy < ny) ? u_curr[z * stride_z + hy * stride_y + hx] : 0.0f;
        }
        if (local_x >= BLOCK_X - STENCIL_RADIUS && local_y >= BLOCK_Y - STENCIL_RADIUS) {
             int hx = global_x + STENCIL_RADIUS, hy = global_y + STENCIL_RADIUS;
             tile[tile_y + STENCIL_RADIUS][tile_x + STENCIL_RADIUS] = (hx < nx && hy < ny) ? u_curr[z * stride_z + hy * stride_y + hx] : 0.0f;
        }

        __syncthreads();

        if (in_domain) {
            const int idx = z * stride_z + xy_idx;

            // Z-axis
            float lap = d_coeff_hybrid[0] * z_c;
            lap += d_coeff_hybrid[1] * (z_m1 + z_p1);
            lap += d_coeff_hybrid[2] * (z_m2 + z_p2);
            lap += d_coeff_hybrid[3] * (z_m3 + z_p3);
            lap += d_coeff_hybrid[4] * (z_m4 + z_p4);

            // X-axis (Shared Mem)
            lap += d_coeff_hybrid[0] * tile[tile_y][tile_x];
            #pragma unroll
            for (int r = 1; r <= STENCIL_RADIUS; r++) {
                lap += d_coeff_hybrid[r] * (tile[tile_y][tile_x + r] + tile[tile_y][tile_x - r]);
            }

            // Y-axis (Shared Mem)
            lap += d_coeff_hybrid[0] * tile[tile_y][tile_x];
            #pragma unroll
            for (int r = 1; r <= STENCIL_RADIUS; r++) {
                lap += d_coeff_hybrid[r] * (tile[tile_y + r][tile_x] + tile[tile_y - r][tile_x]);
            }

            u_next[idx] = 2.0f * z_c - u_prev[idx] + vel[idx] * lap;
        }

        __syncthreads();

        // Rotate
        z_m4 = z_m3; z_m3 = z_m2; z_m2 = z_m1; z_m1 = z_c;
        z_c  = z_p1; z_p1 = z_p2; z_p2 = z_p3; z_p3 = z_p4;
    }
}

void launch_kernel_hybrid(const float* u_prev, const float* u_curr,
                          float* u_next, const float* vel,
                          int nx, int ny, int nz)
{
    ensure_coefficients_hybrid();
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((nx + BLOCK_X - 1) / BLOCK_X, (ny + BLOCK_Y - 1) / BLOCK_Y, 1);
    kernel_hybrid<<<grid, block>>>(u_prev, u_curr, u_next, vel, nx, ny, nz);
}
