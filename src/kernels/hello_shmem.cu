// Richter: Shared Memory 2.5D Tiling Kernel
// Implementation B: XY plane tiled into shared memory, Z streamed.

#include "richter/kernels.h"
#include "richter/config.h"
#include <cstdio>

// Constant memory for FD coefficients (file-scope)
__constant__ float d_coeff_shmem[STENCIL_RADIUS + 1];

static bool coeffs_loaded_shmem = false;

static void ensure_coefficients_shmem() {
    if (!coeffs_loaded_shmem) {
        cudaMemcpyToSymbol(d_coeff_shmem, FD_COEFF, sizeof(FD_COEFF));
        coeffs_loaded_shmem = true;
    }
}

// Shared memory tile dimensions
// The tile includes R-wide halo on each side for the stencil.
//   nx = x stride
//   ny = y stride
//   nz = z stride
constexpr int TILE_X = BLOCK_X + 2 * STENCIL_RADIUS;
constexpr int TILE_Y = BLOCK_Y + 2 * STENCIL_RADIUS;

// The 2.5D Shared Memory Kernel
__global__ void kernel_shmem(const float* __restrict__ u_prev,
                             const float* __restrict__ u_curr,
                             float*       __restrict__ u_next,
                             const float* __restrict__ vel,
                             int nx, int ny, int nz)
{
    __shared__ float tile[TILE_Y][TILE_X];


    const int local_x = threadIdx.x; 
    const int local_y = threadIdx.y; 

    const int global_x = blockIdx.x * BLOCK_X + local_x;
    const int global_y = blockIdx.y * BLOCK_Y + local_y;
    const int global_z = blockIdx.z;  

    // Memory layout strides
    const int stride_y = nx;
    const int stride_z = nx * ny;

    // Tile coordinates (offset by R so halo sits at indices 0..R-1)
    const int tile_x = local_x + STENCIL_RADIUS;
    const int tile_y = local_y + STENCIL_RADIUS;


    if (global_x < nx && global_y < ny && global_z < nz) {
        tile[tile_y][tile_x] = u_curr[global_z * stride_z + global_y * stride_y + global_x];
    }

    if (local_x < STENCIL_RADIUS) {
        int halo_x = global_x - STENCIL_RADIUS;
        if (halo_x >= 0 && global_y < ny && global_z < nz)
            tile[tile_y][local_x] = u_curr[global_z * stride_z + global_y * stride_y + halo_x];
        else
            tile[tile_y][local_x] = 0.0f;
    }

    if (local_x >= BLOCK_X - STENCIL_RADIUS) {
        int halo_x = global_x + STENCIL_RADIUS;
        if (halo_x < nx && global_y < ny && global_z < nz)
            tile[tile_y][tile_x + STENCIL_RADIUS] = u_curr[global_z * stride_z + global_y * stride_y + halo_x];
        else
            tile[tile_y][tile_x + STENCIL_RADIUS] = 0.0f;
    }

    if (local_y < STENCIL_RADIUS) {
        int halo_y = global_y - STENCIL_RADIUS;
        if (halo_y >= 0 && global_x < nx && global_z < nz)
            tile[local_y][tile_x] = u_curr[global_z * stride_z + halo_y * stride_y + global_x];
        else
            tile[local_y][tile_x] = 0.0f;
    }

    if (local_y >= BLOCK_Y - STENCIL_RADIUS) {
        int halo_y = global_y + STENCIL_RADIUS;
        if (halo_y < ny && global_x < nx && global_z < nz)
            tile[tile_y + STENCIL_RADIUS][tile_x] = u_curr[global_z * stride_z + halo_y * stride_y + global_x];
        else
            tile[tile_y + STENCIL_RADIUS][tile_x] = 0.0f;
    }

    // Corners

    if (local_x < STENCIL_RADIUS && local_y < STENCIL_RADIUS) {
        int halo_x = global_x - STENCIL_RADIUS;
        int halo_y = global_y - STENCIL_RADIUS;
        if (halo_x >= 0 && halo_y >= 0 && global_z < nz)
            tile[local_y][local_x] = u_curr[global_z * stride_z + halo_y * stride_y + halo_x];
        else
            tile[local_y][local_x] = 0.0f;
    }

    if (local_x >= BLOCK_X - STENCIL_RADIUS && local_y < STENCIL_RADIUS) {
        int halo_x = global_x + STENCIL_RADIUS;
        int halo_y = global_y - STENCIL_RADIUS;
        if (halo_x < nx && halo_y >= 0 && global_z < nz)
            tile[local_y][tile_x + STENCIL_RADIUS] = u_curr[global_z * stride_z + halo_y * stride_y + halo_x];
        else
            tile[local_y][tile_x + STENCIL_RADIUS] = 0.0f;
    }

    if (local_x < STENCIL_RADIUS && local_y >= BLOCK_Y - STENCIL_RADIUS) {
        int halo_x = global_x - STENCIL_RADIUS;
        int halo_y = global_y + STENCIL_RADIUS;
        if (halo_x >= 0 && halo_y < ny && global_z < nz)
            tile[tile_y + STENCIL_RADIUS][local_x] = u_curr[global_z * stride_z + halo_y * stride_y + halo_x];
        else
            tile[tile_y + STENCIL_RADIUS][local_x] = 0.0f;
    }

    if (local_x >= BLOCK_X - STENCIL_RADIUS && local_y >= BLOCK_Y - STENCIL_RADIUS) {
        int halo_x = global_x + STENCIL_RADIUS;
        int halo_y = global_y + STENCIL_RADIUS;
        if (halo_x < nx && halo_y < ny && global_z < nz)
            tile[tile_y + STENCIL_RADIUS][tile_x + STENCIL_RADIUS] = u_curr[global_z * stride_z + halo_y * stride_y + halo_x];
        else
            tile[tile_y + STENCIL_RADIUS][tile_x + STENCIL_RADIUS] = 0.0f;
    }

    __syncthreads();


    // Halo threads return
    if (global_x < STENCIL_RADIUS || global_x >= nx - STENCIL_RADIUS) return;
    if (global_y < STENCIL_RADIUS || global_y >= ny - STENCIL_RADIUS) return;
    if (global_z < STENCIL_RADIUS || global_z >= nz - STENCIL_RADIUS) return;

    const int flat_idx = global_z * stride_z + global_y * stride_y + global_x;

    float laplacian = 3.0f * d_coeff_shmem[0] * tile[tile_y][tile_x];

    #pragma unroll
    for (int r = 1; r <= STENCIL_RADIUS; r++) {
        laplacian += d_coeff_shmem[r] * (tile[tile_y][tile_x + r] + tile[tile_y][tile_x - r]);
        laplacian += d_coeff_shmem[r] * (tile[tile_y + r][tile_x] + tile[tile_y - r][tile_x]);

        laplacian += d_coeff_shmem[r] * (
            u_curr[flat_idx + r * stride_z] + u_curr[flat_idx - r * stride_z]
        );
    }

    // Time-stepping
    u_next[flat_idx] = 2.0f * tile[tile_y][tile_x] - u_prev[flat_idx] + vel[flat_idx] * laplacian;
}


// Host Launch Wrapper
void launch_kernel_shmem(const float* u_prev, const float* u_curr,
                         float* u_next, const float* vel,
                         int nx, int ny, int nz)
{
    ensure_coefficients_shmem();

    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((nx + BLOCK_X - 1) / BLOCK_X,
              (ny + BLOCK_Y - 1) / BLOCK_Y,
              nz);

    kernel_shmem<<<grid, block>>>(u_prev, u_curr, u_next, vel, nx, ny, nz);
}
