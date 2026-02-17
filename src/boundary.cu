#include "richter/boundary.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>


__global__ void sponge_boundary_kernel(float* __restrict__ u,
                                       int nx, int ny, int nz,
                                       int sponge_width, float alpha)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    // Distance from nearest edge on each axis
    int dist_x = min(x, nx - 1 - x);
    int dist_y = min(y, ny - 1 - y);
    int dist_z = min(z, nz - 1 - z);

    // If completely inside the sponge-free interior, nothing to do
    if (dist_x >= sponge_width && dist_y >= sponge_width && dist_z >= sponge_width)
        return;

    float damping = 1.0f;
    float sw = (float)sponge_width;

    // X-axis damping
    if (dist_x < sponge_width) {
        float normalized = (sw - (float)dist_x) / sw;  // 1.0 at edge, 0.0 at interior
        damping *= expf(-alpha * normalized * normalized);
    }

    // Y-axis damping
    if (dist_y < sponge_width) {
        float normalized = (sw - (float)dist_y) / sw;
        damping *= expf(-alpha * normalized * normalized);
    }

    // Z-axis damping
    if (dist_z < sponge_width) {
        float normalized = (sw - (float)dist_z) / sw;
        damping *= expf(-alpha * normalized * normalized);
    }

    int idx = z * nx * ny + y * nx + x;
    u[idx] *= damping;
}

void apply_sponge_boundary(float* u, int nx, int ny, int nz,
                           int sponge_width, float damping_factor)
{
    if (sponge_width <= 0) return;

    float alpha = damping_factor * 40.0f;  // scale to reasonable range

    dim3 block(32, 16, 1);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              nz);

    sponge_boundary_kernel<<<grid, block>>>(u, nx, ny, nz, sponge_width, alpha);
}
