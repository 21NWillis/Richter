// ─── Richter: FWI GPU Kernels ──────────────────────────────────────
// Kernels for Full Waveform Inversion:
// - Residual computation and L2 misfit reduction
// - Velocity ↔ coefficient conversion
// - Velocity update with clamping
// - 3D Gaussian gradient smoothing (separable)
// - Water layer masking

#include "richter/fwi.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ─── Residual + Misfit ─────────────────────────────────────────────

__global__ void residual_kernel(const float* __restrict__ syn,
                                const float* __restrict__ obs,
                                float* __restrict__ residual,
                                float* __restrict__ partial_sums,
                                int n)
{
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float r = 0.0f;
    if (idx < n) {
        r = syn[idx] - obs[idx];
        residual[idx] = r;
    }
    sdata[tid] = r * r;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

float compute_residual_and_misfit(const float* d_synthetic,
                                  const float* d_observed,
                                  float* d_residual,
                                  int num_receivers, int nt)
{
    int n = num_receivers * nt;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Allocate partial sums and result
    float* d_partial;
    float* d_result;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    // Compute residuals and per-block partial L2 sums
    residual_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_synthetic, d_observed, d_residual, d_partial, n);

    // Reduce partial sums to scalar
    int red_threads = 256;
    int red_blocks = (blocks + red_threads - 1) / red_threads;
    reduce_sum_kernel<<<red_blocks, red_threads, red_threads * sizeof(float)>>>(
        d_partial, d_result, blocks);

    float h_misfit;
    cudaMemcpy(&h_misfit, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_partial);
    cudaFree(d_result);

    return 0.5f * h_misfit;  // L2 misfit = 0.5 * ||r||^2
}

// ─── Forward-only Misfit (no residual output needed) ───────────────

float compute_misfit_only(const float* d_synthetic,
                          const float* d_observed,
                          int num_receivers, int nt)
{
    int n = num_receivers * nt;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float* d_partial;
    float* d_result;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    // Reuse residual kernel with a throwaway residual buffer
    // Actually, let's make a dedicated kernel that doesn't write residual
    // For now, allocate a temp residual buffer
    float* d_temp;
    cudaMalloc(&d_temp, n * sizeof(float));

    residual_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_synthetic, d_observed, d_temp, d_partial, n);

    int red_threads = 256;
    int red_blocks = (blocks + red_threads - 1) / red_threads;
    reduce_sum_kernel<<<red_blocks, red_threads, red_threads * sizeof(float)>>>(
        d_partial, d_result, blocks);

    float h_misfit;
    cudaMemcpy(&h_misfit, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_partial);
    cudaFree(d_result);

    return 0.5f * h_misfit;
}

// ─── Velocity ↔ Coefficient Conversion ─────────────────────────────

__global__ void vel_to_coeff_kernel(const float* __restrict__ v,
                                    float* __restrict__ coeff,
                                    float dt2_dx2, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float vel = v[idx];
        coeff[idx] = vel * vel * dt2_dx2;
    }
}

void velocity_to_coefficient(const float* d_vel_phys, float* d_vel_coeff,
                              float dt, float dx, size_t n)
{
    float dt2_dx2 = (dt * dt) / (dx * dx);
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    vel_to_coeff_kernel<<<blocks, threads>>>(d_vel_phys, d_vel_coeff, dt2_dx2, (int)n);
}

__global__ void coeff_to_vel_kernel(const float* __restrict__ coeff,
                                    float* __restrict__ v,
                                    float dx_dt, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] = sqrtf(coeff[idx]) * dx_dt;
    }
}

void coefficient_to_velocity(const float* d_vel_coeff, float* d_vel_phys,
                              float dt, float dx, size_t n)
{
    float dx_dt = dx / dt;
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    coeff_to_vel_kernel<<<blocks, threads>>>(d_vel_coeff, d_vel_phys, dx_dt, (int)n);
}

// ─── Velocity Update ───────────────────────────────────────────────

__global__ void update_velocity_kernel(float* __restrict__ vel_coeff,
                                       const float* __restrict__ gradient,
                                       float step_size,
                                       float c_min, float c_max,
                                       int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float c = vel_coeff[idx] - step_size * gradient[idx];
        vel_coeff[idx] = fminf(fmaxf(c, c_min), c_max);
    }
}

void apply_velocity_update(float* d_vel_coeff, const float* d_gradient,
                            float step_size, float dt, float dx,
                            float v_min, float v_max, size_t n)
{
    float dt2_dx2 = (dt * dt) / (dx * dx);
    float c_min = v_min * v_min * dt2_dx2;
    float c_max = v_max * v_max * dt2_dx2;
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    update_velocity_kernel<<<blocks, threads>>>(
        d_vel_coeff, d_gradient, step_size, c_min, c_max, (int)n);
}

// ─── Gaussian Smoothing (Separable 3D) ─────────────────────────────

__global__ void smooth_x_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                const float* __restrict__ weights,
                                int radius, int nx, int ny, int nz)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= ny || z >= nz) return;

    for (int x = 0; x < nx; x++) {
        float sum = 0.0f;
        float wsum = 0.0f;
        for (int r = -radius; r <= radius; r++) {
            int xx = x + r;
            if (xx >= 0 && xx < nx) {
                float w = weights[r + radius];
                sum += w * in[z * nx * ny + y * nx + xx];
                wsum += w;
            }
        }
        out[z * nx * ny + y * nx + x] = sum / wsum;
    }
}

__global__ void smooth_y_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                const float* __restrict__ weights,
                                int radius, int nx, int ny, int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || z >= nz) return;

    for (int y = 0; y < ny; y++) {
        float sum = 0.0f;
        float wsum = 0.0f;
        for (int r = -radius; r <= radius; r++) {
            int yy = y + r;
            if (yy >= 0 && yy < ny) {
                float w = weights[r + radius];
                sum += w * in[z * nx * ny + yy * nx + x];
                wsum += w;
            }
        }
        out[z * nx * ny + y * nx + x] = sum / wsum;
    }
}

__global__ void smooth_z_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                const float* __restrict__ weights,
                                int radius, int nx, int ny, int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    for (int z = 0; z < nz; z++) {
        float sum = 0.0f;
        float wsum = 0.0f;
        for (int r = -radius; r <= radius; r++) {
            int zz = z + r;
            if (zz >= 0 && zz < nz) {
                float w = weights[r + radius];
                sum += w * in[zz * nx * ny + y * nx + x];
                wsum += w;
            }
        }
        out[z * nx * ny + y * nx + x] = sum / wsum;
    }
}

void smooth_gradient_3d(float* d_gradient, float* d_temp,
                         int nx, int ny, int nz, float sigma)
{
    if (sigma <= 0.0f) return;

    int radius = (int)ceilf(3.0f * sigma);
    if (radius < 1) radius = 1;
    int ksize = 2 * radius + 1;

    // Build Gaussian weights on host
    std::vector<float> h_weights(ksize);
    float sum = 0.0f;
    for (int i = 0; i < ksize; i++) {
        float x = (float)(i - radius);
        h_weights[i] = expf(-0.5f * x * x / (sigma * sigma));
        sum += h_weights[i];
    }
    for (int i = 0; i < ksize; i++) h_weights[i] /= sum;

    // Upload weights
    float* d_weights;
    cudaMalloc(&d_weights, ksize * sizeof(float));
    cudaMemcpy(d_weights, h_weights.data(), ksize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);

    // X pass: gradient -> temp
    {
        dim3 grid((ny + block.x - 1) / block.x, (nz + block.y - 1) / block.y);
        smooth_x_kernel<<<grid, block>>>(d_gradient, d_temp, d_weights, radius, nx, ny, nz);
    }

    // Y pass: temp -> gradient
    {
        dim3 grid((nx + block.x - 1) / block.x, (nz + block.y - 1) / block.y);
        smooth_y_kernel<<<grid, block>>>(d_temp, d_gradient, d_weights, radius, nx, ny, nz);
    }

    // Z pass: gradient -> temp, then copy back
    {
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
        smooth_z_kernel<<<grid, block>>>(d_gradient, d_temp, d_weights, radius, nx, ny, nz);
    }

    // Copy result back: temp -> gradient
    size_t bytes = (size_t)nx * ny * nz * sizeof(float);
    cudaMemcpy(d_gradient, d_temp, bytes, cudaMemcpyDeviceToDevice);

    cudaFree(d_weights);
}

// ─── FWI Imaging Condition (Laplacian-based) ─────────────────────────
// Proper FWI gradient: grad += Laplacian(u_source) * u_adjoint
// This gives correct sensitivity with depth penetration, unlike the
// zero-lag cross-correlation (u_source * u_adjoint) used in RTM.

__global__ void fwi_imaging_kernel(const float* __restrict__ d_src,
                                    const float* __restrict__ d_adj,
                                    float* __restrict__ d_gradient,
                                    float* __restrict__ d_illum,
                                    int nx, int ny, int nz)
{
    // 8th-order FD coefficients for Laplacian
    const float c0 = -205.0f / 72.0f;
    const float c1 =  8.0f  / 5.0f;
    const float c2 = -1.0f  / 5.0f;
    const float c3 =  8.0f  / 315.0f;
    const float c4 = -1.0f  / 560.0f;
    const int R = 4;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    // Only compute Laplacian in interior (stencil needs R neighbors)
    if (x < R || x >= nx - R || y < R || y >= ny - R || z < R || z >= nz - R) return;

    int idx = z * nx * ny + y * nx + x;

    // Compute 3D Laplacian of source wavefield
    float lap = 3.0f * c0 * d_src[idx];  // 3 * c0 for x+y+z center terms

    // X direction
    lap += c1 * (d_src[idx - 1] + d_src[idx + 1]);
    lap += c2 * (d_src[idx - 2] + d_src[idx + 2]);
    lap += c3 * (d_src[idx - 3] + d_src[idx + 3]);
    lap += c4 * (d_src[idx - 4] + d_src[idx + 4]);

    // Y direction (stride = nx)
    lap += c1 * (d_src[idx - nx] + d_src[idx + nx]);
    lap += c2 * (d_src[idx - 2*nx] + d_src[idx + 2*nx]);
    lap += c3 * (d_src[idx - 3*nx] + d_src[idx + 3*nx]);
    lap += c4 * (d_src[idx - 4*nx] + d_src[idx + 4*nx]);

    // Z direction (stride = nx*ny)
    int sxy = nx * ny;
    lap += c1 * (d_src[idx - sxy] + d_src[idx + sxy]);
    lap += c2 * (d_src[idx - 2*sxy] + d_src[idx + 2*sxy]);
    lap += c3 * (d_src[idx - 3*sxy] + d_src[idx + 3*sxy]);
    lap += c4 * (d_src[idx - 4*sxy] + d_src[idx + 4*sxy]);

    float adj = d_adj[idx];

    // FWI gradient: Laplacian(source) * adjoint
    d_gradient[idx] += lap * adj;

    // Illumination: Laplacian(source)^2 for consistency
    d_illum[idx] += lap * lap;
}

void apply_fwi_imaging_condition(const float* d_src, const float* d_adj,
                                  float* d_gradient, float* d_illum,
                                  int nx, int ny, int nz)
{
    dim3 block(32, 16, 1);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              nz);
    fwi_imaging_kernel<<<grid, block>>>(d_src, d_adj, d_gradient, d_illum,
                                         nx, ny, nz);
}

// ─── Water Layer Mask ──────────────────────────────────────────────

__global__ void water_mask_kernel(float* __restrict__ gradient,
                                  int nx, int ny, int nz,
                                  int water_depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;
    if (z < water_depth) {
        gradient[z * nx * ny + y * nx + x] = 0.0f;
    }
}

void apply_water_mask(float* d_gradient, int nx, int ny, int nz, int water_depth)
{
    if (water_depth <= 0) return;

    dim3 block(32, 16, 1);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              nz);
    water_mask_kernel<<<grid, block>>>(d_gradient, nx, ny, nz, water_depth);
}
