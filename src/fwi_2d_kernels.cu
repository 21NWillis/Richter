// ─── Richter: 2D FWI GPU Kernels ────────────────────────────────────
// All spatial kernels for 2D Full Waveform Inversion:
// - 8th-order FD stencil (X + Z)
// - Sponge absorbing boundary
// - Source injection / receiver recording & injection
// - FWI imaging condition (Laplacian-based)
// - Gaussian gradient smoothing (separable 2D)
// - Water layer mask

#include "richter/fwi_2d.h"
#include "richter/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

// ─── 2D Wave Propagation Stencil ────────────────────────────────────
// 8th-order finite-difference, 2D acoustic wave equation:
//   u_next = 2*u_curr - u_prev + vel*(Laplacian_2d(u_curr))
// where vel = v² * dt² / dx²

__global__ void stencil_2d_kernel(const float* __restrict__ u_prev,
                                   const float* __restrict__ u_curr,
                                   float* __restrict__ u_next,
                                   const float* __restrict__ vel,
                                   int nx, int nz)
{
    const float c0 = -205.0f / 72.0f;
    const float c1 =  8.0f  / 5.0f;
    const float c2 = -1.0f  / 5.0f;
    const float c3 =  8.0f  / 315.0f;
    const float c4 = -1.0f  / 560.0f;
    const int R = 4;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || z >= nz) return;

    int idx = z * nx + x;

    // Boundary: zero for points within stencil radius of edge
    if (x < R || x >= nx - R || z < R || z >= nz - R) {
        u_next[idx] = 0.0f;
        return;
    }

    // 2D Laplacian: X + Z directions only
    float lap = 2.0f * c0 * u_curr[idx];  // 2 * c0 for x+z center terms

    // X direction (stride = 1)
    lap += c1 * (u_curr[idx - 1] + u_curr[idx + 1]);
    lap += c2 * (u_curr[idx - 2] + u_curr[idx + 2]);
    lap += c3 * (u_curr[idx - 3] + u_curr[idx + 3]);
    lap += c4 * (u_curr[idx - 4] + u_curr[idx + 4]);

    // Z direction (stride = nx)
    lap += c1 * (u_curr[idx - nx] + u_curr[idx + nx]);
    lap += c2 * (u_curr[idx - 2*nx] + u_curr[idx + 2*nx]);
    lap += c3 * (u_curr[idx - 3*nx] + u_curr[idx + 3*nx]);
    lap += c4 * (u_curr[idx - 4*nx] + u_curr[idx + 4*nx]);

    u_next[idx] = 2.0f * u_curr[idx] - u_prev[idx] + vel[idx] * lap;
}

void launch_stencil_2d(const float* u_prev, const float* u_curr,
                       float* u_next, const float* vel,
                       int nx, int nz)
{
    dim3 block(32, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (nz + block.y - 1) / block.y);
    stencil_2d_kernel<<<grid, block>>>(u_prev, u_curr, u_next, vel, nx, nz);
}

// ─── 2D Sponge Absorbing Boundary ──────────────────────────────────

__global__ void sponge_2d_kernel(float* __restrict__ u,
                                  int nx, int nz,
                                  int sponge_width, float alpha)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || z >= nz) return;

    int dist_x = min(x, nx - 1 - x);
    int dist_z = min(z, nz - 1 - z);

    if (dist_x >= sponge_width && dist_z >= sponge_width) return;

    float damping = 1.0f;
    float sw = (float)sponge_width;

    if (dist_x < sponge_width) {
        float normalized = (sw - (float)dist_x) / sw;
        damping *= expf(-alpha * normalized * normalized);
    }

    if (dist_z < sponge_width) {
        float normalized = (sw - (float)dist_z) / sw;
        damping *= expf(-alpha * normalized * normalized);
    }

    u[z * nx + x] *= damping;
}

void apply_sponge_2d(float* u, int nx, int nz,
                     int sponge_width, float damping_factor)
{
    if (sponge_width <= 0) return;
    float alpha = damping_factor * 40.0f;

    dim3 block(32, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (nz + block.y - 1) / block.y);
    sponge_2d_kernel<<<grid, block>>>(u, nx, nz, sponge_width, alpha);
}

// ─── 2D Source Injection ────────────────────────────────────────────
// Single kernel call instead of host round-trip

__global__ void inject_source_2d_kernel(float* __restrict__ u,
                                         int idx, float amplitude)
{
    u[idx] += amplitude;
}

void inject_source_2d(float* d_u, int sx, int sz, float amplitude,
                      int nx, int nz)
{
    int idx = sz * nx + sx;
    inject_source_2d_kernel<<<1, 1>>>(d_u, idx, amplitude);
}

// ─── 2D Receiver Recording ─────────────────────────────────────────

__global__ void record_receivers_2d_kernel(const float* __restrict__ u,
                                            float* __restrict__ traces,
                                            const int* __restrict__ rx,
                                            const int* __restrict__ rz,
                                            int num_receivers, int nx,
                                            int t, int nt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_receivers) return;
    int idx = rz[i] * nx + rx[i];
    traces[i * nt + t] = u[idx];
}

void record_receivers_2d(const float* d_u, float* d_traces,
                         const int* d_rx, const int* d_rz,
                         int num_receivers, int nx,
                         int t, int nt)
{
    int threads = 256;
    int blocks = (num_receivers + threads - 1) / threads;
    record_receivers_2d_kernel<<<blocks, threads>>>(d_u, d_traces,
                                                     d_rx, d_rz,
                                                     num_receivers, nx,
                                                     t, nt);
}

// ─── 2D Receiver Injection (Adjoint) ────────────────────────────────

__global__ void inject_receivers_2d_kernel(float* __restrict__ u,
                                            const float* __restrict__ traces,
                                            const int* __restrict__ rx,
                                            const int* __restrict__ rz,
                                            int num_receivers, int nx,
                                            int t, int nt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_receivers) return;
    int idx = rz[i] * nx + rx[i];
    u[idx] += traces[i * nt + t];
}

void inject_receivers_2d(float* d_u, const float* d_traces,
                         const int* d_rx, const int* d_rz,
                         int num_receivers, int nx,
                         int t, int nt)
{
    int threads = 256;
    int blocks = (num_receivers + threads - 1) / threads;
    inject_receivers_2d_kernel<<<blocks, threads>>>(d_u, d_traces,
                                                     d_rx, d_rz,
                                                     num_receivers, nx,
                                                     t, nt);
}

// ─── 2D FWI Imaging Condition (Laplacian-based) ─────────────────────
// Proper FWI gradient: grad += Laplacian(u_source) * u_adjoint
// This gives the correct velocity sensitivity kernel, unlike the
// cross-correlation (src * adj) used for RTM reflectivity imaging.

__global__ void fwi_imaging_2d_kernel(const float* __restrict__ d_src,
                                       const float* __restrict__ d_adj,
                                       float* __restrict__ d_gradient,
                                       float* __restrict__ d_illum,
                                       int nx, int nz)
{
    const float c0 = -205.0f / 72.0f;
    const float c1 =  8.0f  / 5.0f;
    const float c2 = -1.0f  / 5.0f;
    const float c3 =  8.0f  / 315.0f;
    const float c4 = -1.0f  / 560.0f;
    const int R = 4;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || z >= nz) return;
    if (x < R || x >= nx - R || z < R || z >= nz - R) return;

    int idx = z * nx + x;

    // 2D Laplacian of source wavefield
    float lap = 2.0f * c0 * d_src[idx];

    // X direction
    lap += c1 * (d_src[idx - 1] + d_src[idx + 1]);
    lap += c2 * (d_src[idx - 2] + d_src[idx + 2]);
    lap += c3 * (d_src[idx - 3] + d_src[idx + 3]);
    lap += c4 * (d_src[idx - 4] + d_src[idx + 4]);

    // Z direction
    lap += c1 * (d_src[idx - nx] + d_src[idx + nx]);
    lap += c2 * (d_src[idx - 2*nx] + d_src[idx + 2*nx]);
    lap += c3 * (d_src[idx - 3*nx] + d_src[idx + 3*nx]);
    lap += c4 * (d_src[idx - 4*nx] + d_src[idx + 4*nx]);

    float adj = d_adj[idx];

    // FWI gradient: Laplacian(source) * adjoint
    d_gradient[idx] += lap * adj;

    // Illumination: Laplacian(source)² for consistent preconditioning
    d_illum[idx] += lap * lap;
}

void apply_fwi_imaging_2d(const float* d_src, const float* d_adj,
                          float* d_gradient, float* d_illum,
                          int nx, int nz)
{
    dim3 block(32, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (nz + block.y - 1) / block.y);
    fwi_imaging_2d_kernel<<<grid, block>>>(d_src, d_adj, d_gradient, d_illum,
                                            nx, nz);
}

// ─── 2D Gaussian Smoothing (Separable) ──────────────────────────────

__global__ void smooth_x_2d_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    const float* __restrict__ weights,
                                    int radius, int nx, int nz)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= nz) return;

    for (int x = 0; x < nx; x++) {
        float sum = 0.0f;
        float wsum = 0.0f;
        for (int r = -radius; r <= radius; r++) {
            int xx = x + r;
            if (xx >= 0 && xx < nx) {
                float w = weights[r + radius];
                sum += w * in[z * nx + xx];
                wsum += w;
            }
        }
        out[z * nx + x] = sum / wsum;
    }
}

__global__ void smooth_z_2d_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    const float* __restrict__ weights,
                                    int radius, int nx, int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;

    for (int z = 0; z < nz; z++) {
        float sum = 0.0f;
        float wsum = 0.0f;
        for (int r = -radius; r <= radius; r++) {
            int zz = z + r;
            if (zz >= 0 && zz < nz) {
                float w = weights[r + radius];
                sum += w * in[zz * nx + x];
                wsum += w;
            }
        }
        out[z * nx + x] = sum / wsum;
    }
}

void smooth_gradient_2d(float* d_gradient, float* d_temp,
                        int nx, int nz, float sigma)
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

    float* d_weights;
    cudaMalloc(&d_weights, ksize * sizeof(float));
    cudaMemcpy(d_weights, h_weights.data(), ksize * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;

    // X pass: gradient -> temp
    {
        int blocks = (nz + threads - 1) / threads;
        smooth_x_2d_kernel<<<blocks, threads>>>(d_gradient, d_temp, d_weights, radius, nx, nz);
    }

    // Z pass: temp -> gradient
    {
        int blocks = (nx + threads - 1) / threads;
        smooth_z_2d_kernel<<<blocks, threads>>>(d_temp, d_gradient, d_weights, radius, nx, nz);
    }

    cudaFree(d_weights);
}

// ─── Direct Arrival Muting ──────────────────────────────────────────
// Zeros out trace samples before the direct arrival time for each
// source-receiver pair. This removes the surface wave energy that
// dominates the gradient near the surface but carries no deep-structure
// information. Applied to the residual before the backward pass.

__global__ void mute_direct_arrivals_2d_kernel(float* __restrict__ traces,
                                                const int* __restrict__ rx,
                                                const int* __restrict__ rz,
                                                int sx, int sz,
                                                int num_receivers, int nt,
                                                float dt, float dx,
                                                float v_direct, float taper_samples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_receivers) return;

    // Compute offset distance (in meters)
    float dist_x = (float)(rx[i] - sx) * dx;
    float dist_z = (float)(rz[i] - sz) * dx;
    float offset = sqrtf(dist_x * dist_x + dist_z * dist_z);

    // Direct arrival time + buffer
    float t_direct = offset / v_direct;
    int mute_end = (int)(t_direct / dt) + (int)taper_samples;

    // Zero out everything before the direct arrival
    for (int t = 0; t < nt && t < mute_end; t++) {
        // Cosine taper in the last taper_samples before mute_end
        float weight = 0.0f;
        int taper_start = mute_end - (int)taper_samples;
        if (t >= taper_start && taper_samples > 0.0f) {
            float frac = (float)(t - taper_start) / taper_samples;
            weight = 0.5f * (1.0f - cosf(3.14159265f * frac));  // half-cosine taper
        }
        traces[i * nt + t] *= weight;
    }
}

void mute_direct_arrivals_2d(float* d_traces,
                              const int* d_rx, const int* d_rz,
                              int sx, int sz,
                              int num_receivers, int nt,
                              float dt, float dx,
                              float v_direct, float taper_samples)
{
    int threads = 256;
    int blocks = (num_receivers + threads - 1) / threads;
    mute_direct_arrivals_2d_kernel<<<blocks, threads>>>(
        d_traces, d_rx, d_rz, sx, sz,
        num_receivers, nt, dt, dx, v_direct, taper_samples);
}

// ─── Depth Gradient Scaling ─────────────────────────────────────────
// Multiplies gradient by (z - water_depth)^power / (nz - water_depth)^power.
// This compensates for the adjoint wavefield being weaker at depth:
// the source illumination normalization fixes the source side, but the
// receiver side (adjoint) still decays with depth. Depth weighting
// boosts the deep gradient to match.

__global__ void depth_scale_gradient_2d_kernel(float* __restrict__ gradient,
                                                int nx, int nz,
                                                int water_depth, float power)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || z >= nz) return;
    if (z <= water_depth) return;  // water mask handles this separately

    float active_depth = (float)(nz - water_depth);
    float rel_z = (float)(z - water_depth) / active_depth;  // 0 at water_depth, 1 at bottom
    float weight = powf(rel_z, power);

    gradient[z * nx + x] *= weight;
}

void apply_depth_scaling_2d(float* d_gradient, int nx, int nz,
                             int water_depth, float power)
{
    if (power <= 0.0f) return;

    dim3 block(32, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (nz + block.y - 1) / block.y);
    depth_scale_gradient_2d_kernel<<<grid, block>>>(d_gradient, nx, nz,
                                                     water_depth, power);
}

// ─── Per-Row Gradient Normalization ────────────────────────────────
// Normalizes each depth row by its L2 norm to equalize gradient energy
// across depths. This makes the gradient depth-invariant so that anomalies
// like the lens stand out by their lateral structure rather than being
// drowned out by depth-dependent amplitude variations.

void normalize_gradient_per_row_2d(float* d_gradient, int nx, int nz,
                                    int water_depth)
{
    // Download gradient to host for row-wise normalization
    size_t N = (size_t)nx * nz;
    std::vector<float> h_grad(N);
    cudaMemcpy(h_grad.data(), d_gradient, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int z = water_depth + 1; z < nz; z++) {
        float row_norm_sq = 0.0f;
        for (int x = 0; x < nx; x++) {
            float v = h_grad[z * nx + x];
            row_norm_sq += v * v;
        }
        float row_norm = sqrtf(row_norm_sq);
        if (row_norm > 0.0f) {
            float scale = 1.0f / row_norm;
            for (int x = 0; x < nx; x++) {
                h_grad[z * nx + x] *= scale;
            }
        }
    }

    cudaMemcpy(d_gradient, h_grad.data(), N * sizeof(float), cudaMemcpyHostToDevice);
}

// ─── 2D Sponge Zone Gradient Mask ───────────────────────────────────
// Zeros gradient in the sponge absorption zone (left, right, bottom).
// Top is handled separately by the water mask.
// Without this, depth scaling amplifies boundary artifacts at the bottom
// of the model, creating spurious maxima that dominate the gradient and
// prevent recovery of deep features like the lens anomaly.

__global__ void sponge_gradient_mask_2d_kernel(float* __restrict__ gradient,
                                                int nx, int nz,
                                                int sponge_width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || z >= nz) return;

    bool in_sponge = (x < sponge_width) ||
                     (x >= nx - sponge_width) ||
                     (z >= nz - sponge_width);

    if (in_sponge) {
        gradient[z * nx + x] = 0.0f;
    }
}

void apply_sponge_gradient_mask_2d(float* d_gradient, int nx, int nz,
                                    int sponge_width)
{
    if (sponge_width <= 0) return;

    dim3 block(32, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (nz + block.y - 1) / block.y);
    sponge_gradient_mask_2d_kernel<<<grid, block>>>(d_gradient, nx, nz,
                                                     sponge_width);
}

// ─── 2D Water Layer Mask ────────────────────────────────────────────

__global__ void water_mask_2d_kernel(float* __restrict__ gradient,
                                      int nx, int nz, int water_depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || z >= nz) return;
    if (z <= water_depth) {
        gradient[z * nx + x] = 0.0f;
    }
}

void apply_water_mask_2d(float* d_gradient, int nx, int nz, int water_depth)
{
    if (water_depth <= 0) return;

    dim3 block(32, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (nz + block.y - 1) / block.y);
    water_mask_2d_kernel<<<grid, block>>>(d_gradient, nx, nz, water_depth);
}

// ─── Layer-Stripping Shallow Freeze ────────────────────────────────
// Zeros gradient above freeze_depth with a cosine taper to avoid
// discontinuities. Used to freeze converged shallow velocities and
// force the optimizer to update deeper features like the lens.

__global__ void shallow_freeze_2d_kernel(float* __restrict__ gradient,
                                          int nx, int nz,
                                          int freeze_depth, int taper_width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || z >= nz) return;

    if (z < freeze_depth - taper_width) {
        // Fully frozen
        gradient[z * nx + x] = 0.0f;
    } else if (z < freeze_depth) {
        // Cosine taper: 0 at (freeze_depth - taper_width), 1 at freeze_depth
        float frac = (float)(z - (freeze_depth - taper_width)) / (float)taper_width;
        float weight = 0.5f * (1.0f - cosf(3.14159265f * frac));
        gradient[z * nx + x] *= weight;
    }
    // z >= freeze_depth: no change
}

void apply_shallow_freeze_2d(float* d_gradient, int nx, int nz,
                              int freeze_depth, int taper_width)
{
    if (freeze_depth <= 0) return;
    if (taper_width < 0) taper_width = 0;

    dim3 block(32, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (nz + block.y - 1) / block.y);
    shallow_freeze_2d_kernel<<<grid, block>>>(d_gradient, nx, nz,
                                               freeze_depth, taper_width);
}
