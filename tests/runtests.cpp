// ─── Richter: Test Runner ───────────────────────────────────────────
// Validates kernel correctness against CPU reference implementation.

#include "richter/model.h"
#include "richter/wavelet.h"
#include "richter/boundary.h"
#include "richter/config.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "richter/kernels.h"

// ─── CPU Reference: Naive 3D FDTD Stencil ──────────────────────────
static void cpu_stencil(const float* u_prev, const float* u_curr,
                        float* u_next, const float* vel,
                        int nx, int ny, int nz)
{
    const int stride_y = nx;
    const int stride_z = nx * ny;

    for (int z = STENCIL_RADIUS; z < nz - STENCIL_RADIUS; z++) {
        for (int y = STENCIL_RADIUS; y < ny - STENCIL_RADIUS; y++) {
            for (int x = STENCIL_RADIUS; x < nx - STENCIL_RADIUS; x++) {
                int idx = z * stride_z + y * stride_y + x;

                float laplacian = 3.0f * FD_COEFF[0] * u_curr[idx];
                for (int r = 1; r <= STENCIL_RADIUS; r++) {
                    laplacian += FD_COEFF[r] * (
                        u_curr[idx + r]            + u_curr[idx - r]            +
                        u_curr[idx + r * stride_y] + u_curr[idx - r * stride_y] +
                        u_curr[idx + r * stride_z] + u_curr[idx - r * stride_z]
                    );
                }

                u_next[idx] = 2.0f * u_curr[idx] - u_prev[idx]
                            + vel[idx] * laplacian;
            }
        }
    }
}

// ─── Test: Single Stencil Step ──────────────────────────────────────
static bool test_naive_vs_cpu() {
    printf("[TEST] Naive kernel vs CPU reference... ");

    const int N = 64;  // Small cube for quick validation
    Grid grid = { N, N, N, DEFAULT_DX, DEFAULT_DX, DEFAULT_DX, DEFAULT_DT, 1 };
    size_t total = grid.total_points();

    std::vector<float> h_prev(total, 0.0f);
    std::vector<float> h_curr(total, 0.0f);
    std::vector<float> h_vel(total, 0.0f);
    std::vector<float> h_cpu_out(total, 0.0f);
    std::vector<float> h_gpu_out(total, 0.0f);

    // Initialize with a deterministic pattern
    float coeff = DEFAULT_VELOCITY * DEFAULT_VELOCITY * DEFAULT_DT * DEFAULT_DT
                / (DEFAULT_DX * DEFAULT_DX);
    for (size_t i = 0; i < total; i++) {
        h_curr[i] = sinf((float)i * 0.001f);
        h_prev[i] = h_curr[i] * 0.99f;
        h_vel[i]  = coeff;
    }

    // CPU reference
    cpu_stencil(h_prev.data(), h_curr.data(), h_cpu_out.data(),
                h_vel.data(), N, N, N);

    // GPU: allocate, copy, compute, copy back
    float *d_prev, *d_curr, *d_next, *d_vel;
    size_t bytes = total * sizeof(float);
    cudaMalloc(&d_prev, bytes);
    cudaMalloc(&d_curr, bytes);
    cudaMalloc(&d_next, bytes);
    cudaMalloc(&d_vel,  bytes);

    cudaMemcpy(d_prev, h_prev.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr, h_curr.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_next, 0, bytes);
    cudaMemcpy(d_vel,  h_vel.data(),  bytes, cudaMemcpyHostToDevice);

    launch_kernel_naive(d_prev, d_curr, d_next, d_vel, N, N, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu_out.data(), d_next, bytes, cudaMemcpyDeviceToHost);

    // Compare
    float max_err = 0.0f;
    for (size_t i = 0; i < total; i++) {
        float err = fabsf(h_gpu_out[i] - h_cpu_out[i]);
        if (err > max_err) max_err = err;
    }

    cudaFree(d_prev); cudaFree(d_curr);
    cudaFree(d_next); cudaFree(d_vel);

    if (max_err < 1e-4f) {
        printf("PASS (max error: %.2e)\n", max_err);
        return true;
    } else {
        printf("FAIL (max error: %.2e)\n", max_err);
        return false;
    }
}

// ─── Test: Ricker Wavelet ───────────────────────────────────────────
static bool test_ricker_wavelet() {
    printf("[TEST] Ricker wavelet generation... ");

    const int nt = 1000;
    const float dt = 0.001f;
    const float freq = 15.0f;
    std::vector<float> wavelet(nt);

    generate_ricker_wavelet(wavelet.data(), nt, dt, freq);

    // Wavelet should peak near t = 1/freq ≈ 67ms → sample ~67
    int peak_idx = 0;
    float peak_val = -1e30f;
    for (int i = 0; i < nt; i++) {
        if (wavelet[i] > peak_val) {
            peak_val = wavelet[i];
            peak_idx = i;
        }
    }

    int expected_peak = (int)(1.0f / (freq * dt));
    bool pass = (abs(peak_idx - expected_peak) <= 2) && (peak_val > 0.9f);

    if (pass) {
        printf("PASS (peak at sample %d, value %.4f)\n", peak_idx, peak_val);
    } else {
        printf("FAIL (peak at %d expected ~%d, value %.4f)\n",
               peak_idx, expected_peak, peak_val);
    }
    return pass;
}

// ─── Test: Sponge Boundary Damping ──────────────────────────────────
static bool test_sponge_boundary() {
    printf("[TEST] Sponge boundary damping... ");

    const int N = 64;
    const int sponge_width = 10;
    const float damping_factor = 0.015f;
    size_t total = (size_t)N * N * N;
    size_t bytes = total * sizeof(float);

    // Fill with 1.0 on device
    std::vector<float> h_field(total, 1.0f);
    float* d_field;
    cudaMalloc(&d_field, bytes);
    cudaMemcpy(d_field, h_field.data(), bytes, cudaMemcpyHostToDevice);

    // Apply sponge
    apply_sponge_boundary(d_field, N, N, N, sponge_width, damping_factor);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_field.data(), d_field, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_field);

    // Center should be untouched (1.0)
    int center = (N/2) * N * N + (N/2) * N + (N/2);
    bool center_ok = (fabsf(h_field[center] - 1.0f) < 1e-6f);

    // Corner (0,0,0) should be heavily damped (< 1.0)
    bool corner_ok = (h_field[0] < 0.99f);

    // Edge midpoint should be somewhat damped
    int edge_pt = 0 * N * N + (N/2) * N + (N/2);  // z=0, centered in XY
    bool edge_ok = (h_field[edge_pt] < 0.99f);

    bool pass = center_ok && corner_ok && edge_ok;
    if (pass) {
        printf("PASS (center=%.4f, corner=%.6f, edge=%.6f)\n",
               h_field[center], h_field[0], h_field[edge_pt]);
    } else {
        printf("FAIL (center=%.4f [%s], corner=%.6f [%s], edge=%.6f [%s])\n",
               h_field[center], center_ok ? "ok" : "BAD",
               h_field[0], corner_ok ? "ok" : "BAD",
               h_field[edge_pt], edge_ok ? "ok" : "BAD");
    }
    return pass;
}

// ─── Main ───────────────────────────────────────────────────────────
int main() {
    printf("═══════════════════════════════════════\n");
    printf("  Richter — Test Suite\n");
    printf("═══════════════════════════════════════\n\n");

    int passed = 0, total = 0;

    total++; if (test_ricker_wavelet())  passed++;
    total++; if (test_naive_vs_cpu())    passed++;
    total++; if (test_sponge_boundary()) passed++;

    printf("\n─── Results: %d/%d passed ───\n", passed, total);
    return (passed == total) ? 0 : 1;
}
