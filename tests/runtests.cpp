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
#include "richter/rtm.h"

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

// ─── Test: Shared Memory Kernel vs CPU ──────────────────────────────
static bool test_shmem_vs_cpu() {
    printf("[TEST] Shared memory kernel vs CPU reference... ");

    const int N = 64;
    Grid grid = { N, N, N, DEFAULT_DX, DEFAULT_DX, DEFAULT_DX, DEFAULT_DT, 1 };
    size_t total = grid.total_points();

    std::vector<float> h_prev(total, 0.0f);
    std::vector<float> h_curr(total, 0.0f);
    std::vector<float> h_vel(total, 0.0f);
    std::vector<float> h_cpu_out(total, 0.0f);
    std::vector<float> h_gpu_out(total, 0.0f);

    float coeff = DEFAULT_VELOCITY * DEFAULT_VELOCITY * DEFAULT_DT * DEFAULT_DT
                / (DEFAULT_DX * DEFAULT_DX);
    for (size_t i = 0; i < total; i++) {
        h_curr[i] = sinf((float)i * 0.001f);
        h_prev[i] = h_curr[i] * 0.99f;
        h_vel[i]  = coeff;
    }

    cpu_stencil(h_prev.data(), h_curr.data(), h_cpu_out.data(),
                h_vel.data(), N, N, N);

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

    launch_kernel_shmem(d_prev, d_curr, d_next, d_vel, N, N, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu_out.data(), d_next, bytes, cudaMemcpyDeviceToHost);

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

// ─── Test: Register Rotation Kernel vs CPU ──────────────────────────
static bool test_register_vs_cpu() {
    printf("[TEST] Register rotation kernel vs CPU reference... ");

    const int N = 64;
    Grid grid = { N, N, N, DEFAULT_DX, DEFAULT_DX, DEFAULT_DX, DEFAULT_DT, 1 };
    size_t total = grid.total_points();

    std::vector<float> h_prev(total, 0.0f);
    std::vector<float> h_curr(total, 0.0f);
    std::vector<float> h_vel(total, 0.0f);
    std::vector<float> h_cpu_out(total, 0.0f);
    std::vector<float> h_gpu_out(total, 0.0f);

    float coeff = DEFAULT_VELOCITY * DEFAULT_VELOCITY * DEFAULT_DT * DEFAULT_DT
                / (DEFAULT_DX * DEFAULT_DX);
    for (size_t i = 0; i < total; i++) {
        h_curr[i] = sinf((float)i * 0.001f);
        h_prev[i] = h_curr[i] * 0.99f;
        h_vel[i]  = coeff;
    }

    cpu_stencil(h_prev.data(), h_curr.data(), h_cpu_out.data(),
                h_vel.data(), N, N, N);

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

    launch_kernel_register(d_prev, d_curr, d_next, d_vel, N, N, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu_out.data(), d_next, bytes, cudaMemcpyDeviceToHost);

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

// ─── Test: Hybrid Kernel vs CPU ─────────────────────────────────────
static bool test_hybrid_vs_cpu() {
    printf("[TEST] Hybrid (shmem+register) kernel vs CPU reference... ");

    const int N = 64;
    Grid grid = { N, N, N, DEFAULT_DX, DEFAULT_DX, DEFAULT_DX, DEFAULT_DT, 1 };
    size_t total = grid.total_points();

    std::vector<float> h_prev(total, 0.0f);
    std::vector<float> h_curr(total, 0.0f);
    std::vector<float> h_vel(total, 0.0f);
    std::vector<float> h_cpu_out(total, 0.0f);
    std::vector<float> h_gpu_out(total, 0.0f);

    float coeff = DEFAULT_VELOCITY * DEFAULT_VELOCITY * DEFAULT_DT * DEFAULT_DT
                / (DEFAULT_DX * DEFAULT_DX);
    for (size_t i = 0; i < total; i++) {
        h_curr[i] = sinf((float)i * 0.001f);
        h_prev[i] = h_curr[i] * 0.99f;
        h_vel[i]  = coeff;
    }

    cpu_stencil(h_prev.data(), h_curr.data(), h_cpu_out.data(),
                h_vel.data(), N, N, N);

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

    launch_kernel_hybrid(d_prev, d_curr, d_next, d_vel, N, N, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu_out.data(), d_next, bytes, cudaMemcpyDeviceToHost);

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

// ─── Test: RTM Flat Reflector ───────────────────────────────────────
// Smoke test: verify the RTM pipeline runs end-to-end without errors,
// produces a non-zero image, and doesn't diverge (no NaN/Inf).
static bool test_rtm_flat_reflector() {
    printf("[TEST] RTM flat reflector imaging... ");

    const int N  = 48;
    const int reflector_z = 30;

    float dx = DEFAULT_DX;
    float dt = DEFAULT_DT;
    float freq = 15.0f;
    float v1 = 2000.0f, v2 = 2800.0f;

    int sponge_width = N / 6;
    int src_z = sponge_width + 2;

    float grid_speed = v1 * dt / dx;
    int round_trip = (int)(2.0f * (reflector_z - src_z) / grid_speed);
    int nt = (int)(round_trip * 1.8f);

    Grid grid = { N, N, N, dx, dx, dx, dt, nt };
    size_t total = grid.total_points();

    std::vector<float> h_vel(total);
    for (int z = 0; z < N; z++) {
        float v = (z < reflector_z) ? v1 : v2;
        float coeff = v * v * dt * dt / (dx * dx);
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                h_vel[z * N * N + y * N + x] = coeff;
    }

    std::vector<float> h_wavelet(nt);
    generate_ricker_wavelet(h_wavelet.data(), nt, dt, freq);
    Source src = { N/2, N/2, src_z, freq, h_wavelet.data() };

    int num_rec = N - 2 * sponge_width;
    std::vector<int> rx(num_rec), ry(num_rec), rz(num_rec);
    for (int i = 0; i < num_rec; i++) {
        rx[i] = sponge_width + i;
        ry[i] = N / 2;
        rz[i] = src_z;
    }
    std::vector<float> traces(num_rec * nt, 0.0f);
    ReceiverSet rec = { num_rec, rx.data(), ry.data(), rz.data(), traces.data() };

    DeviceState state;
    richter_init(grid, src, state);
    state.d_vel.copyFromHost(h_vel.data(), total);

    std::vector<float> h_image(total, 0.0f);
    richter_rtm(grid, src, rec, state, h_image.data(),
                KernelType::REGISTER_ROT, 50);

    // Check: non-zero and no NaN/Inf
    float max_amp = 0.0f;
    bool has_nan = false;
    for (size_t i = 0; i < total; i++) {
        float v = h_image[i];
        if (v != v) { has_nan = true; break; }  // NaN check
        if (v < 0) v = -v;
        if (v > max_amp) max_amp = v;
    }

    richter_cleanup(state);

    if (!has_nan && max_amp > 0.0f && max_amp < 1e30f) {
        printf("PASS (nt=%d, max_amp=%.2e, no NaN/Inf)\n", nt, max_amp);
        return true;
    } else {
        printf("FAIL (nt=%d, max_amp=%.2e, nan=%d)\n", nt, max_amp, has_nan);
        return false;
    }
}

// ─── Main ───────────────────────────────────────────────────────────
int main() {
    printf("═══════════════════════════════════════\n");
    printf("  Richter — Test Suite\n");
    printf("═══════════════════════════════════════\n\n");

    int passed = 0, total = 0;

    total++; if (test_ricker_wavelet())       passed++;
    total++; if (test_naive_vs_cpu())         passed++;
    total++; if (test_shmem_vs_cpu())         passed++;
    total++; if (test_register_vs_cpu())      passed++;
    total++; if (test_hybrid_vs_cpu())        passed++;
    total++; if (test_sponge_boundary())      passed++;
    total++; if (test_rtm_flat_reflector())   passed++;

    printf("\n─── Results: %d/%d passed ───\n", passed, total);
    return (passed == total) ? 0 : 1;
}
