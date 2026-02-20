// ─── Richter: Benchmarking Harness ──────────────────────────────────
// Measures GPts/s and effective bandwidth for each kernel implementation.

#include "richter/model.h"
#include "richter/kernels.h"
#include "richter/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "richter/cuda_buffer.h"

struct BenchResult {
    const char* name;
    double gpts_per_sec;
    double effective_bw_gb;
    double pct_peak;
};

static BenchResult run_benchmark(const char* name,
                                  void (*launcher)(const float*, const float*,
                                                   float*, const float*,
                                                   int, int, int),
                                  int nx, int ny, int nz,
                                  int warmup, int iters,
                                  double peak_bw_gb)
{
    size_t total = (size_t)nx * ny * nz;
    size_t bytes = total * sizeof(float);

    CudaBuffer<float> d_prev(total);
    CudaBuffer<float> d_curr(total);
    CudaBuffer<float> d_next(total);
    CudaBuffer<float> d_vel(total);

    d_prev.zero();
    d_curr.zero();
    d_next.zero();
    d_vel.zero();

    // Warmup
    for (int i = 0; i < warmup; i++) {
        launcher(d_prev.data(), d_curr.data(), d_next.data(), d_vel.data(), nx, ny, nz);
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        launcher(d_prev.data(), d_curr.data(), d_next.data(), d_vel.data(), nx, ny, nz);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double elapsed_s = elapsed_ms / 1000.0;

    // Metrics
    // Reads: u_prev, u_curr, vel (3 arrays). Writes: u_next (1 array).
    // Each is `total * sizeof(float)` bytes.
    double bytes_transferred = 4.0 * total * sizeof(float) * iters;
    double eff_bw = bytes_transferred / elapsed_s / 1e9;
    double gpts = (double)total * iters / elapsed_s / 1e9;
    double pct = eff_bw / peak_bw_gb * 100.0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Destructors handle cudaFree

    return { name, gpts, eff_bw, pct };
}

int main(int argc, char** argv) {
    // Grid size (default 256^3, override with argv)
    int N = 256;
    if (argc > 1) N = atoi(argv[1]);

    // Adjust peak BW for your GPU (default: RTX 3070 = 448 GB/s)
    double peak_bw = 448.0;
    if (argc > 2) peak_bw = atof(argv[2]);

    int warmup = 5;
    int iters  = 20;

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Richter — Benchmark    Grid: %d³    Peak BW: %.0f GB/s\n", N, peak_bw);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    printf("%-20s %10s %12s %10s\n", "Kernel", "GPts/s", "Eff BW", "% Peak");
    printf("────────────────────────────────────────────────────────────\n");

    auto result = run_benchmark("Naive", launch_kernel_naive,
                                 N, N, N, warmup, iters, peak_bw);
    printf("%-20s %10.3f %10.1f GB/s %8.1f%%\n",
           result.name, result.gpts_per_sec, result.effective_bw_gb, result.pct_peak);

    auto shmem_result = run_benchmark("Shared Memory", launch_kernel_shmem,
                                 N, N, N, warmup, iters, peak_bw);
    printf("%-20s %10.3f %10.1f GB/s %8.1f%%\n",
           shmem_result.name, shmem_result.gpts_per_sec, shmem_result.effective_bw_gb, shmem_result.pct_peak);

    auto reg_result = run_benchmark("Register Rotation", launch_kernel_register,
                                 N, N, N, warmup, iters, peak_bw);
    printf("%-20s %10.3f %10.1f GB/s %8.1f%%\n",
           reg_result.name, reg_result.gpts_per_sec, reg_result.effective_bw_gb, reg_result.pct_peak);

    auto hybrid_result = run_benchmark("Hybrid (Shmem+Reg)", launch_kernel_hybrid,
                                 N, N, N, warmup, iters, peak_bw);
    printf("%-20s %10.3f %10.1f GB/s %8.1f%%\n",
           hybrid_result.name, hybrid_result.gpts_per_sec, hybrid_result.effective_bw_gb, hybrid_result.pct_peak);

    printf("\n");
    return 0;
}
