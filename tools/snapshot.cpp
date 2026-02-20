// ─── Richter: Snapshot Tool ─────────────────────────────────────────
// Runs a short simulation and dumps a 2D XY slice as a .npy file
// for visualization with Python / matplotlib.

#include "richter/model.h"
#include "richter/kernels.h"
#include "richter/wavelet.h"
#include "richter/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ─── Write a 2D float32 array as a NumPy .npy file ─────────────────
static void write_npy(const char* filename, const float* data, int rows, int cols) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", filename); return; }

    // .npy format: magic, version, header, data
    // Header is a Python dict string describing dtype, order, shape
    char header[128];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", rows, cols);

    // Pad header to align data to 64 bytes (magic=6 + version=2 + header_len=2 = 10)
    int total_header = 10 + hlen + 1;  // +1 for trailing newline
    int padding = 64 - (total_header % 64);
    if (padding == 64) padding = 0;
    int padded_len = hlen + padding + 1;

    // Magic number
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    fwrite(magic, 1, 6, f);

    // Version 1.0
    unsigned char version[] = {1, 0};
    fwrite(version, 1, 2, f);

    // Header length (little-endian uint16)
    unsigned short hl = (unsigned short)padded_len;
    fwrite(&hl, 2, 1, f);

    // Header string
    fwrite(header, 1, hlen, f);

    // Padding spaces + newline
    for (int i = 0; i < padding; i++) fputc(' ', f);
    fputc('\n', f);

    // Raw float32 data
    fwrite(data, sizeof(float), (size_t)rows * cols, f);

    fclose(f);
    printf("Wrote %s (%d x %d float32)\n", filename, rows, cols);
}

int main(int argc, char** argv) {
    // ─── Configuration ──────────────────────────────────────────────
    int N  = 256;          // Grid size (N^3)
    int nt = 200;          // Number of timesteps

    if (argc > 1) N  = atoi(argv[1]);
    if (argc > 2) nt = atoi(argv[2]);

    float dx = DEFAULT_DX;
    float dt = DEFAULT_DT;
    float freq = 15.0f;    // Ricker wavelet peak frequency

    printf("═══════════════════════════════════════\n");
    printf("  Richter — Snapshot Tool\n");
    printf("  Grid: %d³   Steps: %d\n", N, nt);
    printf("═══════════════════════════════════════\n\n");

    // ─── Setup ──────────────────────────────────────────────────────
    Grid grid = { N, N, N, dx, dx, dx, dt, nt };

    // Precompute velocity model: coeff = v² * dt² / dx²
    float coeff = DEFAULT_VELOCITY * DEFAULT_VELOCITY * dt * dt / (dx * dx);
    std::vector<float> h_vel(grid.total_points(), coeff);

    // Generate Ricker wavelet
    std::vector<float> h_wavelet(nt);
    generate_ricker_wavelet(h_wavelet.data(), nt, dt, freq);

    // Source at grid center
    Source src = { N/2, N/2, N/2, freq, h_wavelet.data() };

    // Init device state
    DeviceState state;
    richter_init(grid, src, state);

    // Upload velocity model
    state.d_vel.copyFromHost(h_vel.data(), grid.total_points());

    // ─── Run simulation ─────────────────────────────────────────────
    printf("Running %d timesteps...\n", nt);
    richter_forward(grid, src, state, KernelType::REGISTER_ROT);
    cudaDeviceSynchronize();
    printf("Done.\n\n");

    // ─── Extract middle XY slice ────────────────────────────────────
    std::vector<float> h_field(grid.total_points());
    richter_snapshot(grid, state, h_field.data());

    int z_slice = N / 2;
    std::vector<float> slice(N * N);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            slice[y * N + x] = h_field[z_slice * N * N + y * N + x];
        }
    }

    // ─── Write to .npy ──────────────────────────────────────────────
    write_npy("slice_xy.npy", slice.data(), N, N);

    // Print some stats
    float minv =  1e30f, maxv = -1e30f;
    for (int i = 0; i < N * N; i++) {
        if (slice[i] < minv) minv = slice[i];
        if (slice[i] > maxv) maxv = slice[i];
    }
    printf("Slice stats: min=%.6e  max=%.6e\n", minv, maxv);

    // ─── Cleanup ────────────────────────────────────────────────────
    richter_cleanup(state);
    printf("\nRun: python tools/view_slice.py\n");
    return 0;
}
