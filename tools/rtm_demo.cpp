// RTM Demo
// End-to-end Reverse Time Migration on synthetic data.
//
// Velocity model: constant 2000 m/s with a horizontal reflector at
// z = 3*nz/4 where velocity jumps to 3000 m/s.
// Source: center of the surface (z = sponge margin).
// Receivers: line along the surface at z = sponge margin.
// Output: RTM image as a .npy file (XZ slice through center y).

#include "richter/model.h"
#include "richter/rtm.h"
#include "richter/kernels.h"
#include "richter/wavelet.h"
#include "richter/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

// Write a 2D float32 array as a NumPy .npy file
static void write_npy(const char* filename, const float* data, int rows, int cols) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", filename); return; }

    char header[128];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", rows, cols);

    int total_header = 10 + hlen + 1;
    int padding = 64 - (total_header % 64);
    if (padding == 64) padding = 0;
    int padded_len = hlen + padding + 1;

    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    fwrite(magic, 1, 6, f);
    unsigned char version[] = {1, 0};
    fwrite(version, 1, 2, f);
    unsigned short hl = (unsigned short)padded_len;
    fwrite(&hl, 2, 1, f);
    fwrite(header, 1, hlen, f);
    for (int i = 0; i < padding; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), (size_t)rows * cols, f);
    fclose(f);
    printf("Wrote %s (%d x %d float32)\n", filename, rows, cols);
}

int main(int argc, char** argv) {
    // Configuration
    int N  = 256;           // Grid size (N^3)
    int cp_interval = 50;

    if (argc > 1) N  = atoi(argv[1]);
    if (argc > 3) cp_interval = atoi(argv[3]);

    float dx   = DEFAULT_DX;
    float dt   = 0.0005f;   // half of default for CFL stability with v2=3000
    float freq = 15.0f;

    int sponge = 20;  // sponge boundary width

    // Auto-compute nt: wave must travel source→reflector→surface round trip
    // Use v1 for travel time (wave travels through upper layer for most of the path)
    int reflector_z = N / 2;   // center of grid — well inside active zone
    int src_z = sponge + 2;
    float grid_speed = DEFAULT_VELOCITY * dt / dx;  // grid points per step
    int round_trip = (int)(2.0f * (reflector_z - src_z) / grid_speed);
    int nt = (int)(round_trip * 1.5f);  // 50% margin
    if (argc > 2 && atoi(argv[2]) > 0) nt = atoi(argv[2]);  // 0 = auto

    printf("═══════════════════════════════════════\n");
    printf("  Richter — RTM Demo\n");
    printf("  Grid: %d³   Steps: %d   CP: %d   Shots: %d\n", N, nt, cp_interval,
           (argc > 4) ? atoi(argv[4]) : 5);
    printf("═══════════════════════════════════════\n\n");

    // Velocity Model
    // Two-layer model: v1 = 2000 m/s above reflector, v2 = 3000 m/s below
    float v1 = 2000.0f, v2 = 3000.0f;

    Grid grid = { N, N, N, dx, dx, dx, dt, nt };
    size_t total = grid.total_points();

    std::vector<float> h_vel(total);
    for (int z = 0; z < N; z++) {
        float v = (z < reflector_z) ? v1 : v2;
        float coeff = v * v * dt * dt / (dx * dx);
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                h_vel[z * N * N + y * N + x] = coeff;
            }
        }
    }
    printf("Velocity model: %.0f m/s above z=%d, %.0f m/s below\n",
           v1, reflector_z, v2);

    // Sources — line of shots along x-axis for multi-shot stacking
    std::vector<float> h_wavelet(nt);
    generate_ricker_wavelet(h_wavelet.data(), nt, dt, freq);

    int num_shots = 5;
    if (argc > 4) num_shots = atoi(argv[4]);

    int active_x = N - 2 * sponge;  // usable grid width inside sponge
    std::vector<Source> sources(num_shots);
    for (int s = 0; s < num_shots; s++) {
        int sx = sponge + (active_x * (s + 1)) / (num_shots + 1);
        sources[s] = { sx, N/2, src_z, freq, h_wavelet.data() };
        printf("Shot %d: source at (%d, %d, %d)\n", s, sources[s].sx, sources[s].sy, sources[s].sz);
    }

    // Receivers
    // 2D grid of receivers at the surface for full aperture coverage
    int rec_nx = N - 2 * sponge;
    int rec_ny = N - 2 * sponge;
    int num_rec = rec_nx * rec_ny;
    std::vector<int> rx(num_rec), ry(num_rec), rz(num_rec);
    int idx = 0;
    for (int iy = 0; iy < rec_ny; iy++) {
        for (int ix = 0; ix < rec_nx; ix++) {
            rx[idx] = sponge + ix;
            ry[idx] = sponge + iy;
            rz[idx] = src_z;
            idx++;
        }
    }

    std::vector<float> traces(num_rec * nt, 0.0f);
    ReceiverSet rec = { num_rec, rx.data(), ry.data(), rz.data(), traces.data() };
    printf("Receivers: %d x %d grid at z=%d (%d total)\n", rec_nx, rec_ny, src_z, num_rec);

    // Init device state (use first source for wavelet upload)
    DeviceState state;
    richter_init(grid, sources[0], state);
    state.d_vel.copyFromHost(h_vel.data(), total);

    // Background velocity (homogeneous, v1 only) for direct-arrival subtraction
    float bg_coeff = v1 * v1 * dt * dt / (dx * dx);
    std::vector<float> h_vel_bg(total, bg_coeff);

    // Run multi-shot RTM
    std::vector<float> h_image(total, 0.0f);
    printf("\nRunning multi-shot RTM (%d shots)...\n", num_shots);
    richter_rtm_multishot(grid, sources.data(), num_shots, rec, state,
                          h_image.data(), KernelType::REGISTER_ROT,
                          cp_interval, h_vel_bg.data());
    printf("RTM complete.\n\n");

    // Extract XZ slice at y = N/2
    int y_slice = N / 2;
    std::vector<float> slice(N * N);  // rows=nz, cols=nx
    for (int z = 0; z < N; z++) {
        for (int x = 0; x < N; x++) {
            slice[z * N + x] = h_image[z * N * N + y_slice * N + x];
        }
    }

    write_npy("rtm_image.npy", slice.data(), N, N);

    // Stats
    float minv = 1e30f, maxv = -1e30f;
    for (int i = 0; i < N * N; i++) {
        if (slice[i] < minv) minv = slice[i];
        if (slice[i] > maxv) maxv = slice[i];
    }
    printf("Image stats: min=%.6e  max=%.6e\n", minv, maxv);
    printf("Reflector expected at z=%d\n", reflector_z);

    // Cleanup
    richter_cleanup(state);
    printf("\nRun: python tools/view_rtm.py\n");
    return 0;
}
