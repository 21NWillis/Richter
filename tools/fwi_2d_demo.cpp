// 2D FWI Demo
// End-to-end Full Waveform Inversion on 2D synthetic data.
//
// 1. Create a "true" velocity model (gradient + Gaussian lens anomaly)
// 2. Generate observed data by forward modeling with the true model
// 3. Start from a smooth initial guess (background gradient only)
// 4. Run 2D FWI to recover the velocity model
// 5. Output .npy files for visualization
//
// Usage: fwi_2d_demo [grid_size] [num_shots] [num_iterations]

#include "richter/fwi_2d.h"
#include "richter/fwi.h"       // reuse flat-array kernels
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

// Write a 1D float32 array as a NumPy .npy file
static void write_npy_1d(const char* filename, const float* data, int n) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", filename); return; }

    char header[128];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }", n);

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
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    printf("Wrote %s (%d float32)\n", filename, n);
}

int main(int argc, char** argv) {
    // Configuration
    int N = 256;           // 2D grid: much larger than 3D is feasible
    int num_shots = 8;
    int num_iters = 300;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_shots = atoi(argv[2]);
    if (argc > 3) num_iters = atoi(argv[3]);

    float dx = DEFAULT_DX;
    float dt = 0.0005f;

    int sponge = 20;
    int src_z = sponge + 2;
    int reflector_z = N / 2;

    // Auto-compute nt
    float grid_speed = DEFAULT_VELOCITY * dt / dx;
    int round_trip = (int)(2.0f * (reflector_z - src_z) / grid_speed);
    int nt = (int)(round_trip * 1.5f);

    printf("═══════════════════════════════════════\n");
    printf("  Richter — 2D FWI Demo\n");
    printf("  Grid: %dx%d  Steps: %d  Shots: %d\n", N, N, nt, num_shots);
    printf("═══════════════════════════════════════\n\n");

    Grid2D grid = { N, N, dx, dx, dt, nt };
    size_t total = grid.total_points();

    // ── TRUE VELOCITY MODEL ────────────────────────────────────────
    // Linear gradient + Gaussian low-velocity lens anomaly.
    float v_top = 2000.0f, v_bot = 3500.0f;
    int active_start = src_z + 5;
    int active_depth = N - active_start - sponge;
    int lens_cx = N / 2, lens_cz = active_start + active_depth / 3;
    float lens_radius = N / 6.0f;
    float lens_strength = -400.0f;

    std::vector<float> h_vel_true(total);
    for (int z = 0; z < N; z++) {
        float v_bg = v_top + (v_bot - v_top) * (float)z / (float)(N - 1);
        for (int x = 0; x < N; x++) {
            float dx2 = (float)(x - lens_cx) * (x - lens_cx);
            float dz2 = (float)(z - lens_cz) * (z - lens_cz);
            float r2 = (dx2 + dz2) / (lens_radius * lens_radius);
            float anomaly = lens_strength * expf(-r2);
            float v = v_bg + anomaly;
            h_vel_true[z * N + x] = v;
        }
    }
    printf("True model: gradient %.0f-%.0f m/s + %.0f m/s lens at (%d,%d) r=%.0f\n",
           v_top, v_bot, lens_strength, lens_cx, lens_cz, lens_radius);

    // ── INITIAL VELOCITY MODEL ─────────────────────────────────────
    std::vector<float> h_vel_initial(total);
    for (int z = 0; z < N; z++) {
        float v = v_top + (v_bot - v_top) * (float)z / (float)(N - 1);
        for (int x = 0; x < N; x++)
            h_vel_initial[z * N + x] = v;
    }
    printf("Initial model: linear gradient %.0f - %.0f m/s (no lens)\n", v_top, v_bot);

    // ── SOURCES ────────────────────────────────────────────────────
    std::vector<float> h_wavelet(nt);
    generate_ricker_wavelet(h_wavelet.data(), nt, dt, 15.0f);

    int active_x = N - 2 * sponge;
    std::vector<Source2D> sources(num_shots);
    for (int s = 0; s < num_shots; s++) {
        int sx = sponge + (active_x * (s + 1)) / (num_shots + 1);
        sources[s] = { sx, src_z, 15.0f, h_wavelet.data() };
        printf("Shot %d: source at (%d, %d)\n", s, sx, src_z);
    }

    // ── RECEIVERS ──────────────────────────────────────────────────
    int rec_nx = N - 2 * sponge;
    int num_rec = rec_nx;
    std::vector<int> rx(num_rec), rz(num_rec);
    for (int i = 0; i < num_rec; i++) {
        rx[i] = sponge + i;
        rz[i] = src_z;
    }
    ReceiverSet2D rec = { num_rec, rx.data(), rz.data() };
    printf("Receivers: %d along x at z=%d\n", num_rec, src_z);

    // ── GENERATE OBSERVED DATA ─────────────────────────────────────
    printf("\nGenerating observed data with true model...\n");

    DeviceState2D obs_state;
    obs_state.d_u_prev = CudaBuffer<float>(total);
    obs_state.d_u_curr = CudaBuffer<float>(total);
    obs_state.d_u_next = CudaBuffer<float>(total);
    obs_state.d_vel    = CudaBuffer<float>(total);
    obs_state.d_wavelet = CudaBuffer<float>(nt);

    {
        std::vector<float> h_coeff(total);
        float dt2_dx2 = dt * dt / (dx * dx);
        for (size_t i = 0; i < total; i++) {
            h_coeff[i] = h_vel_true[i] * h_vel_true[i] * dt2_dx2;
        }
        obs_state.d_vel.copyFromHost(h_coeff.data(), total);
    }
    obs_state.d_wavelet.copyFromHost(h_wavelet.data(), nt);

    CudaBuffer<int> d_rx(num_rec), d_rz(num_rec);
    d_rx.copyFromHost(rx.data(), num_rec);
    d_rz.copyFromHost(rz.data(), num_rec);

    CudaBuffer<float> d_obs_syn((size_t)num_rec * nt);

    std::vector<std::vector<float>> h_obs_all(num_shots);
    std::vector<const float*> h_obs_ptrs(num_shots);

    for (int s = 0; s < num_shots; s++) {
        // Forward model with true velocity
        int sponge_w = (grid.nx < 64) ? grid.nx / 6 : 20;
        obs_state.d_u_prev.zero();
        obs_state.d_u_curr.zero();
        obs_state.d_u_next.zero();
        d_obs_syn.zero();

        for (int t = 0; t < nt; t++) {
            inject_source_2d(obs_state.d_u_curr.data(),
                             sources[s].sx, sources[s].sz,
                             sources[s].wavelet[t], grid.nx, grid.nz);

            launch_stencil_2d(obs_state.d_u_prev.data(), obs_state.d_u_curr.data(),
                              obs_state.d_u_next.data(), obs_state.d_vel.data(),
                              grid.nx, grid.nz);
            apply_sponge_2d(obs_state.d_u_next.data(), grid.nx, grid.nz,
                            sponge_w, 0.015f);

            obs_state.d_u_prev.swap(obs_state.d_u_curr);
            obs_state.d_u_curr.swap(obs_state.d_u_next);

            record_receivers_2d(obs_state.d_u_curr.data(), d_obs_syn.data(),
                                d_rx.data(), d_rz.data(),
                                num_rec, grid.nx, t, nt);
        }
        cudaDeviceSynchronize();

        h_obs_all[s].resize((size_t)num_rec * nt);
        cudaMemcpy(h_obs_all[s].data(), d_obs_syn.data(),
                   (size_t)num_rec * nt * sizeof(float), cudaMemcpyDeviceToHost);
        h_obs_ptrs[s] = h_obs_all[s].data();

        float max_amp = 0.0f;
        for (size_t i = 0; i < (size_t)num_rec * nt; i++) {
            float v = fabsf(h_obs_all[s][i]);
            if (v > max_amp) max_amp = v;
        }
        printf("  Shot %d observed: max amplitude = %.6e\n", s, max_amp);
    }

    // Cleanup observation state
    obs_state.d_u_prev = CudaBuffer<float>();
    obs_state.d_u_curr = CudaBuffer<float>();
    obs_state.d_u_next = CudaBuffer<float>();
    obs_state.d_vel    = CudaBuffer<float>();
    obs_state.d_wavelet = CudaBuffer<float>();

    printf("Observed data generated.\n");

    // ── FWI CONFIGURATION ──────────────────────────────────────────
    // Single frequency stage at the source frequency (15 Hz).
    // Multi-scale requires bandpass filtering the observed data to each
    // stage's frequency band, which is not yet implemented.
    float freq_stages[] = { 15.0f };
    int iters_per_stage[] = { num_iters };

    FWIConfig2D config;
    config.max_iterations = num_iters;
    config.initial_step_size = 0.5f;     // CG can handle larger steps
    config.step_size_reduction = 0.5f;
    config.max_line_search_steps = 20;
    config.misfit_tolerance = 1e-10f;    // effectively disable early convergence
    config.gradient_smooth_sigma = 2.0f;
    config.v_min = 1500.0f;
    config.v_max = 4000.0f;
    config.water_depth = src_z;
    config.checkpoint_interval = 50;
    config.mute_direct_v = v_top;        // mute using surface velocity
    config.mute_taper_samples = 30.0f;   // smooth taper transition
    config.depth_scale_power = 3.0f;     // aggressive depth boost for lens recovery
    config.num_frequency_stages = 1;
    config.frequency_stages = freq_stages;
    config.iterations_per_stage = iters_per_stage;

    // ── RUN FWI ────────────────────────────────────────────────────
    std::vector<float> h_vel_output(total);
    FWIResult2D fwi_result;

    printf("\nRunning 2D FWI...\n");
    richter_fwi_2d(grid, sources.data(), num_shots, rec,
                   h_obs_ptrs.data(), h_vel_initial.data(),
                   h_vel_output.data(), config, &fwi_result);

    // ── OUTPUT ──────────────────────────────────────────────────────
    printf("\nWriting outputs...\n");

    write_npy("fwi_2d_true.npy", h_vel_true.data(), N, N);
    write_npy("fwi_2d_initial.npy", h_vel_initial.data(), N, N);
    write_npy("fwi_2d_velocity.npy", h_vel_output.data(), N, N);

    // Difference: inverted - initial (shows what FWI actually changed)
    {
        std::vector<float> diff(total);
        for (size_t i = 0; i < total; i++)
            diff[i] = h_vel_output[i] - h_vel_initial[i];
        write_npy("fwi_2d_delta.npy", diff.data(), N, N);
    }

    // Misfit curve
    if (!fwi_result.misfit_history.empty()) {
        write_npy_1d("fwi_2d_misfit.npy", fwi_result.misfit_history.data(),
                      (int)fwi_result.misfit_history.size());
    }

    // Stats
    float min_true = 1e30f, max_true = -1e30f;
    float min_inv = 1e30f, max_inv = -1e30f;
    float max_err = 0.0f, sum_err = 0.0f;
    for (size_t i = 0; i < total; i++) {
        if (h_vel_true[i] < min_true) min_true = h_vel_true[i];
        if (h_vel_true[i] > max_true) max_true = h_vel_true[i];
        if (h_vel_output[i] < min_inv) min_inv = h_vel_output[i];
        if (h_vel_output[i] > max_inv) max_inv = h_vel_output[i];
        float err = fabsf(h_vel_true[i] - h_vel_output[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
    }
    printf("\nTrue velocity:     [%.0f, %.0f] m/s\n", min_true, max_true);
    printf("Inverted velocity: [%.0f, %.0f] m/s\n", min_inv, max_inv);
    printf("Max error: %.1f m/s   Mean error: %.1f m/s\n",
           max_err, sum_err / total);
    printf("Iterations: %d   Final misfit: %.6e\n",
           fwi_result.iterations_completed,
           fwi_result.misfit_history.empty() ? 0.0f : fwi_result.misfit_history.back());

    printf("\nRun: python tools/view_fwi_2d.py\n");
    return 0;
}
