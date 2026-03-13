// 2D FWI Demo — Multi-Scale Frequency Continuation
// End-to-end Full Waveform Inversion on 2D synthetic data.
//
// 1. Create a "true" velocity model (gradient + Gaussian lens anomaly)
// 2. For each frequency stage (low → high):
//    a. Generate observed data at that frequency
//    b. Run FWI, using output from previous stage as starting model
// 3. Output .npy files for visualization
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

// Generate observed data for all shots at a given frequency
static void generate_observed_data(
    const Grid2D& grid, int num_shots,
    const std::vector<int>& shot_sx, int src_z,
    float freq,
    const CudaBuffer<float>& d_vel_true,
    CudaBuffer<int>& d_rx, CudaBuffer<int>& d_rz, int num_rec,
    std::vector<std::vector<float>>& h_obs_all)
{
    size_t total = grid.total_points();
    size_t trace_size = (size_t)num_rec * grid.nt;
    int sponge_w = (grid.nx < 64) ? grid.nx / 6 : 20;

    std::vector<float> h_wavelet(grid.nt);
    generate_ricker_wavelet(h_wavelet.data(), grid.nt, grid.dt, freq);

    DeviceState2D obs_state;
    obs_state.d_u_prev = CudaBuffer<float>(total);
    obs_state.d_u_curr = CudaBuffer<float>(total);
    obs_state.d_u_next = CudaBuffer<float>(total);
    obs_state.d_vel    = CudaBuffer<float>(total);
    obs_state.d_wavelet = CudaBuffer<float>(grid.nt);

    cudaMemcpy(obs_state.d_vel.data(), d_vel_true.data(),
               total * sizeof(float), cudaMemcpyDeviceToDevice);
    obs_state.d_wavelet.copyFromHost(h_wavelet.data(), grid.nt);

    CudaBuffer<float> d_obs_syn(trace_size);

    h_obs_all.resize(num_shots);

    for (int s = 0; s < num_shots; s++) {
        obs_state.d_u_prev.zero();
        obs_state.d_u_curr.zero();
        obs_state.d_u_next.zero();
        d_obs_syn.zero();

        for (int t = 0; t < grid.nt; t++) {
            inject_source_2d(obs_state.d_u_curr.data(),
                             shot_sx[s], src_z,
                             h_wavelet[t], grid.nx, grid.nz);

            launch_stencil_2d(obs_state.d_u_prev.data(), obs_state.d_u_curr.data(),
                              obs_state.d_u_next.data(), obs_state.d_vel.data(),
                              grid.nx, grid.nz);
            apply_sponge_2d(obs_state.d_u_next.data(), grid.nx, grid.nz,
                            sponge_w, 0.015f);

            obs_state.d_u_prev.swap(obs_state.d_u_curr);
            obs_state.d_u_curr.swap(obs_state.d_u_next);

            record_receivers_2d(obs_state.d_u_curr.data(), d_obs_syn.data(),
                                d_rx.data(), d_rz.data(),
                                num_rec, grid.nx, t, grid.nt);
        }
        cudaDeviceSynchronize();

        h_obs_all[s].resize(trace_size);
        cudaMemcpy(h_obs_all[s].data(), d_obs_syn.data(),
                   trace_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

int main(int argc, char** argv) {
    // Configuration
    int N = 256;
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
    printf("  Richter — 2D FWI Demo (Multi-Scale)\n");
    printf("  Grid: %dx%d  Steps: %d  Shots: %d\n", N, N, nt, num_shots);
    printf("═══════════════════════════════════════\n\n");

    Grid2D grid = { N, N, dx, dx, dt, nt };
    size_t total = grid.total_points();

    // ── TRUE VELOCITY MODEL ────────────────────────────────────────
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

    // ── SOURCE POSITIONS ─────────────────────────────────────────
    int active_x = N - 2 * sponge;
    std::vector<int> shot_sx(num_shots);
    for (int s = 0; s < num_shots; s++) {
        shot_sx[s] = sponge + (active_x * (s + 1)) / (num_shots + 1);
        printf("Shot %d: source at (%d, %d)\n", s, shot_sx[s], src_z);
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

    // Upload true velocity as coefficient to device (for observed data generation)
    CudaBuffer<float> d_vel_true(total);
    {
        std::vector<float> h_coeff(total);
        float dt2_dx2 = dt * dt / (dx * dx);
        for (size_t i = 0; i < total; i++) {
            h_coeff[i] = h_vel_true[i] * h_vel_true[i] * dt2_dx2;
        }
        d_vel_true.copyFromHost(h_coeff.data(), total);
    }

    CudaBuffer<int> d_rx(num_rec), d_rz(num_rec);
    d_rx.copyFromHost(rx.data(), num_rec);
    d_rz.copyFromHost(rz.data(), num_rec);

    // ── MULTI-SCALE FREQUENCY CONTINUATION ────────────────────────
    // Gradual frequency progression: 3→5→7→10→12→15 Hz
    // L-BFGS should sustain convergence longer than CG (~22 iter wall).
    // Smaller frequency jumps give better continuity of model updates.
    const int NUM_STAGES = 6;
    float stage_freqs[NUM_STAGES] = { 3.0f, 5.0f, 7.0f, 10.0f, 12.0f, 15.0f };
    int stage_iters[NUM_STAGES];
    {
        // Distribute iterations: more at low freq (bigger model updates),
        // taper off at high freq (fine detail, less budget needed)
        // Default total ~100 iterations
        stage_iters[0] = std::min(25, num_iters);
        stage_iters[1] = std::min(20, std::max(5, num_iters - 25));
        stage_iters[2] = std::min(15, std::max(5, num_iters - 45));
        stage_iters[3] = std::min(15, std::max(5, num_iters - 60));
        stage_iters[4] = std::min(10, std::max(5, num_iters - 75));
        stage_iters[5] = std::max(5, num_iters - 85);
    }

    printf("\n═══ Multi-Scale Schedule ═══\n");
    for (int s = 0; s < NUM_STAGES; s++) {
        printf("  Stage %d: f=%.0f Hz, %d iterations\n",
               s + 1, stage_freqs[s], stage_iters[s]);
    }
    printf("════════════════════════════\n");

    // Current model starts as initial guess
    std::vector<float> h_vel_current(h_vel_initial);
    std::vector<float> h_vel_output(total);
    std::vector<float> all_misfits;

    for (int stage = 0; stage < NUM_STAGES; stage++) {
        float freq = stage_freqs[stage];
        int iters = stage_iters[stage];

        printf("\n╔════════════════════════════════════════╗\n");
        printf("║ Stage %d/%d: f=%.0f Hz, %d iterations     ║\n",
               stage + 1, NUM_STAGES, freq, iters);
        printf("╚════════════════════════════════════════╝\n");

        // Generate observed data at this frequency
        printf("Generating observed data at %.0f Hz...\n", freq);
        std::vector<std::vector<float>> h_obs_all;
        generate_observed_data(grid, num_shots, shot_sx, src_z, freq,
                               d_vel_true, d_rx, d_rz, num_rec, h_obs_all);

        std::vector<const float*> h_obs_ptrs(num_shots);
        for (int s = 0; s < num_shots; s++)
            h_obs_ptrs[s] = h_obs_all[s].data();

        // Build sources at this frequency
        std::vector<float> h_wavelet(nt);
        generate_ricker_wavelet(h_wavelet.data(), nt, dt, freq);

        std::vector<Source2D> sources(num_shots);
        for (int s = 0; s < num_shots; s++) {
            sources[s] = { shot_sx[s], src_z, freq, h_wavelet.data() };
        }

        // Configure this stage
        float stage_freq_arr[] = { freq };
        int stage_iter_arr[] = { iters };

        FWIConfig2D config;
        config.max_iterations = iters;
        config.initial_step_size = 0.5f;
        config.step_size_reduction = 0.5f;
        config.max_line_search_steps = 20;
        config.misfit_tolerance = 1e-10f;
        // Scale smoothing with wavelength: wider at low freq, tighter at high
        config.gradient_smooth_sigma = std::max(2.0f, 8.0f * (5.0f / freq));
        config.v_min = 1500.0f;
        config.v_max = 4000.0f;
        config.water_depth = src_z;
        config.checkpoint_interval = 50;
        config.mute_direct_v = v_top;
        config.mute_taper_samples = 30.0f;
        config.depth_scale_power = 2.0f;     // z^2 geometric spreading compensation
        config.layer_strip_iter = 0;   // no layer stripping in multi-scale
        config.layer_strip_depth = 0;
        config.layer_strip_taper = 0;
        config.num_frequency_stages = 1;
        config.frequency_stages = stage_freq_arr;
        config.iterations_per_stage = stage_iter_arr;

        FWIResult2D stage_result;

        richter_fwi_2d(grid, sources.data(), num_shots, rec,
                       h_obs_ptrs.data(), h_vel_current.data(),
                       h_vel_output.data(), config, &stage_result);

        // Stage output becomes next stage input
        h_vel_current = h_vel_output;

        // Accumulate misfit history
        for (float m : stage_result.misfit_history) {
            all_misfits.push_back(m);
        }

        // Stage stats
        float stage_max_err = 0.0f, stage_sum_err = 0.0f;
        for (size_t i = 0; i < total; i++) {
            float err = fabsf(h_vel_true[i] - h_vel_output[i]);
            if (err > stage_max_err) stage_max_err = err;
            stage_sum_err += err;
        }
        printf("\n[Stage %d] Max error: %.1f m/s  Mean error: %.1f m/s\n",
               stage + 1, stage_max_err, stage_sum_err / total);
    }

    // ── OUTPUT ──────────────────────────────────────────────────────
    printf("\nWriting outputs...\n");

    write_npy("fwi_2d_true.npy", h_vel_true.data(), N, N);
    write_npy("fwi_2d_initial.npy", h_vel_initial.data(), N, N);
    write_npy("fwi_2d_velocity.npy", h_vel_output.data(), N, N);

    // Difference: inverted - initial
    {
        std::vector<float> diff(total);
        for (size_t i = 0; i < total; i++)
            diff[i] = h_vel_output[i] - h_vel_initial[i];
        write_npy("fwi_2d_delta.npy", diff.data(), N, N);
    }

    // Combined misfit curve
    if (!all_misfits.empty()) {
        write_npy_1d("fwi_2d_misfit.npy", all_misfits.data(),
                      (int)all_misfits.size());
    }

    // Final stats
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
    printf("\n═══════════════════════════════════════\n");
    printf("  FINAL RESULTS\n");
    printf("═══════════════════════════════════════\n");
    printf("True velocity:     [%.0f, %.0f] m/s\n", min_true, max_true);
    printf("Inverted velocity: [%.0f, %.0f] m/s\n", min_inv, max_inv);
    printf("Max error: %.1f m/s   Mean error: %.1f m/s\n",
           max_err, sum_err / total);
    printf("Total iterations: %d   Final misfit: %.6e\n",
           (int)all_misfits.size(),
           all_misfits.empty() ? 0.0f : all_misfits.back());

    printf("\nRun: python tools/view_fwi_2d.py\n");
    return 0;
}
