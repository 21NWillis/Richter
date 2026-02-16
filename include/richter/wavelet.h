#pragma once

/// Generate a Ricker wavelet (Mexican-hat) for a given peak frequency.
/// @param wavelet  Output buffer (must be pre-allocated, length = nt)
/// @param nt       Number of time samples
/// @param dt       Time step in seconds
/// @param peak_freq Peak frequency in Hz
void generate_ricker_wavelet(float* wavelet, int nt, float dt, float peak_freq);
