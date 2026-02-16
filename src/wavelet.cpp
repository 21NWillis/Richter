// ─── Richter: Ricker Wavelet Generator ──────────────────────────────
// The Ricker wavelet (Mexican-hat) is the standard source signature
// in seismic modeling. It's the second derivative of a Gaussian.
//
// Formula: w(t) = (1 - 2π²f²t²) * exp(-π²f²t²)

#include "richter/wavelet.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void generate_ricker_wavelet(float* wavelet, int nt, float dt, float peak_freq) {
    // Shift so the wavelet peak is at t = 1.0/peak_freq (avoids initial transient)
    float t_shift = 1.0f / peak_freq;

    for (int i = 0; i < nt; i++) {
        float t = i * dt - t_shift;
        float pi_f_t = (float)M_PI * peak_freq * t;
        float pft2 = pi_f_t * pi_f_t;
        wavelet[i] = (1.0f - 2.0f * pft2) * expf(-pft2);
    }
}
