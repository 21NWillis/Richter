// Richter: AVX2 + FMA + OpenMP CPU Stencil Kernel

#include "richter/kernels.h"
#include "richter/config.h"
#include <immintrin.h>
#include <cstring>

void launch_kernel_cpu_avx(const float* __restrict__ u_prev,
                           const float* __restrict__ u_curr,
                           float*       __restrict__ u_next,
                           const float* __restrict__ vel,
                           int nx, int ny, int nz)
{
    const int stride_y = nx;
    const int stride_z = nx * ny;
    const int R = STENCIL_RADIUS;

    const __m256 c0 = _mm256_set1_ps(FD_COEFF[0]);
    const __m256 c1 = _mm256_set1_ps(FD_COEFF[1]);
    const __m256 c2 = _mm256_set1_ps(FD_COEFF[2]);
    const __m256 c3 = _mm256_set1_ps(FD_COEFF[3]);
    const __m256 c4 = _mm256_set1_ps(FD_COEFF[4]);
    const __m256 three = _mm256_set1_ps(3.0f);
    const __m256 two   = _mm256_set1_ps(2.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = R; z < nz - R; z++) {
        for (int y = R; y < ny - R; y++) {
            int base = z * stride_z + y * stride_y;

            // AVX2 vectorized loop: 8 floats per iteration
            int x = R;
            for (; x <= nx - R - 8; x += 8) {
                int idx = base + x;

                __m256 center = _mm256_loadu_ps(&u_curr[idx]);

                // 3D Laplacian: 3 * c0 * center
                __m256 lap = _mm256_mul_ps(three, _mm256_mul_ps(c0, center));

                // r = 1
                __m256 sum1 = _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 1]),
                    _mm256_loadu_ps(&u_curr[idx - 1]));
                sum1 = _mm256_add_ps(sum1, _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + stride_y]),
                    _mm256_loadu_ps(&u_curr[idx - stride_y])));
                sum1 = _mm256_add_ps(sum1, _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + stride_z]),
                    _mm256_loadu_ps(&u_curr[idx - stride_z])));
                lap = _mm256_fmadd_ps(c1, sum1, lap);

                // r = 2
                __m256 sum2 = _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 2]),
                    _mm256_loadu_ps(&u_curr[idx - 2]));
                sum2 = _mm256_add_ps(sum2, _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 2 * stride_y]),
                    _mm256_loadu_ps(&u_curr[idx - 2 * stride_y])));
                sum2 = _mm256_add_ps(sum2, _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 2 * stride_z]),
                    _mm256_loadu_ps(&u_curr[idx - 2 * stride_z])));
                lap = _mm256_fmadd_ps(c2, sum2, lap);

                // r = 3
                __m256 sum3 = _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 3]),
                    _mm256_loadu_ps(&u_curr[idx - 3]));
                sum3 = _mm256_add_ps(sum3, _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 3 * stride_y]),
                    _mm256_loadu_ps(&u_curr[idx - 3 * stride_y])));
                sum3 = _mm256_add_ps(sum3, _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 3 * stride_z]),
                    _mm256_loadu_ps(&u_curr[idx - 3 * stride_z])));
                lap = _mm256_fmadd_ps(c3, sum3, lap);

                // r = 4
                __m256 sum4 = _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 4]),
                    _mm256_loadu_ps(&u_curr[idx - 4]));
                sum4 = _mm256_add_ps(sum4, _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 4 * stride_y]),
                    _mm256_loadu_ps(&u_curr[idx - 4 * stride_y])));
                sum4 = _mm256_add_ps(sum4, _mm256_add_ps(
                    _mm256_loadu_ps(&u_curr[idx + 4 * stride_z]),
                    _mm256_loadu_ps(&u_curr[idx - 4 * stride_z])));
                lap = _mm256_fmadd_ps(c4, sum4, lap);

                // u_next = 2 * u_curr - u_prev + vel * laplacian
                __m256 u_p = _mm256_loadu_ps(&u_prev[idx]);
                __m256 v   = _mm256_loadu_ps(&vel[idx]);
                __m256 result = _mm256_fmadd_ps(v, lap,
                                _mm256_sub_ps(_mm256_mul_ps(two, center), u_p));
                _mm256_storeu_ps(&u_next[idx], result);
            }

            // Scalar cleanup
            for (; x < nx - R; x++) {
                int idx = base + x;

                float laplacian = 3.0f * FD_COEFF[0] * u_curr[idx];
                for (int r = 1; r <= R; r++) {
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
