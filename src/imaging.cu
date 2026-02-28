// ─── Richter: Imaging Condition Kernel ──────────────────────────────
// Cross-correlation imaging condition: image[i] += src[i] * rcv[i]
// Applied at every timestep during simultaneous forward re-propagation
// and backward propagation.
//
// Source illumination compensation: illum[i] += src[i]²
// Applied after all timesteps to normalize the image and suppress
// the direct-arrival artifact near the source.

#include "richter/rtm.h"
#include <cuda_runtime.h>

__global__ void cross_correlate_kernel(const float* __restrict__ src_field,
                                       const float* __restrict__ rcv_field,
                                       float* __restrict__ image,
                                       int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        image[idx] += src_field[idx] * rcv_field[idx];
    }
}

void apply_imaging_condition(const float* d_src_field,
                             const float* d_rcv_field,
                             float* d_image, size_t n)
{
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    cross_correlate_kernel<<<blocks, threads>>>(d_src_field, d_rcv_field,
                                                d_image, (int)n);
}

// Source Illumination Accumulation
__global__ void accumulate_illumination_kernel(const float* __restrict__ src_field,
                                                float* __restrict__ illum,
                                                int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = src_field[idx];
        illum[idx] += s * s;
    }
}

void accumulate_source_illumination(const float* d_src_field,
                                     float* d_illum, size_t n)
{
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    accumulate_illumination_kernel<<<blocks, threads>>>(d_src_field, d_illum, (int)n);
}

// Source Illumination Normalization
__global__ void normalize_by_illumination_kernel(float* __restrict__ image,
                                                  const float* __restrict__ illum,
                                                  int n, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float denom = illum[idx] + epsilon;
        image[idx] /= denom;
    }
}

void normalize_by_illumination(float* d_image, const float* d_illum,
                                size_t n, float epsilon)
{
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    normalize_by_illumination_kernel<<<blocks, threads>>>(d_image, d_illum,
                                                           (int)n, epsilon);
}
