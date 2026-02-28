// Receiver Kernels
// Record pressure at receiver locations (forward) and inject traces
// back into the pressure field (adjoint / backward).

#include "richter/rtm.h"
#include <cuda_runtime.h>
#include <cstdio>

// Record: Gather pressure at receiver locations
__global__ void record_receivers_kernel(const float* __restrict__ u,
                                        float* __restrict__ traces,
                                        const int* __restrict__ rx,
                                        const int* __restrict__ ry,
                                        const int* __restrict__ rz,
                                        int num_receivers, int nx, int ny,
                                        int t, int nt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_receivers) return;

    int idx = rz[i] * nx * ny + ry[i] * nx + rx[i];
    traces[i * nt + t] = u[idx];
}

void launch_record_receivers(const float* d_u, float* d_traces,
                             const int* d_rx, const int* d_ry, const int* d_rz,
                             int num_receivers, int nx, int ny,
                             int t, int nt)
{
    int threads = 256;
    int blocks = (num_receivers + threads - 1) / threads;
    record_receivers_kernel<<<blocks, threads>>>(d_u, d_traces,
                                                  d_rx, d_ry, d_rz,
                                                  num_receivers, nx, ny,
                                                  t, nt);
}

// Inject: Scatter trace values into pressure field (adjoint)
__global__ void inject_receivers_kernel(float* __restrict__ u,
                                        const float* __restrict__ traces,
                                        const int* __restrict__ rx,
                                        const int* __restrict__ ry,
                                        const int* __restrict__ rz,
                                        int num_receivers, int nx, int ny,
                                        int t, int nt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_receivers) return;

    int idx = rz[i] * nx * ny + ry[i] * nx + rx[i];
    u[idx] += traces[i * nt + t];
}

void launch_inject_receivers(float* d_u, const float* d_traces,
                             const int* d_rx, const int* d_ry, const int* d_rz,
                             int num_receivers, int nx, int ny,
                             int t, int nt)
{
    int threads = 256;
    int blocks = (num_receivers + threads - 1) / threads;
    inject_receivers_kernel<<<blocks, threads>>>(d_u, d_traces,
                                                  d_rx, d_ry, d_rz,
                                                  num_receivers, nx, ny,
                                                  t, nt);
}
