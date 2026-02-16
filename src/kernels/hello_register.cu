// ─── Richter: Register Rotation / Sliding Window Kernel ─────────────
// Implementation C: Registers hold Z-axis stencil values, sliding down.
// Stub — to be implemented in Phase 2 (Days 7-8).

#include "richter/kernels.h"
#include "richter/config.h"
#include <cstdio>

void launch_kernel_register(const float* u_prev, const float* u_curr,
                            float* u_next, const float* vel,
                            int nx, int ny, int nz)
{
    // TODO: Implement register-rotation kernel
    fprintf(stderr, "[Richter] hello_register not yet implemented.\n");
}
