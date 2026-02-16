// ─── Richter: Shared Memory 2.5D Tiling Kernel ─────────────────────
// Implementation B: XY plane tiled into shared memory, Z streamed.
// Stub — to be implemented in Phase 2 (Days 5-6).

#include "richter/kernels.h"
#include "richter/config.h"
#include <cstdio>

void launch_kernel_shmem(const float* u_prev, const float* u_curr,
                         float* u_next, const float* vel,
                         int nx, int ny, int nz)
{
    // TODO: Implement shared-memory 2.5D blocking kernel
    fprintf(stderr, "[Richter] hello_shmem not yet implemented.\n");
}
