#pragma once

/// Apply a simple sponge (exponential damping) absorbing boundary condition.
/// This prevents artificial reflections from the grid edges.
/// @param u       Pressure field to damp (device pointer)
/// @param nx,ny,nz Grid dimensions
/// @param sponge_width Number of cells at each edge to damp (typically 20-40)
/// @param damping_factor Maximum damping strength (0.0 = none, 1.0 = full absorb)
void apply_sponge_boundary(float* u, int nx, int ny, int nz,
                           int sponge_width, float damping_factor);
