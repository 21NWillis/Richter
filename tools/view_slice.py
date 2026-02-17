#!/usr/bin/env python3
"""Richter — View a 2D XY slice snapshot from the simulation."""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main():
    # Default path — look next to the executable or in current dir
    npy_path = "slice_xy.npy"
    if len(sys.argv) > 1:
        npy_path = sys.argv[1]

    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        print("Run ./build/snapshot first to generate the slice.")
        sys.exit(1)

    data = np.load(npy_path)
    print(f"Loaded {npy_path}: shape={data.shape}, dtype={data.dtype}")
    print(f"  min={data.min():.6e}  max={data.max():.6e}")

    # Symmetric colorbar centered at zero for pressure fields
    vmax = max(abs(data.min()), abs(data.max()))
    if vmax == 0:
        vmax = 1.0  # avoid division by zero if field is empty

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(data, cmap='seismic', origin='lower', aspect='equal',
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Pressure', shrink=0.8)
    ax.set_xlabel('X (grid points)')
    ax.set_ylabel('Y (grid points)')
    ax.set_title(f'XY Slice at z=N/2  ({data.shape[0]}×{data.shape[1]})')

    # Save to file
    out_path = npy_path.replace('.npy', '.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
