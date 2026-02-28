#!/usr/bin/env python3
"""Richter: RTM Image Viewer
Loads rtm_image.npy and renders both the raw cross-correlation image
and the Laplacian-filtered image (standard RTM post-processing).

The Laplacian filter suppresses the low-frequency source artifact that
dominates the raw cross-correlation, revealing the actual reflectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import sys
import os

def main():
    fname = "rtm_image.npy"
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    if not os.path.exists(fname):
        print(f"Error: {fname} not found. Run ./rtm_demo first.")
        sys.exit(1)

    img = np.load(fname)
    print(f"Loaded {fname}: shape={img.shape}, dtype={img.dtype}")
    print(f"  min={img.min():.6e}  max={img.max():.6e}")

    # Apply Laplacian filter (2D, on the XZ slice)
    # This is standard RTM post-processing: suppresses the low-frequency
    # source artifact and enhances reflector boundaries
    img_lap = laplace(img)
    print(f"  Laplacian: min={img_lap.min():.6e}  max={img_lap.max():.6e}")

    # --- Create side-by-side figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Raw cross-correlation
    vmax_raw = np.percentile(np.abs(img), 99)
    if vmax_raw == 0: vmax_raw = 1.0
    im1 = ax1.imshow(img, cmap='seismic', aspect='auto',
                     vmin=-vmax_raw, vmax=vmax_raw,
                     extent=[0, img.shape[1], img.shape[0], 0])
    ax1.set_xlabel('X (grid points)', fontsize=11)
    ax1.set_ylabel('Z / Depth (grid points)', fontsize=11)
    ax1.set_title('Raw Cross-Correlation', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Laplacian-filtered
    vmax_lap = np.percentile(np.abs(img_lap), 99)
    if vmax_lap == 0: vmax_lap = 1.0
    im2 = ax2.imshow(img_lap, cmap='seismic', aspect='auto',
                     vmin=-vmax_lap, vmax=vmax_lap,
                     extent=[0, img.shape[1], img.shape[0], 0])
    ax2.set_xlabel('X (grid points)', fontsize=11)
    ax2.set_ylabel('Z / Depth (grid points)', fontsize=11)
    ax2.set_title('Laplacian Filtered (reflectors)', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle('RTM Image — XZ Slice', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('rtm_image.png', dpi=150, bbox_inches='tight')
    print("Saved rtm_image.png")
    plt.show()

if __name__ == '__main__':
    main()
