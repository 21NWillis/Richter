#!/usr/bin/env python3
"""Visualize FWI gradient and illumination dumps for debugging."""
import numpy as np
import matplotlib.pyplot as plt
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for fname, title, ax, cmap in [
    ('fwi_2d_raw_gradient.raw', 'Raw Gradient (before processing)', axes[0], 'seismic'),
    ('fwi_2d_illumination.raw', 'Illumination', axes[1], 'hot'),
    ('fwi_2d_gradient_iter1.raw', 'Processed Gradient (iter 1)', axes[2], 'seismic'),
]:
    try:
        data = np.fromfile(fname, dtype=np.float32).reshape(N, N)
        if cmap == 'seismic':
            vmax = np.percentile(np.abs(data), 99)
            if vmax == 0: vmax = 1
            im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax,
                          extent=[0, N, N, 0])
        else:
            im = ax.imshow(data, cmap=cmap, aspect='auto', extent=[0, N, N, 0])
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Z / Depth')

        # Print stats
        print(f"{title}:")
        print(f"  range: [{data.min():.3e}, {data.max():.3e}]")
        print(f"  at lens (96,128): {data[96,128]:.3e}")
        print(f"  max|val| row 25: {np.abs(data[25,:]).max():.3e}")
        print(f"  max|val| row 96: {np.abs(data[96,:]).max():.3e}")
        print()
    except Exception as e:
        ax.text(0.5, 0.5, f'Not found:\n{fname}', ha='center', va='center',
                transform=ax.transAxes)
        print(f"Could not load {fname}: {e}")

plt.tight_layout()
plt.savefig('fwi_2d_gradient_debug.png', dpi=150, bbox_inches='tight')
print("Saved fwi_2d_gradient_debug.png")
plt.show()
