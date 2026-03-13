#!/usr/bin/env python3
"""Visualize FWI 2D gradient dumps to diagnose energy distribution."""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')

iters = [1, 2, 3, 10, 50]
grads = {}
for i in iters:
    path = os.path.join(build_dir, f'fwi_2d_gradient_iter{i}.npy')
    if os.path.exists(path):
        grads[i] = np.load(path)
        print(f"Iter {i}: shape={grads[i].shape}, min={grads[i].min():.4e}, max={grads[i].max():.4e}, "
              f"absmax={np.abs(grads[i]).max():.4e}, mean_abs={np.abs(grads[i]).mean():.4e}")

if not grads:
    print("No gradient files found!")
    sys.exit(1)

nz, nx = list(grads.values())[0].shape

# --- Figure 1: Gradient images at each iteration ---
fig, axes = plt.subplots(1, len(grads), figsize=(4*len(grads), 5))
if len(grads) == 1:
    axes = [axes]
for ax, (it, g) in zip(axes, grads.items()):
    vmax = np.percentile(np.abs(g), 99)
    im = ax.imshow(g, cmap='seismic', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(f'Iter {it}')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Z (grid)')
    plt.colorbar(im, ax=ax, shrink=0.7)
fig.suptitle('Raw Gradient at Each Iteration', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(build_dir, 'gradient_images.png'), dpi=150)

# --- Figure 2: Depth profiles of gradient magnitude ---
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for it, g in grads.items():
    row_rms = np.sqrt(np.mean(g**2, axis=1))
    ax1.plot(row_rms, np.arange(nz), label=f'Iter {it}')
ax1.set_xlabel('RMS gradient magnitude')
ax1.set_ylabel('Depth (grid)')
ax1.invert_yaxis()
ax1.set_title('Gradient RMS vs Depth')
ax1.legend()
ax1.set_xscale('log')

# Lateral profile at lens depth (z~96) and shallow (z~30)
lens_z = 96
shallow_z = 30
g1 = grads[1]
ax2.plot(np.abs(g1[shallow_z, :]), label=f'z={shallow_z} (shallow)')
ax2.plot(np.abs(g1[lens_z, :]), label=f'z={lens_z} (lens)')
ax2.set_xlabel('X (grid)')
ax2.set_ylabel('|gradient|')
ax2.set_title('Lateral Gradient Profile (Iter 1)')
ax2.legend()
ax2.set_yscale('log')

fig2.tight_layout()
fig2.savefig(os.path.join(build_dir, 'gradient_profiles.png'), dpi=150)

# --- Figure 3: Ratio analysis ---
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
g1 = grads[1]
row_rms = np.sqrt(np.mean(g1**2, axis=1))
row_rms_safe = np.where(row_rms > 0, row_rms, 1e-30)
# Show ratio relative to lens depth
lens_rms = row_rms_safe[lens_z]
ratio = row_rms_safe / lens_rms
ax3.plot(ratio, np.arange(nz))
ax3.axhline(y=lens_z, color='r', linestyle='--', label=f'Lens depth (z={lens_z})')
ax3.axhline(y=30, color='b', linestyle='--', label='Water bottom (~z=30)')
ax3.set_xlabel('Gradient RMS ratio (relative to lens depth)')
ax3.set_ylabel('Depth (grid)')
ax3.invert_yaxis()
ax3.set_xscale('log')
ax3.set_title('Gradient Strength Ratio vs Lens Depth (Iter 1)')
ax3.legend()
fig3.tight_layout()
fig3.savefig(os.path.join(build_dir, 'gradient_ratio.png'), dpi=150)

print(f"\nSaved: gradient_images.png, gradient_profiles.png, gradient_ratio.png in {build_dir}")
print(f"\nKey diagnostic: shallow/lens RMS ratio = {row_rms_safe[shallow_z]/lens_rms:.1f}x")
# plt.show()  # non-interactive
