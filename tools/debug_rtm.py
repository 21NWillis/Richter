import numpy as np
img = np.load('rtm_image.npy')
print(f'shape={img.shape}, min={img.min():.4e}, max={img.max():.4e}')
z_profile = np.sum(np.abs(img), axis=1)
peak_z = np.argmax(z_profile)
print(f'Peak Z-profile energy at z={peak_z}')
print()
for z in range(img.shape[0]):
    bar = '#' * int(50 * z_profile[z] / (z_profile.max() + 1e-30))
    print(f'  z={z:3d}: {z_profile[z]:12.4e}  {bar}')
