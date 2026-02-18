import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT = Path("TEMPEST_RAD")
LUT_FILE = "roughness_lut_spectral_v1_factors.h5"
OUTPUT_DIR = Path("TEMPEST_RAD/diagnostics/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Data
with h5py.File(LUT_FILE, 'r') as f:
    factors = f["normalization_factors"][...]  # [theta, angle, lat, wave]
    wavelengths = f["wavelength"][...]
    latitudes = f["latitude"][...]
    theta_vals = f["theta"][...]

# We assume single theta and opening angle for now as per generator config
# factors shape: (1, 1, 37, 5) if standard grid used
# Extract the 2D array: (Latitude, Wavelength)
factors_2d = factors[0, 0, :, :]

# Plot 1: Factor vs Wavelength for specific latitudes
target_lats = [0.0, 45.0, 80.0]
fig, ax = plt.subplots(figsize=(10, 6))

for target in target_lats:
    # Find nearest latitude index
    idx = np.argmin(np.abs(latitudes - target))
    actual_lat = latitudes[idx]
    
    y_vals = factors_2d[idx, :]
    ax.plot(wavelengths, y_vals, 'o-', linewidth=2, label=f"Lat {actual_lat:.1f}°")

ax.set_xlabel("Wavelength (microns)")
ax.set_ylabel("Normalization Factor (Rough/Smooth Energy Ratio)")
ax.set_title("Normalization Factor vs Wavelength")
ax.grid(True, which="both", linestyle="--", alpha=0.6)
ax.legend()
ax.set_xscale('log')
plt.savefig(OUTPUT_DIR / "norm_factor_vs_wavelength.png")
print(f"Saved {OUTPUT_DIR / 'norm_factor_vs_wavelength.png'}")

# Plot 2: Factor vs Latitude for one wavelength bin (e.g. 15 microns)
target_wave = 15.0
w_idx = np.argmin(np.abs(wavelengths - target_wave))
actual_wave = wavelengths[w_idx]

fig, ax = plt.subplots(figsize=(10, 6))
y_vals = factors_2d[:, w_idx]

ax.plot(latitudes, y_vals, 's-', color='darkred', linewidth=2)
ax.set_xlabel("Latitude (degrees)")
ax.set_ylabel(f"Normalization Factor (at {actual_wave} µm)")
ax.set_title(f"Normalization Factor vs Latitude (λ={actual_wave} µm)")
ax.grid(True, linestyle="--", alpha=0.6)
plt.savefig(OUTPUT_DIR / "norm_factor_vs_latitude.png")
print(f"Saved {OUTPUT_DIR / 'norm_factor_vs_latitude.png'}")

# Print Table for Inspection
print("\nNormalization Factors Table:")
print(f"{'Lat (deg)':<10} | " + " | ".join([f"{w:>7.1f} um" for w in wavelengths]))
print("-" * 70)
for i, lat in enumerate(latitudes):
    row_str = " | ".join([f"{f:.4f}" for f in factors_2d[i, :]])
    print(f"{lat:<10.1f} | {row_str}")
