import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Paths
# Assumes run from TEMPEST root
ROOT = Path(".")
if len(sys.argv) > 1:
    LUT_FILE = sys.argv[1]
else:
    # Look for the new standard file first
    if os.path.exists("roughness_lut_spectral_v1.h5"):
        LUT_FILE = "roughness_lut_spectral_v1.h5"
    else:
        LUT_FILE = "roughness_lut_spectral_v1_factors.h5"

OUTPUT_DIR = Path("TEMPEST_RAD/diagnostics/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try to find the file
if os.path.exists(LUT_FILE):
    print(f"Using {LUT_FILE} from CWD")
elif os.path.exists(f"TEMPEST_RAD/diagnostics/{LUT_FILE}"):
    LUT_FILE = f"TEMPEST_RAD/diagnostics/{LUT_FILE}"
    print(f"Using {LUT_FILE}")
else:
    print(f"Error: {LUT_FILE} not found. Please run generator first.")
    sys.exit(1)
             
print(f"Loading {LUT_FILE}...")

# Load Data
with h5py.File(LUT_FILE, 'r') as f:
    factors_raw = f["normalization_factors"][...]
    # wavelengths = f["wavelength"][...] # might be gone in new logic? No, just not used for factor
    latitudes = f["latitude"][...]
    theta_vals = f["theta"][...]

# Handle Dimensions
# Old format: [theta, angle, lat, wave] -> (1, 1, 37, 5)
# New format: [theta, angle, lat] -> (1, 1, 37) (Bolometric scalar)

if factors_raw.ndim == 4:
    print("Detected old format (Spectral Factors). Calculating mean over wavelength for plot.")
    # Assuming shape [1, 1, N_lat, N_wave]
    # Reduce to [N_lat]
    factors = np.mean(factors_raw[0, 0, :, :], axis=1)
    title_suffix = "(Spectral Mean)"
elif factors_raw.ndim == 3:
    print("Detected new format (Bolometric Scalar).")
    # Assuming shape [1, 1, N_lat]
    factors = factors_raw[0, 0, :]
    if factors.ndim > 1: # Just in case
        factors = factors.flatten()
    title_suffix = "(Bolometric)"
else:
    print(f"Unexpected shape: {factors_raw.shape}")
    factors = factors_raw.flatten()

# 1. Plot vs Latitude (Full Range)
plt.figure(figsize=(10, 6))
plt.plot(latitudes, factors, 'o-', linewidth=2, label=f'Factor {title_suffix}')
plt.xlabel("Latitude (deg)")
plt.ylabel("Normalization Factor")
plt.title(f"Normalization Factor vs Latitude {title_suffix} - Full Range")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
out_full = OUTPUT_DIR / "norm_factor_vs_latitude_bolometric_full.png"
plt.savefig(out_full)
print(f"Saved {out_full}")

# 2. Plot vs Latitude (Cutoff at 80)
mask = latitudes <= 80.0
# Also filter invalid values if any
if np.any(np.isnan(factors[mask])):
    print("Warning: NaNs detected in factors, skipping them in plot.")
    valid = ~np.isnan(factors)
    mask = mask & valid

plt.figure(figsize=(10, 6))
plt.plot(latitudes[mask], factors[mask], 'o-', linewidth=2, color='tab:blue', label=f'Factor {title_suffix}')
plt.xlabel("Latitude (deg)")
plt.ylabel("Normalization Factor")
plt.title(f"Normalization Factor vs Latitude (Lat <= 80)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
out_80 = OUTPUT_DIR / "norm_factor_vs_latitude_bolometric_80deg.png"
plt.savefig(out_80)
print(f"Saved {out_80}")

# Print Table
print("\nNormalization Factors Table:")
print(f"{'Lat (deg)':<10} | {'Factor':<10}")
print("-" * 25)
for i in range(len(latitudes)):
    print(f"{latitudes[i]:<10.1f} | {factors[i]:.4f}")
