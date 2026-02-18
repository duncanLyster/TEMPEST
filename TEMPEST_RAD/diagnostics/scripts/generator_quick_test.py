#!/usr/bin/env python3
"""
Quick test configuration for Moon LUT generation (~2 minute run).

Moon parameters:
- TI = 55 J m⁻² K⁻¹ s⁻½
- P = 708.72 hours
- Theta ≈ 0.027 (closest to 0.0316 in standard grid)

This generates a minimal LUT with:
- 1 Theta value (Moon's actual value)
- 5 Latitude samples (0°, 22.5°, 45°, 67.5°, 90°)
- All wavelength/viewing angle resolution preserved
- Reduced subfacets (100) and rays (500) for speed

Total: 5 cases (~2 minutes)
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/duncan/Desktop/DPhil/TEMPEST')

import TEMPEST_RAD.generator as gen

# Moon's Theta value (TI=55, P=708.72h)
# Theta = TI * sqrt(omega) / (emissivity * sigma * Tss³) ≈ 0.027
gen.THETA_VALUES = np.array([0.0316])  # Closest standard grid point

# Sparse latitude sampling (captures key behaviors)
gen.LATITUDE_VALUES = np.array([0.0, 22.5, 45.0, 67.5, 90.0])

# Reduce computational cost for fast testing
gen.CRATER_SUBFACETS = 100  # Reduced from 300 (faster meshing + thermal solve)
gen.VIEW_FACTOR_RAYS = 500  # Reduced from 2000 (faster ray tracing)
gen.SIM_TIMESTEPS = 360     # Reduced from 720 (faster thermal simulation)

# Update output filename
gen.OUTPUT_FILE = "roughness_lut_moon_test.h5"

print("="*80)
print("Quick Test Configuration for Moon")
print("="*80)
print(f"Theta value: {gen.THETA_VALUES[0]:.4f} (TI ≈ 55 J m⁻² K⁻¹ s⁻½ at P=708h)")
print(f"Latitudes: {len(gen.LATITUDE_VALUES)} samples")
print(f"Crater subfacets: {gen.CRATER_SUBFACETS} (reduced for speed)")
print(f"View factor rays: {gen.VIEW_FACTOR_RAYS} (reduced for speed)")
print(f"Sim timesteps: {gen.SIM_TIMESTEPS} (reduced for speed)")
print(f"Total cases: {len(gen.THETA_VALUES) * len(gen.LATITUDE_VALUES)}")
print(f"Estimated time: ~2 minutes")
print("="*80)
print()

if __name__ == '__main__':
    gen.main()
