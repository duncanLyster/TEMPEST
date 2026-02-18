#!/usr/bin/env python3
"""
Diagnostic script to check if the roughness correction is being applied correctly.
"""

import numpy as np
import h5py

# Planck function
def planck_function(wavelength_um, temp_k):
    c1 = 1.191042e8
    c2 = 1.4387752e4
    t_safe = np.maximum(temp_k, 1e-5)
    return c1 / (wavelength_um**5 * (np.exp(c2 / (wavelength_um * t_safe)) - 1))

# Load LUT
with h5py.File('roughness_lut_spectral_v1.h5', 'r') as f:
    wavelengths = f['wavelength'][:]
    lut = f['lut'][:]
    
# Test case: disk center, midday, lat=0
# Emission = 0°, Azimuth = 0°, Time = midday (45)
factors = [1.650, 1.336, 1.125, 1.000, 0.980]  # From our earlier check

# Test temperature
T_smooth = 300.0  # K

# Mixing fraction for RMS = 40°
tan_rms = np.tan(np.radians(40.0))
tan_hemi = np.tan(np.radians(57.0))
f = np.clip(tan_rms / tan_hemi, 0.0, 1.0)

print(f"Test Configuration:")
print(f"  Temperature: {T_smooth} K")
print(f"  RMS angle: 40°")
print(f"  Mixing fraction f: {f:.3f}")
print(f"  Geometry: disk center, midday")
print()

# Calculate spectra
spec_smooth = np.array([planck_function(w, T_smooth) for w in wavelengths])
spec_rough = spec_smooth * ((1.0 - f) + f * np.array(factors))

print("Spectral Radiance:")
print(f"  Wave (μm) | Smooth | Rough  | Factor | Change")
print("-" * 55)
for i, w in enumerate(wavelengths):
    change_pct = ((spec_rough[i] / spec_smooth[i]) - 1.0) * 100
    print(f"  {w:7.1f}   | {spec_smooth[i]:6.2e} | {spec_rough[i]:6.2e} | {factors[i]:.3f}  | {change_pct:+.1f}%")

# Integrate to get bolometric radiance
rad_bol_smooth = np.trapz(spec_smooth, x=wavelengths)
rad_bol_rough = np.trapz(spec_rough, x=wavelengths)

print()
print(f"Bolometric Radiance:")
print(f"  Smooth: {rad_bol_smooth:.4e} W/m²/sr")
print(f"  Rough:  {rad_bol_rough:.4e} W/m²/sr")
print(f"  Ratio:  {rad_bol_rough / rad_bol_smooth:.4f}")

# Convert to brightness temperature
sigma = 5.670374419e-8
T_eff_smooth = (np.pi * rad_bol_smooth / sigma) ** 0.25
T_eff_rough = (np.pi * rad_bol_rough / sigma) ** 0.25

print()
print(f"Effective Brightness Temperature:")
print(f"  Smooth: {T_eff_smooth:.2f} K")
print(f"  Rough:  {T_eff_rough:.2f} K")
print(f"  Difference: {T_eff_rough - T_eff_smooth:+.2f} K")

print()
if T_eff_rough > T_eff_smooth:
    print("✓ Rough surface appears WARMER (expected for disk center at midday)")
else:
    print("✗ Rough surface appears COOLER (unexpected! indicates a bug)")
