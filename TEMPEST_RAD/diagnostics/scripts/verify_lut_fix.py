#!/usr/bin/env python3
"""
Test that the fixed LUT now produces the expected behavior.
"""

import numpy as np
import h5py

# Planck function
def planck_function(wavelength_um, temp_k):
    c1 = 1.191042e8
    c2 = 1.4387752e4
    t_safe = np.maximum(temp_k, 1e-5)
    return c1 / (wavelength_um**5 * (np.exp(c2 / (wavelength_um * t_safe)) - 1))

# Load FIXED LUT
with h5py.File('roughness_lut_spectral_v1.h5', 'r') as f:
    wavelengths = f['wavelength'][:]
    lut = f['lut'][:]

print("="*70)
print("VERIFICATION TEST: Fixed LUT")
print("="*70)
print()

# Test: Disk center at local noon should have factors > 1.0
# Geometry: emission=0°, azimuth=0°, sun_phase=0° (noon), lat=0°
theta_idx = 0  # θ = 1.0
ang_idx = 0    # 90° hemisphere
lat_idx = 0    # equator
time_idx = 0   # NOW this should be noon!
e_idx = 0      # emission = 0°
azi_idx = 0    # azimuth = 0°

print("Test 1: Disk center at local NOON")
print(f"  Geometry: emission=0°, azimuth=0°, sun_phase=0° (noon)")
print()

factors = []
for w_idx, wave in enumerate(wavelengths):
    factor = lut[theta_idx, ang_idx, lat_idx, time_idx, w_idx, e_idx, azi_idx]
    factors.append(factor)
    print(f"  {wave:6.1f} μm: factor = {factor:.3f}", end="")
    if factor > 1.1:
        print(" ✓ (enhanced)")
    elif factor > 1.0:
        print(" ~ (slightly enhanced)")
    else:
        print(" ✗ (REDUCED - unexpected!)")

print()

# Test with actual calculation
T_smooth = 300.0
tan_rms = np.tan(np.radians(40.0))
tan_hemi = np.tan(np.radians(57.0))
f = np.clip(tan_rms / tan_hemi, 0.0, 1.0)

spec_smooth = np.array([planck_function(w, T_smooth) for w in wavelengths])
spec_rough = spec_smooth * ((1.0 - f) + f * np.array(factors))

rad_bol_smooth = np.trapz(spec_smooth, x=wavelengths)
rad_bol_rough = np.trapz(spec_rough, x=wavelengths)

sigma = 5.670374419e-8
T_eff_smooth = (np.pi * rad_bol_smooth / sigma) ** 0.25
T_eff_rough = (np.pi * rad_bol_rough / sigma) ** 0.25

print(f"Test 2: Brightness Temperature Calculation")
print(f"  Physical temperature: {T_smooth:.1f} K")
print(f"  Roughness: RMS=40° → f={f:.3f}")
print(f"  Smooth Tb: {T_eff_smooth:.2f} K")
print(f"  Rough Tb:  {T_eff_rough:.2f} K")
print(f"  Difference: {T_eff_rough - T_eff_smooth:+.2f} K")
print()

if T_eff_rough > T_eff_smooth:
    print("✓✓✓ TEST PASSED! Rough surface is WARMER as expected!")
    print("    The LUT fix has resolved the issue.")
else:
    print("✗✗✗ TEST FAILED! Rough surface is still cooler.")
    print("    Additional debugging needed.")

print()
print("="*70)

# Test at different local times  
print("\nTest 3: Variation with Local Time")
print(f"  (All at disk center, emission=0°)")
print()
print(f"  Time Idx | Sun Phase | Factor@8μm | Expected")
print(f"  " + "-"*55)

w_idx = 1  # 8 microns
for t_idx in [0, 22, 45, 67]:
    factor = lut[theta_idx, ang_idx, lat_idx, t_idx, w_idx, e_idx, azi_idx]
    sun_phase = t_idx / 90 * 360
    
    if t_idx == 0:
        expected = "NOON (highest)"
    elif t_idx < 45:
        expected = "Morning"
    elif t_idx == 45:
        expected = "Evening"
    else:
        expected = "Night"
    
    print(f"    {t_idx:3d}    |  {sun_phase:6.1f}°  |   {factor:.3f}   | {expected}")

print()
peak_t = np.argmax([lut[theta_idx, ang_idx, lat_idx, t, w_idx, e_idx, azi_idx] for t in range(90)])
print(f"  Peak brightness at t_idx = {peak_t} (sun_phase = {peak_t/90*360:.0f}°)")

if peak_t < 10:
    print(f"  ✓ Peak near sun_phase=0° as expected (accounting for thermal lag)")
else:
    print(f"  ✗ Peak is far from sun_phase=0° - possible issue remains")
