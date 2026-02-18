#!/usr/bin/env python3
"""
Test the fix at the actual thermal peak (sun_phase = 0°)
"""
import os
import sys
import numpy as np
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.append(str(root_dir))
os.chdir(root_dir)

from src.utilities.config import Config
from TEMPEST_RAD.simulator import (
    load_shape_model, compute_geometry, RoughnessLUT, 
    planck_function, rms_to_fraction, calculate_theta
)

print("="*70)
print("TEST: Fix at Thermal Peak (sun_phase = 0°)")
print("="*70)

CONFIG_PATH = "private/data/config/moon/moon_config.yaml"
LUT_PATH = "roughness_lut_spectral_v1.h5"

config = Config(CONFIG_PATH)
facets, _ = load_shape_model(config.path_to_shape_model_file)
theta = calculate_theta(config)
lut = RoughnessLUT(LUT_PATH, target_theta=theta, target_rms=90.0)

# Zero phase geometry
sun_vec = np.array(config.sunlight_direction)
obs_vec = sun_vec
rot_axis = np.array([0, 0, 1])
lats, phases_wrong, emis, azis = compute_geometry(facets, sun_vec, obs_vec, rot_axis)
visible = emis < 90

# Force sun_phase = 0° (thermal peak)
phases_fixed = np.zeros_like(phases_wrong)

print(f"\nForcing sun_phase = 0° for ALL facets (thermal peak)")
print(f"Visible facets: {np.sum(visible)}/{len(facets)}")

# Test at multiple wavelengths
wavelengths = lut.axes['wavelength']
print(f"\nCorrection Factors at sun_phase=0° (thermal peak):")
for wave in wavelengths:
    factors = lut.get_correction_factors(lats, phases_fixed, emis, azis, wavelength=wave)
    factors_vis = factors[visible]
    print(f"  λ={wave:6.1f} μm: min={factors_vis.min():.4f}, max={factors_vis.max():.4f}, mean={factors_vis.mean():.4f}, >1.0: {np.sum(factors_vis > 1.0)}/{len(factors_vis)}")

# Bolometric calculation
T_test = 300.0
rms = 28.0
f = rms_to_fraction(rms)

# Calculate bolometric enhancement
spec_smooth = np.array([planck_function(w, T_test) for w in wavelengths])
spec_rough = np.zeros((np.sum(visible), len(wavelengths)))

for i, wave in enumerate(wavelengths):
    rad_s = planck_function(wave, T_test)
    factors = lut.get_correction_factors(lats, phases_fixed, emis, azis, wavelength=wave)
    factors_vis = factors[visible]
    spec_rough[:, i] = rad_s * ((1.0 - f) + f * factors_vis)

# Integrate
rad_bol_smooth = np.trapz(spec_smooth, x=wavelengths)
rad_bol_rough = np.trapz(spec_rough, x=wavelengths, axis=1).mean()

sigma = 5.670374419e-8
T_smooth = (np.pi * rad_bol_smooth / sigma) ** 0.25
T_rough = (np.pi * rad_bol_rough / sigma) ** 0.25

print(f"\nBolometric Brightness Temperature:")
print(f"  Smooth: {T_smooth:.2f} K")
print(f"  Rough:  {T_rough:.2f} K")
print(f"  Diff:   {T_rough - T_smooth:+.2f} K ({((T_rough/T_smooth - 1)*100):+.1f}%)")

if T_rough > T_smooth:
    print("\n✓✓✓ SUCCESS! Rough is WARMER at thermal peak")
else:
    print("\n✗✗✗ STILL WRONG - rough is cooler even at peak")

print("="*70)
