#!/usr/bin/env python3
"""
Test if the geometry fix actually resolves the issue.
"""
import os
import sys
import numpy as np
from pathlib import Path

# Setup paths
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))
os.chdir(root_dir)

from src.utilities.config import Config
from TEMPEST_RAD.simulator import (
    load_shape_model, 
    compute_geometry, 
    RoughnessLUT, 
    planck_function, 
    rms_to_fraction, 
    calculate_theta
)

print("="*70)
print("TEST: Geometry Fix for LUT Lookup")
print("="*70)
print()

# Load config and shape model
CONFIG_PATH = "private/data/config/moon/moon_config.yaml"
LUT_PATH = "roughness_lut_spectral_v1.h5"

config = Config(CONFIG_PATH)
facets, mesh = load_shape_model(config.path_to_shape_model_file)
n_facets = len(facets)

print(f"Loaded {n_facets} facets")

# Load LUT
theta = calculate_theta(config)
lut = RoughnessLUT(LUT_PATH, target_theta=theta, target_rms=90.0)

# Test at zero phase
sun_vec = np.array(config.sunlight_direction)
obs_vec = sun_vec  # Zero phase
rot_axis = np.array([0, 0, 1])

# Compute geometry (will have wrong per-facet sun_phases)
lats, phases_wrong, emis, azis = compute_geometry(facets, sun_vec, obs_vec, rot_axis)

visible = emis < 90
print(f"\nVisible facets: {np.sum(visible)}/{n_facets}")

# Show the bug
print(f"\nBUG DEMONSTRATION:")
print(f"  Phases from compute_geometry: min={phases_wrong[visible].min():.1f}°, max={phases_wrong[visible].max():.1f}°, std={phases_wrong[visible].std():.1f}°")
print(f"  Each facet has different 'sun_phase' (local time)")

# Get correction factors with WRONG phases
wave_test = lut.axes['wavelength'][1]  # 8 microns
factors_wrong = lut.get_correction_factors(lats, phases_wrong, emis, azis, wavelength=wave_test)
print(f"\n  With WRONG phases (per-facet local times):")
print(f"    Factors: min={factors_wrong[visible].min():.4f}, max={factors_wrong[visible].max():.4f}, mean={factors_wrong[visible].mean():.4f}")
print(f"    >1.0: {np.sum(factors_wrong[visible] > 1.0)}/{np.sum(visible)}")

# Apply FIX: Override with global body rotation phase
time_hours = 12.0  # Midday
period = getattr(config, 'rotation_period_hours', 24.0)
body_rotation_phase = (time_hours / period * 360.0) % 360.0
phases_fixed = np.full_like(phases_wrong, body_rotation_phase)

print(f"\nFIX APPLIED:")
print(f"  Time: {time_hours}h / {period}h period")
print(f"  Body rotation phase: {body_rotation_phase:.1f}° (SAME for all facets)")
print(f"  Phases after fix: min={phases_fixed[visible].min():.1f}°, max={phases_fixed[visible].max():.1f}°, std={phases_fixed[visible].std():.1f}°")

# Get correction factors with FIXED phases
factors_fixed = lut.get_correction_factors(lats, phases_fixed, emis, azis, wavelength=wave_test)
print(f"\n  With FIXED phases (global body rotation):")
print(f"    Factors: min={factors_fixed[visible].min():.4f}, max={factors_fixed[visible].max():.4f}, mean={factors_fixed[visible].mean():.4f}")
print(f"    >1.0: {np.sum(factors_fixed[visible] > 1.0)}/{np.sum(visible)}")

# Test effect on temperature
T_test = 300.0
rms = 28.0
f = rms_to_fraction(rms)

# With wrong phases
multiplier_wrong = (1.0 - f) + f * factors_wrong[visible].mean()
T_rough_wrong = T_test * multiplier_wrong

# With fixed phases  
multiplier_fixed = (1.0 - f) + f * factors_fixed[visible].mean()
T_rough_fixed = T_test * multiplier_fixed

print(f"\nTEMPERATURE EFFECT:")
print(f"  Smooth: {T_test:.2f} K")
print(f"  Rough (WRONG phases): {T_rough_wrong:.2f} K ({T_rough_wrong - T_test:+.2f} K)")
print(f"  Rough (FIXED phases): {T_rough_fixed:.2f} K ({T_rough_fixed - T_test:+.2f} K)")

print()
if T_rough_fixed > T_test:
    print("✓✓✓ FIX WORKS! Rough surface is now WARMER")
    print(f"    Enhancement: {((T_rough_fixed/T_test - 1)*100):+.1f}%")
else:
    print("✗✗✗ FIX FAILED - still showing rough as cooler")

print()
print("="*70)
