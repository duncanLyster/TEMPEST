#!/usr/bin/env python3
"""
Simulate the notebook's validation test with all fixes applied
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
print("NOTEBOOK VALIDATION TEST (with all fixes)")
print("="*70)

CONFIG_PATH = "private/data/config/moon/moon_config.yaml"
LUT_PATH = "roughness_lut_spectral_v1.h5"
OUTPUT_DIR = "output/retrieval_analysis"

config = Config(CONFIG_PATH)
facets, _ = load_shape_model(config.path_to_shape_model_file)
n_facets = len(facets)

# Load temperatures
temps_all = np.loadtxt(os.path.join(OUTPUT_DIR, "temperatures.csv"), delimiter=',')
if temps_all.shape[0] != n_facets and temps_all.shape[1] == n_facets:
    temps_all = temps_all.T

# Load LUT
theta = calculate_theta(config)
lut = RoughnessLUT(LUT_PATH, target_theta=theta, target_rms=90.0)

def get_temps_at_time(time_hours):
    period = getattr(config, 'rotation_period_hours', 24.0)
    n_steps = temps_all.shape[1]
    idx = int((time_hours % period) / period * n_steps)
    idx = np.clip(idx, 0, n_steps - 1)
    return temps_all[:, idx]

def calculate_bolometric_tb(time_hours, roughness_rms, phase_angle):
    sun_vec = np.array(config.sunlight_direction)
    from src.utilities.utils import rotate_vector
    rot_axis = np.array([0, 0, 1])
    
    perp_vec = np.cross(sun_vec, rot_axis)
    if np.linalg.norm(perp_vec) < 1e-6: perp_vec = np.array([0, 1, 0])
    obs_vec = rotate_vector(sun_vec, perp_vec, np.radians(phase_angle))
    
    lats, phases, emis, azis = compute_geometry(facets, sun_vec, obs_vec, rot_axis)
    
    # FIX #1: Override per-facet local times with global body rotation phase
    period = getattr(config, 'rotation_period_hours', 24.0)
    body_rotation_phase = (time_hours / period * 360.0) % 360.0
    phases = np.full_like(phases, body_rotation_phase)
    
    temps_smooth = get_temps_at_time(time_hours)
    f = rms_to_fraction(roughness_rms)
    
    wavelengths = lut.axes['wavelength']
    full_spectra = np.zeros((n_facets, len(wavelengths)))
    
    for i, wave in enumerate(wavelengths):
        rad_smooth = planck_function(wave, temps_smooth)
        factors = lut.get_correction_factors(lats, phases, emis, azis, wavelength=wave)
        full_spectra[:, i] = rad_smooth * ((1.0 - f) + f * factors)
    
    rad_bol = np.trapezoid(full_spectra, x=wavelengths, axis=1)
    sigma = 5.670374419e-8
    t_eff = (np.pi * rad_bol / sigma) ** 0.25
    t_eff[emis > 90] = 0
    
    return t_eff, obs_vec, emis

# Test at zero phase
time_tgt = 12.0
phase_tgt = 0.0
roughness_rms = 40.0

print(f"\n[Test 1] Zero Phase Beaming Check")
print(f"Time: {time_tgt}h, Phase: {phase_tgt}°, RMS: {roughness_rms}°")
print()

tb_smooth_0, _, emis_0 = calculate_bolometric_tb(time_tgt, 0.0, 0.0)  
tb_rough_0, _, _ = calculate_bolometric_tb(time_tgt, roughness_rms, 0.0)

mask_0 = emis_0 < 90
mean_s_0 = np.mean(tb_smooth_0[mask_0])
mean_r_0 = np.mean(tb_rough_0[mask_0])

print(f"Mean Disk Temp @ Phase 0°:")
print(f"  Smooth: {mean_s_0:.2f} K")
print(f"  Rough:  {mean_r_0:.2f} K")
print(f"  Diff:   {mean_r_0 - mean_s_0:+.2f} K")

if mean_r_0 > mean_s_0:
    print("\n✓✓✓ TEST PASSED! Rough model is warmer (Beaming detected)")
    print(f"    Enhancement: {((mean_r_0/mean_s_0 - 1)*100):+.1f}%")
else:
    print("\n✗✗✗ TEST FAILED! Rough model is not warmer")

print("\n" + "="*70)
print("EXPECTED RESULT: Rough surface should be a few K warmer at zero phase")
print("due to thermal beaming from crater interiors.")
print("="*70)
