#!/usr/bin/env python3
"""Diagnose the phase=90° zero-values bug and cold poles issue."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.utilities.config import Config
from src.utilities.utils import rotate_vector
from TEMPEST_RAD.simulator import load_shape_model

CONFIG_PATH = "private/data/config/moon/moon_config.yaml"
LUT_PATH = "roughness_lut_spectral_v1.h5"

config = Config(CONFIG_PATH)
facets, mesh = load_shape_model(config.path_to_shape_model_file)
n_facets = len(facets)

from TEMPEST_RAD.simulator import compute_geometry, rms_to_fraction
from TEMPEST_RAD.lut import RoughnessLUT

# Minimal Planck function
def planck_function(wavelength_um, temp_k):
    c1 = 1.191042e8
    c2 = 1.4387752e4
    t_safe = np.maximum(temp_k, 1e-5)
    return c1 / (wavelength_um**5 * (np.exp(c2 / (wavelength_um * t_safe)) - 1))

# Load temps
temps_path = "output/retrieval_analysis/temperatures.csv"
try:
    temps_all = np.loadtxt(temps_path, delimiter=',')
except:
    temps_all = np.loadtxt(temps_path, delimiter=',', skiprows=1)
if temps_all.shape[0] != n_facets and temps_all.shape[1] == n_facets:
    temps_all = temps_all.T

period = getattr(config, 'rotation_period_hours', 24.0)
n_steps = temps_all.shape[1]
time_hours = 12.0
idx = int((time_hours % period) / period * n_steps)
idx = np.clip(idx, 0, n_steps - 1)
temps_smooth = temps_all[:, idx]

# Load LUT
from TEMPEST_RAD.simulator import calculate_theta
theta = calculate_theta(config)
lut = RoughnessLUT(LUT_PATH, target_theta=theta, target_rms=90.0)

sun_vec = np.array(config.sunlight_direction, dtype=float)
rot_axis = np.array([0, 0, 1], dtype=float)

roughness_rms = 28.0
f = rms_to_fraction(roughness_rms)
wavelengths = lut.axes['wavelength']

print(f"Shape model: {n_facets} facets")
print(f"Period: {period} hours")
print(f"Temps at t={time_hours}h: min={temps_smooth.min():.1f}, max={temps_smooth.max():.1f}, mean={temps_smooth.mean():.1f} K")
print(f"Roughness fraction f={f:.3f}")
print(f"Sun direction: {sun_vec}")
print()

# Test specific phase angles
test_phases = [0, 44, 45, 88, 89, 90, 91, 92, 135, 180, -90, -45]

for phase_angle in test_phases:
    obs_vec = rotate_vector(sun_vec, rot_axis, np.radians(phase_angle))
    
    lats, phases_geo, emis, azis = compute_geometry(facets, sun_vec, obs_vec, rot_axis)
    
    # Override phases as the notebook does
    body_rotation_phase = (time_hours / period * 360.0) % 360.0
    phases = np.full_like(phases_geo, body_rotation_phase)
    
    visible = emis < 90
    n_vis = np.sum(visible)
    
    # Compute rough TB
    sigma = 5.670374419e-8
    full_spectra = np.zeros((n_facets, len(wavelengths)))
    for i, wave in enumerate(wavelengths):
        rad_smooth = planck_function(wave, temps_smooth)
        factors = lut.get_correction_factors(lats, phases, emis, azis, wavelength=wave)
        full_spectra[:, i] = rad_smooth * ((1.0 - f) + f * factors)
    
    rad_bol = np.trapz(full_spectra, x=wavelengths, axis=1)
    t_eff = (np.pi * rad_bol / sigma) ** 0.25
    t_eff[emis > 90] = 0
    
    # Smooth TB
    full_spectra_s = np.zeros((n_facets, len(wavelengths)))
    for i, wave in enumerate(wavelengths):
        full_spectra_s[:, i] = planck_function(wave, temps_smooth)
    rad_bol_s = np.trapz(full_spectra_s, x=wavelengths, axis=1)
    t_eff_s = (np.pi * rad_bol_s / sigma) ** 0.25
    t_eff_s[emis > 90] = 0
    
    # Check factors at first wavelength
    factors_check = lut.get_correction_factors(lats, phases, emis, azis, wavelength=wavelengths[0])
    
    if n_vis > 0:
        vis_s = t_eff_s[visible]
        vis_r = t_eff[visible]
        vis_f = factors_check[visible]
        vis_e = emis[visible]
        vis_a = azis[visible]
        vis_l = lats[visible]
        
        # Check for zeros, nans, infs
        n_zero_s = np.sum(vis_s == 0)
        n_zero_r = np.sum(vis_r == 0)
        n_nan_f = np.sum(np.isnan(vis_f))
        n_zero_f = np.sum(vis_f == 0)
        
        print(f"Phase={phase_angle:+4d} | obs={obs_vec.round(3)} | vis={n_vis:4d}/{n_facets} | "
              f"Ts={np.mean(vis_s):6.1f}K | Tr={np.mean(vis_r):6.1f}K | dT={np.mean(vis_r)-np.mean(vis_s):+5.1f}K | "
              f"F_mean={np.mean(vis_f):.3f} | F_min={np.min(vis_f):.3f} | F_max={np.max(vis_f):.3f} | "
              f"zero_s={n_zero_s} zero_r={n_zero_r} nan_f={n_nan_f} zero_f={n_zero_f} | "
              f"emi=[{np.min(vis_e):.1f},{np.max(vis_e):.1f}] | "
              f"azi=[{np.min(vis_a):.1f},{np.max(vis_a):.1f}] | "
              f"lat=[{np.min(vis_l):.1f},{np.max(vis_l):.1f}]")
    else:
        print(f"Phase={phase_angle:+4d} | obs={obs_vec.round(3)} | vis=0/{n_facets} | NO VISIBLE FACETS")

print()
print("="*80)
print("COLD POLES DIAGNOSTIC")
print("="*80)

# Check LUT behavior at high latitudes  
phase_angle = 0
obs_vec = rotate_vector(sun_vec, rot_axis, np.radians(phase_angle))
lats, _, emis, azis = compute_geometry(facets, sun_vec, obs_vec, rot_axis)
body_rotation_phase = (time_hours / period * 360.0) % 360.0
phases = np.full(n_facets, body_rotation_phase)
visible = emis < 90

# Bin by latitude
lat_bins = [0, 15, 30, 45, 60, 75, 90]
for i in range(len(lat_bins)-1):
    lat_lo, lat_hi = lat_bins[i], lat_bins[i+1]
    mask = visible & (lats >= lat_lo) & (lats < lat_hi)
    n_in_bin = np.sum(mask)
    if n_in_bin == 0:
        print(f"  Lat [{lat_lo:2d}-{lat_hi:2d}]: no visible facets")
        continue
    
    # Get factors at each wavelength
    all_factors = []
    for wave in wavelengths:
        fac = lut.get_correction_factors(lats[mask], phases[mask], emis[mask], azis[mask], wavelength=wave)
        all_factors.append(fac)
    all_factors = np.column_stack(all_factors)  # (n_facets, n_waves)
    
    # Compute TB
    spec_rough = np.zeros((n_in_bin, len(wavelengths)))
    spec_smooth = np.zeros((n_in_bin, len(wavelengths)))
    for iw, wave in enumerate(wavelengths):
        rs = planck_function(wave, temps_smooth[mask])
        spec_smooth[:, iw] = rs
        spec_rough[:, iw] = rs * ((1-f) + f * all_factors[:, iw])
    
    tb_s = (np.pi * np.trapz(spec_smooth, x=wavelengths, axis=1) / sigma) ** 0.25
    tb_r = (np.pi * np.trapz(spec_rough, x=wavelengths, axis=1) / sigma) ** 0.25
    
    mean_fac = np.mean(all_factors)
    min_fac = np.min(all_factors)
    max_fac = np.max(all_factors)
    
    print(f"  Lat [{lat_lo:2d}-{lat_hi:2d}]: n={n_in_bin:4d} | Ts={np.mean(tb_s):6.1f}K | Tr={np.mean(tb_r):6.1f}K | "
          f"dT={np.mean(tb_r)-np.mean(tb_s):+5.1f}K | "
          f"F_mean={mean_fac:.3f} F_min={min_fac:.3f} F_max={max_fac:.3f} | "
          f"emi=[{np.min(emis[mask]):.0f}-{np.max(emis[mask]):.0f}]")

# Also check what the LUT raw values look like at high latitudes
print()
print("Raw LUT factors at specific lat/emi/azi combinations (phase=6.1 deg):")
test_points = [
    (0, 0, 0, "equator, nadir, subsolar"),
    (0, 30, 0, "equator, e=30, subsolar"), 
    (0, 60, 0, "equator, e=60, subsolar"),
    (60, 0, 0, "lat=60, nadir, subsolar"),
    (60, 30, 0, "lat=60, e=30, subsolar"),
    (75, 0, 0, "lat=75, nadir, subsolar"),
    (85, 0, 0, "lat=85, nadir, subsolar"),
    (89, 0, 0, "lat=89, nadir, subsolar"),
    (90, 0, 0, "lat=90 (pole), nadir"),
]

for lat, emi, azi, label in test_points:
    factors_per_wave = []
    for wave in wavelengths:
        fval = lut.get_correction_factors(
            np.array([lat], dtype=float),
            np.array([body_rotation_phase]),
            np.array([emi], dtype=float),
            np.array([azi], dtype=float),
            wavelength=wave
        )
        factors_per_wave.append(fval[0])
    fstr = ", ".join(f"{fv:.3f}" for fv in factors_per_wave)
    print(f"  {label:35s}: R=[{fstr}]")

print("\nDone.")
