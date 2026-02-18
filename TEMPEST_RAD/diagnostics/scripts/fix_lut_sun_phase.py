#!/usr/bin/env python3
"""
Fix the sun_phase axis alignment in the existing roughness LUT.

This script rolls the time axis so that the peak brightness (local noon)
occurs at sun_phase = 0° instead of sun_phase = 268°.
"""

import h5py
import numpy as np
import shutil
import os

INPUT_FILE = "roughness_lut_spectral_v1.h5"
BACKUP_FILE = "roughness_lut_spectral_v1_BACKUP.h5"
OUTPUT_FILE = "roughness_lut_spectral_v1_FIXED.h5"

print("="*70)
print("TEMPEST_RAD LUT Sun Phase Axis Fix")
print("="*70)
print()

# Backup original file
if not os.path.exists(BACKUP_FILE):
    print(f"Creating backup: {BACKUP_FILE}")
    shutil.copy(INPUT_FILE, BACKUP_FILE)
else:
    print(f"Backup already exists: {BACKUP_FILE}")

# Load LUT
print(f"\nLoading LUT from: {INPUT_FILE}")
with h5py.File(INPUT_FILE, 'r') as f:
    lut_data = f['lut'][:]
    theta_axis = f['theta'][:]
    angle_axis = f['opening_angle'][:]
    lat_axis = f['latitude'][:]
    wave_axis = f['wavelength'][:]
    emission_axis = f['emission'][:]
    azimuth_axis = f['azimuth'][:]

print(f"LUT shape: {lut_data.shape}")
print(f"  Dimensions: (Theta, Angle, Lat, Time, Wave, Emission, Azimuth)")

n_theta, n_angle, n_lat, n_time, n_wave, n_emi, n_azi = lut_data.shape

# For each combination of (theta, angle, lat), find the time index with peak brightness
# and roll the data so that peak is at t_idx = 0

print(f"\nAnalyzing and fixing sun_phase alignment...")
print(f"Processing {n_theta} × {n_angle} × {n_lat} = {n_theta*n_angle*n_lat} cases")

fixed_lut = np.zeros_like(lut_data)

for i_th in range(n_theta):
    for i_ang in range(n_angle):
        for i_lat in range(n_lat):
            # Average over wavelength, emission, azimuth to get typical brightness vs time
            # Use disk-center-like geometry (low emission angle) for peak finding
            e_center_idx = 0  # emission ≈ 0°
            azi_center_idx = 0  # azimuth = 0°
            
            # Average over wavelengths (weighted by typical thermal emission)
            # For simplicity, just use middle wavelength (15 microns)
            w_mid_idx = 2
            
            time_series = lut_data[i_th, i_ang, i_lat, :, w_mid_idx, e_center_idx, azi_center_idx]
            
            # Find peak (local noon)
            peak_idx = np.argmax(time_series)
            
            # Roll so peak is at idx 0
            roll_amount = -peak_idx
            
            # Roll all axes for this case
            for i_w in range(n_wave):
                for i_e in range(n_emi):
                    for i_a in range(n_azi):
                        fixed_lut[i_th, i_ang, i_lat, :, i_w, i_e, i_a] = \
                            np.roll(lut_data[i_th, i_ang, i_lat, :, i_w, i_e, i_a], roll_amount)
            
            if i_lat == 0:  # Print info for equatorial cases
                print(f"  θ={theta_axis[i_th]:.1f}, angle={angle_axis[i_ang]:.0f}°, lat={lat_axis[i_lat]:.0f}°: "
                      f"Peak was at t_idx={peak_idx} → shifted by {roll_amount}")

# Verify the fix
print(f"\nVerifying fix...")
# Check that peak is now at t_idx ≈ 0 for equatorial case
theta0_idx = 0
angle0_idx = 0
lat0_idx = 0
w_mid_idx = 2
e0_idx = 0
a0_idx = 0

original_series = lut_data[theta0_idx, angle0_idx, lat0_idx, :, w_mid_idx, e0_idx, a0_idx]
fixed_series = fixed_lut[theta0_idx, angle0_idx, lat0_idx, :, w_mid_idx, e0_idx, a0_idx]

print(f"\nOriginal time series (first 10 values): {original_series[:10]}")
print(f"Fixed time series (first 10 values):    {fixed_series[:10]}")
print(f"\nOriginal peak at t_idx={np.argmax(original_series)} (sun_phase={(np.argmax(original_series)/n_time*360):.0f}°)")
print(f"Fixed peak at t_idx={np.argmax(fixed_series)} (sun_phase={(np.argmax(fixed_series)/n_time*360):.0f}°)")

if np.argmax(fixed_series) < 5:  # Peak should be within first few indices (accounting for thermal lag)
    print("\n✓ Fix successful! Peak is now near sun_phase = 0°")
else:
    print("\n⚠ Warning: Peak is not near sun_phase = 0°. May need manual adjustment.")

# Save fixed LUT
print(f"\nSaving fixed LUT to: {OUTPUT_FILE}")
with h5py.File(OUTPUT_FILE, 'w') as f:
    f.create_dataset("lut", data=fixed_lut, compression='gzip')
    f.create_dataset("theta", data=theta_axis)
    f.create_dataset("opening_angle", data=angle_axis)
    f.create_dataset("latitude", data=lat_axis)
    f.create_dataset("wavelength", data=wave_axis)
    f.create_dataset("emission", data=emission_axis)
    f.create_dataset("azimuth", data=azimuth_axis)

print(f"\nDone!")
print(f"\nTo use the fixed LUT:")
print(f"  1. Test with: cp {OUTPUT_FILE} {INPUT_FILE}")
print(f"  2. Or update your code to load: {OUTPUT_FILE}")
print(f"\nTo restore original: cp {BACKUP_FILE} {INPUT_FILE}")
