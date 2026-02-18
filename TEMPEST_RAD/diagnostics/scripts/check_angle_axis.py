#!/usr/bin/env python3
"""
Check the opening angle axis values
"""
import h5py
import numpy as np

with h5py.File("roughness_lut_spectral_v1.h5", 'r') as f:
    opening_angles = f['opening_angle'][:]
    lut = f['lut'][:]
    
    print("="*70)
    print("Opening Angle Axis")
    print("="*70)
    print(f"\nOpening angles: {opening_angles}")
    print(f"Shape: {opening_angles.shape}")
    
    # Check values at each angle for disk center, equator, noon, 15 microns
    theta_idx = 0
    lat_idx = 0
    time_idx = 0
    wave_idx = 2  # 15 microns
    emi_idx = 0
    azi_idx = 0
    
    print(f"\nCorrection factors at disk center, noon, 15μm:")
    for ang_idx, angle in enumerate(opening_angles):
        val = lut[theta_idx, ang_idx, lat_idx, time_idx, wave_idx, emi_idx, azi_idx]
        print(f"  Opening angle {angle:5.1f}° (idx={ang_idx}): factor = {val:.4f}")
    
    # The RoughnessLUT class selects idx_rms based on target_rms
    # It's looking for opening_angle ≈ 90°
    target_rms = 90.0
    idx_selected = np.abs(opening_angles - target_rms).argmin()
    print(f"\nFor target_rms=90°, RoughnessLUT selects:")
    print(f"  idx={idx_selected}, angle={opening_angles[idx_selected]}°")
    print(f"  factor at this angle: {lut[theta_idx, idx_selected, lat_idx, time_idx, wave_idx, emi_idx, azi_idx]:.4f}")

print("="*70)
