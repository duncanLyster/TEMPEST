#!/usr/bin/env python3
"""
Debug the RoughnessLUT interpolation - why are we getting wrong values?
"""
import h5py
import numpy as np

LUT_PATH = "roughness_lut_spectral_v1.h5"

print("="*70)
print("DEBUG: RoughnessLUT Interpolation Issue")
print("="*70)

with h5py.File(LUT_PATH, 'r') as f:
    print(f"\nLUT File Structure:")
    print(f"  Keys: {list(f.keys())}")
    
    ratios = f['avg_ratios'][:]
    print(f"\n  avg_ratios shape: {ratios.shape}")
    print(f"  Axes: {f.attrs.get('axes', 'Not found')}")
    
    # Check what's at the thermal peak for disk center
    # Dimensions: (Theta, Angle, Lat, Time, Wave, Emission, Azimuth)
    # At disk center, emission=0, azimuth doesn't matter
    # At equator, lat=0
    # At thermal peak, time_idx=0 (after our fix)
    
    theta_idx = 0  # Smallest theta
    angle_idx = -1  # 90° opening angle
    lat_idx = ratios.shape[2] // 2  # Middle latitude (equator)
    time_idx = 0  # Thermal peak (after fix)
    emi_idx = 0  # Disk center
    azi_idx = 0  # Doesn't matter for disk center
    
    print(f"\nRAW LUT Values at thermal peak (t_idx=0), disk center:")
    print(f"  Indices: theta={theta_idx}, angle={angle_idx}, lat={lat_idx}, time={time_idx}, emi={emi_idx}, azi={azi_idx}")
    
    for wave_idx in range(ratios.shape[4]):
        val = ratios[theta_idx, angle_idx, lat_idx, time_idx, wave_idx, emi_idx, azi_idx]
        print(f"    Wave {wave_idx}: {val:.4f}")
    
    # Now check the axis values
    print(f"\nAxis Information:")
    for key in f.keys():
        if 'axis' in key or key.endswith('_axis'):
            axis_data = f[key][:]
            print(f"  {key}: {axis_data}")
    
    # Check available attributes
    print(f"\nFile Attributes:")
    for key in f.attrs.keys():
        print(f"  {key}: {f.attrs[key]}")

print("="*70)
