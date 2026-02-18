#!/usr/bin/env python3
"""
Check if the LUT file we're loading has the fix applied
"""
import h5py
import numpy as np

print("="*70)
print("Checking Current LUT File")
print("="*70)

with h5py.File("roughness_lut_spectral_v1.h5", 'r') as f:
    lut_data = f['lut'][:]
    
    # Check disk center, equator, mid wavelength
    theta_idx = 0
    angle_idx = -1  # 90° opening angle
    lat_idx = lut_data.shape[2] // 2
    wave_idx = 2  # 15 microns
    emi_idx = 0  # Disk center
    azi_idx = 0
    
    time_series = lut_data[theta_idx, angle_idx, lat_idx, :, wave_idx, emi_idx, azi_idx]
    
    peak_idx = np.argmax(time_series)
    peak_value = time_series[peak_idx]
    value_at_0 = time_series[0]
    
    print(f"\nTime series at disk center, equator, 15 μm:")
    print(f"  Value at t_idx=0:  {value_at_0:.4f}")
    print(f"  Peak value:        {peak_value:.4f}")
    print(f"  Peak at t_idx:     {peak_idx}")
    print(f"  Peak sun_phase:    {peak_idx/len(time_series)*360:.0f}°")
    
    if peak_idx == 0:
        print("\n✓ LUT HAS THE FIX - peak is at t_idx=0")
    else:
        print(f"\n✗ LUT DOES NOT HAVE FIX - peak is at t_idx={peak_idx}")
        
    # Check all wavelengths at t_idx=0
    print(f"\nCorrection factors at t_idx=0 (should be peak):")
    for w_idx in range(lut_data.shape[4]):
        val = lut_data[theta_idx, angle_idx, lat_idx, 0, w_idx, emi_idx, azi_idx]
        print(f"  Wave {w_idx}: {val:.4f}")

print("="*70)
