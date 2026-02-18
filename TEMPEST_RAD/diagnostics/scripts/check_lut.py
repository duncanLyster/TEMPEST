#!/usr/bin/env python3
import h5py
import numpy as np

print("Checking LUT file...")
with h5py.File('roughness_lut_spectral_v1.h5', 'r') as f:
    print("Datasets:", list(f.keys()))
    
    if 'lut' in f:
        lut_data = f['lut'][:]
        print(f"\nLUT Shape: {lut_data.shape}")
        
        # Dimensions: (Theta, Angle, Lat, Time, Wave, Emi, Azi)
        midday = lut_data.shape[3] // 2
        zero_phase = lut_data[0, 0, 0, midday, :, 0, 0]
        
        print(f"\nZero-phase correction factors (midday, e=0°, a=0°):")
        print(f"  Values: {zero_phase}")
        print(f"  Min: {np.nanmin(zero_phase):.6f}")
        print(f"  Max: {np.nanmax(zero_phase):.6f}")
        print(f"  Mean: {np.nanmean(zero_phase):.6f}")
        print(f"  Count > 1.0: {np.sum(zero_phase > 1.0)}/{len(zero_phase)}")
        
        # Check all times at zero phase
        all_zero_phase = lut_data[0, 0, 0, :, :, 0, 0]
        print(f"\nAll times at zero-phase (e=0°, a=0°):")
        print(f"  Min: {np.nanmin(all_zero_phase):.6f}")
        print(f"  Max: {np.nanmax(all_zero_phase):.6f}")
        print(f"  Mean: {np.nanmean(all_zero_phase):.6f}")
        print(f"  Count > 1.0: {np.sum(all_zero_phase > 1.0)}/{all_zero_phase.size}")
        
        # Overall stats
        all_valid = lut_data[~np.isnan(lut_data)]
        print(f"\nOverall LUT:")
        print(f"  Min: {np.nanmin(all_valid):.6f}")
        print(f"  Max: {np.nanmax(all_valid):.6f}")
        print(f"  Mean: {np.nanmean(all_valid):.6f}")
        print(f"  Count > 1.0: {np.sum(all_valid > 1.0)}/{len(all_valid)}")
