#!/usr/bin/env python3
"""Test script to verify the flat facet area fix"""
import sys
import os
sys.path.insert(0, '/Users/duncan/Desktop/DPhil/TEMPEST')
os.chdir('/Users/duncan/Desktop/DPhil/TEMPEST/TEMPEST_RAD')
import numpy as np
from TEMPEST_RAD.generator import process_single_case
from TEMPEST_RAD.config_rad import create_config

print("Testing single case with fixed flat facet area...")
print("=" * 60)

# Test case: theta=10, opening_angle=90, latitude=0
config = create_config()
result = process_single_case(theta=10.0, opening_angle=90.0, lat=0.0, config=config)

if result is None or np.all(np.isnan(result)):
    print("ERROR: Result is None or all NaN")
else:
    print(f"Result shape: {result.shape}")
    print(f"Valid entries: {np.sum(~np.isnan(result))}/{result.size}")
    
    # Check zero-phase values (emission=0, azimuth=0, various times and wavelengths)
    zero_phase_values = result[:, :, 0, 0]  # All times, all wavelengths, emission=0, azimuth=0
    
    print(f"\nZero-phase correction factors (e=0°, a=0°):")
    print(f"  Min: {np.nanmin(zero_phase_values):.4f}")
    print(f"  Max: {np.nanmax(zero_phase_values):.4f}")
    print(f"  Mean: {np.nanmean(zero_phase_values):.4f}")
    print(f"  Median: {np.nanmedian(zero_phase_values):.4f}")
    
    # Check if any values > 1.0 (beaming effect)
    n_above_one = np.sum(zero_phase_values > 1.0)
    print(f"  Values > 1.0: {n_above_one}/{zero_phase_values.size}")
    
    # Show sample values at midday (t=LUT_TIMESTEPS//2)
    midday_idx = result.shape[0] // 2
    print(f"\nMidday (t={midday_idx}) zero-phase values by wavelength:")
    for w_idx, wave in enumerate([8.0, 10.0, 12.0, 15.0, 20.0]):
        val = result[midday_idx, w_idx, 0, 0]
        print(f"  λ={wave:4.1f} μm: {val:.4f}")
    
    # Check all viewing geometries at midday
    all_values = result[midday_idx, :, :, :]
    print(f"\nAll viewing geometries at midday:")
    print(f"  Min: {np.nanmin(all_values):.4f}")
    print(f"  Max: {np.nanmax(all_values):.4f}")
    print(f"  Mean: {np.nanmean(all_values):.4f}")
    print(f"  Values > 1.0: {np.sum(all_values > 1.0)}/{all_values.size}")

print("\n" + "=" * 60)
print("Test complete!")
