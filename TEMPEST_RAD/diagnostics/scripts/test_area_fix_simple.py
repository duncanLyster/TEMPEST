#!/usr/bin/env python3
"""Simple test of the area fix"""
import sys
sys.path.insert(0, '/Users/duncan/Desktop/DPhil/TEMPEST')

import numpy as np
from TEMPEST_RAD.generator import process_single_case, ReferenceConfig

print("Testing area fix...")
print("=" * 60)

config = ReferenceConfig()
result = process_single_case(theta=10.0, opening_angle=90.0, lat=0.0, config=config)

print(f"\nResult shape: {result.shape}")
print(f"Valid entries: {np.sum(~np.isnan(result))}/{result.size}")

# Zero-phase statistics
zero_phase = result[:, :, 0, 0]
print(f"\nZero-phase (e=0°, a=0°):")
print(f"  Min: {np.nanmin(zero_phase):.4f}")
print(f"  Max: {np.nanmax(zero_phase):.4f}")
print(f"  Mean: {np.nanmean(zero_phase):.4f}")
print(f"  > 1.0: {np.sum(zero_phase > 1.0)}/{zero_phase.size}")

# All geometries at midday
midday = result[result.shape[0]//2]
print(f"\nMidday all angles:")
print(f"  Min: {np.nanmin(midday):.4f}")
print(f"  Max: {np.nanmax(midday):.4f}")
print(f"  Mean: {np.nanmean(midday):.4f}")
print(f"  > 1.0: {np.sum(midday > 1.0)}/{midday.size}")

print("\n" + "=" * 60)
