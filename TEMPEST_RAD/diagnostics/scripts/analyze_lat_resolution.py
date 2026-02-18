#!/usr/bin/env python3
"""
Investigate if low latitude resolution causes interpolation artifacts
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("LUT Latitude Resolution Analysis")
print("="*70)

with h5py.File("roughness_lut_spectral_v1.h5", 'r') as f:
    lat_axis = f['latitude'][:]
    lut = f['lut'][:]
    
    print(f"\nCurrent latitude axis: {lat_axis}")
    print(f"Gaps: {np.diff(lat_axis)}°")
    
    # Check values at grid points
    theta_idx = 0
    angle_idx = 0
    time_idx = 0  # Noon
    wave_idx = 2  # 15 microns
    emi_idx = 0   # Disk center
    azi_idx = 0
    
    print(f"\n--- Raw LUT Values at Grid Points (Noon, Disk Center, 15μm) ---")
    for lat_idx, lat in enumerate(lat_axis):
        val = lut[theta_idx, angle_idx, lat_idx, time_idx, wave_idx, emi_idx, azi_idx]
        print(f"  Lat={lat:5.1f}°: factor={val:.4f}")
    
    # Now test interpolation at intermediate points
    print(f"\n--- Interpolated Values (Linear) Between Grid Points ---")
    
    # Test interpolation in each gap
    gaps = [
        (0, 30, "Equatorial"),
        (30, 60, "Mid-latitude"),
        (60, 85, "High-latitude")
    ]
    
    for lat_start, lat_end, name in gaps:
        idx_start = np.where(lat_axis == lat_start)[0][0]
        idx_end = np.where(lat_axis == lat_end)[0][0]
        
        val_start = lut[theta_idx, angle_idx, idx_start, time_idx, wave_idx, emi_idx, azi_idx]
        val_end = lut[theta_idx, angle_idx, idx_end, time_idx, wave_idx, emi_idx, azi_idx]
        
        print(f"\n{name} gap ({lat_start}° to {lat_end}°):")
        print(f"  Start: {val_start:.4f} | End: {val_end:.4f} | Drop: {val_start - val_end:.4f}")
        
        # Sample intermediate points
        test_lats = np.linspace(lat_start, lat_end, 7)[1:-1]  # Exclude endpoints
        for test_lat in test_lats:
            # Linear interpolation
            frac = (test_lat - lat_start) / (lat_end - lat_start)
            interp_val = val_start + frac * (val_end - val_start)
            print(f"    {test_lat:5.1f}°: {interp_val:.4f}")

# Physical explanation
print(f"\n{'='*70}")
print("PHYSICAL INTERPRETATION:")
print("='*70}")
print("""
At HIGH latitudes (>60°), the sun is at a GRAZING angle:
  - Smooth surface: Low flux due to cos(zenith) effect
  - Rough surface: EVEN LOWER flux due to shadowing in craters
  - Result: Rough < Smooth (factors < 1.0) is PHYSICALLY CORRECT

This is NOT a beaming effect - it's the opposite!
Beaming (rough > smooth) only occurs when you can see illuminated crater walls.
At polar latitudes, you mostly see shadows.

HOWEVER: With only 4 latitude points, the interpolator might:
1. Misplace the beaming→shadowing transition
2. Underestimate values in gaps if transition is nonlinear
3. Create artifacts for facets between grid points

SOLUTION: Generate higher-resolution LUT with more latitude samples:
  - Current: [0, 30, 60, 85] (4 points, 25-30° gaps)
  - Suggested: [0, 15, 30, 45, 60, 75, 85, 90] (8 points, 15° gaps)
  - Or even: 0° to 90° in 5° steps (19 points)
""")
print("="*70)
