#!/usr/bin/env python3
"""
Verify the sun_phase axis offset bug.
"""

import h5py
import numpy as np

with h5py.File('roughness_lut_spectral_v1.h5', 'r') as f:
    lut = f['lut'][:]
    
n_time = lut.shape[3]  # 90
sun_phases = np.linspace(0, 360, n_time, endpoint=False)

# Check when correction factors are highest
lat_idx, theta_idx, ang_idx = 0, 0, 0
wave_idx = 1  # 8 microns
e_idx, azi_idx = 0, 0

print("="*70)
print("BUG ANALYSIS: Sun Phase Offset in LUT")
print("="*70)
print()

factors_vs_time = []
for t_idx in range(n_time):
    factor = lut[theta_idx, ang_idx, lat_idx, t_idx, wave_idx, e_idx, azi_idx]
    factors_vs_time.append(factor)

factors_vs_time = np.array(factors_vs_time)
max_idx = np.argmax(factors_vs_time)
min_idx = np.argmin(factors_vs_time)

print(f"Maximum correction factor: {factors_vs_time[max_idx]:.3f} at t_idx={max_idx}")
print(f"  Corresponding sun_phase in LUT: {sun_phases[max_idx]:.1f}°")
print(f"  This should be LOCAL NOON (when crater is hottest/brightest)")
print()

print(f"Minimum correction factor: {factors_vs_time[min_idx]:.3f} at t_idx={min_idx}")
print(f"  Corresponding sun_phase in LUT: {sun_phases[min_idx]:.1f}°")
print(f"  This should be NIGHT/DAWN (when crater is coolest/dimmest)")
print() 

print("EXPECTED behavior:")
print(f"  - sun_phase = 0° should correspond to LOCAL NOON")
print(f"  - sun_phase = 180° should correspond to LOCAL MIDNIGHT")
print()

print("ACTUAL behavior from LUT:")
print(f"  - sun_phase = {sun_phases[max_idx]:.0f}° has MAX factor (brightest → NOON)")
print(f"  - sun_phase = {sun_phases[min_idx]:.0f}° has MIN factor (dimmest → MIDNIGHT)")
print()

offset = sun_phases[max_idx]
print(f"DIAGNOSIS:")
if abs(offset) < 45:
    print(f"  ✓ LUT axes are correct (max at ~0°)")
elif abs(offset - 180) < 45:
    print(f"  ✗ LUT has 180° OFFSET BUG!")
    print(f"     When retrieval asks for sun_phase=0° (noon), ")
    print(f"     it gets t_idx={0} which is actually MIDNIGHT!")
    print()
    print(f"  FIX: In generator.py, change:")
    print(f"       sun_phases = np.linspace(0, 360, n_time, endpoint=False)")
    print(f"  TO:")
    print(f"       sun_phases = np.linspace(180, 540, n_time, endpoint=False) % 360")
    print(f"  OR rearrange the LUT data to start at noon instead of midnight")
else:
    print(f"  ? Unexpected pattern - needs investigation")
