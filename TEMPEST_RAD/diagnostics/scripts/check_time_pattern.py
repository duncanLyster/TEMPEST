#!/usr/bin/env python3
"""
Check correction factors at different local times to understand the pattern.
"""

import h5py
import numpy as np

with h5py.File('roughness_lut_spectral_v1.h5', 'r') as f:
    lut = f['lut'][:]
    emission_angles = f['emission'][:]
    wavelengths = f['wavelength'][:]
    
n_time = lut.shape[3]  # Should be 90
sun_phases = np.linspace(0, 360, n_time, endpoint=False)

# Check correction factor at disk center (e=0, a=0) across different times
lat_idx = 0
theta_idx = 0  
ang_idx = 0
wave_idx = 1  # 8 microns
e_idx = 0  # emission = 0°
azi_idx = 0  # azimuth = 0°

print(f"Correction factors at disk center (e=0°, a=0°, 8μm) vs local time:")
print(f"")
print(f"Time Idx | Sun Phase | Factor | Interpretation")
print(f"-" * 60)

for t_idx in [0, 22, 45, 67, 89]:
    factor = lut[theta_idx, ang_idx, lat_idx, t_idx, wave_idx, e_idx, azi_idx]
    phase_deg = sun_phases[t_idx]
    
    # Interpret the local time
    if phase_deg < 45:
        time_str = "Early morning"
    elif phase_deg < 135:
        time_str = "Late morning/noon"
    elif phase_deg < 225:
        time_str = "Afternoon"
    elif phase_deg < 315:
        time_str = "Evening"
    else:
        time_str = "Night/dawn"
    
    print(f"  {t_idx:3d}    |  {phase_deg:6.1f}°  | {factor:.3f}  | {time_str}")

print()
print("Physical interpretation:")
print("  - sun_phase = 0° means facet normal points AT sun (local noon)")
print("  - sun_phase = 180° means facet normal points AWAY from sun (midnight)")
print()
print("Expected pattern for craters:")
print("  - Around noon (phase~0°): Should have highest enhancement")
print("  - Around dawn/dusk: Lower values due to shadows")  
print("  - Midnight: Minimal values (shadowed)")
