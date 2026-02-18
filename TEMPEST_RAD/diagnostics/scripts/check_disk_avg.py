#!/usr/bin/env python3
import h5py
import numpy as np

with h5py.File('roughness_lut_spectral_v1.h5', 'r') as f:
    lut = f['lut'][:]
    emission_angles = f['emission'][:]
    wavelengths = f['wavelength'][:]
    
    # At midday, lat=0, check if disk-integrated correction is ~1.0
    midday_idx = 45
    lat_idx = 0
    theta_idx = 0
    ang_idx = 0
    wave_idx = 1  # 8 microns
    
    print(f'Wavelength: {wavelengths[wave_idx]} um')
    print()
    print('Computing disk-averaged correction factor (weighted by projected area):')
    print()
    
    # For disk-integrated obs, weight by cos(e) * sin(e) for solid angle element
    total_weight = 0.0
    total_weighted_factor = 0.0
    
    for e_idx, e_ang in enumerate(emission_angles):
        e_rad = np.radians(e_ang)
        # Average over azimuth
        avg_factor = 0.0
        for azi_idx in range(10):
            avg_factor += lut[theta_idx, ang_idx, lat_idx, midday_idx, wave_idx, e_idx, azi_idx]
        avg_factor /= 10.0
        
        # Weight: sin(e) * cos(e) for area element  
        weight = np.sin(e_rad) * np.cos(e_rad)
        total_weight += weight
        total_weighted_factor += weight * avg_factor
        
        print(f'e={e_ang:5.1f}deg: avg_factor={avg_factor:.3f}, weight={weight:.4f}, contrib={weight*avg_factor:.4f}')
    
    disk_avg = total_weighted_factor / total_weight
    print()
    print(f'Disk-averaged correction factor: {disk_avg:.4f}')
    print(f'(Should be ~1.0 if normalization is correct)')
