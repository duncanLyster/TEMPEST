#!/usr/bin/env python3
import h5py
import numpy as np

with h5py.File('roughness_lut_spectral_v1.h5', 'r') as f:
    lut = f['lut'][:]
    emission_angles = f['emission'][:]
    wavelengths = f['wavelength'][:]
    
    lat_idx = 0
    theta_idx = 0
    ang_idx = 0
    wave_idx = 1  # 8 microns
    
    print(f'Wavelength: {wavelengths[wave_idx]} um')
    print()
    print('Time-averaged disk-integrated correction factor:')
    print()
    
    n_time = lut.shape[3]
    time_disk_avgs = []
    
    for t_idx in range(n_time):
        # For each time, compute disk-averaged correction
        total_weight = 0.0
        total_weighted_factor = 0.0
        
        for e_idx, e_ang in enumerate(emission_angles):
            e_rad = np.radians(e_ang)
            # Average over azimuth
            avg_factor = 0.0
            for azi_idx in range(10):
                avg_factor += lut[theta_idx, ang_idx, lat_idx, t_idx, wave_idx, e_idx, azi_idx]
            avg_factor /= 10.0
            
            # Weight: sin(e) * cos(e) for area element  
            weight = np.sin(e_rad) * np.cos(e_rad)
            total_weight += weight
            total_weighted_factor += weight * avg_factor
        
        disk_avg = total_weighted_factor / total_weight
        time_disk_avgs.append(disk_avg)
    
    time_disk_avgs = np.array(time_disk_avgs)
    overall_avg = np.mean(time_disk_avgs)
    
    print(f'Time-averaged disk correction: {overall_avg:.4f}')
    print(f'Min across day: {np.min(time_disk_avgs):.4f}')
    print(f'Max across day: {np.max(time_disk_avgs):.4f}')
    print(f'Std across day: {np.std(time_disk_avgs):.4f}')
    print()
    print('(Should be ~1.0 if normalization is correct)')
    
    # Show a few time samples
    print()
    print('Sample times:')
    for t_idx in [0, 22, 45, 67]:
        print(f'  t={t_idx}/90: disk_avg={time_disk_avgs[t_idx]:.4f}')
