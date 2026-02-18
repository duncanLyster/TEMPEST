#!/usr/bin/env python3
"""
Check LUT latitude structure and values at polar latitudes
"""
import h5py
import numpy as np

with h5py.File("roughness_lut_spectral_v1.h5", 'r') as f:
    lat_axis = f['latitude'][:]
    lut = f['lut'][:]
    
    print("="*70)
    print("LUT Latitude Analysis")
    print("="*70)
    
    print(f"\nLatitude axis: {lat_axis}")
    print(f"Number of latitude bins: {len(lat_axis)}")
    
    # Check correction factors at each latitude
    # Dimensions: (Theta, Angle, Lat, Time, Wave, Emission, Azimuth)
    theta_idx = 0
    angle_idx = 0  
    wave_idx = 2  # 15 microns
    emi_idx = 0   # Disk center
    azi_idx = 0
    
    print(f"\nCorrection factors at disk center, 15μm:")
    print(f"Averaged over time (full rotation):\n")
    
    for lat_idx, lat in enumerate(lat_axis):
        time_series = lut[theta_idx, angle_idx, lat_idx, :, wave_idx, emi_idx, azi_idx]
        
        mean_factor = np.mean(time_series)
        max_factor = np.max(time_series)
        min_factor = np.min(time_series)
        
        print(f"  Lat={lat:5.1f}°: mean={mean_factor:.4f}, max={max_factor:.4f}, min={min_factor:.4f}")
        
        if mean_factor < 0.95:
            print(f"           ⚠️  Time-averaged factor < 1.0!")
    
    # Check if normalization was done per-latitude
    print(f"\nChecking normalization (should be ~1.0 per latitude):")
    print(f"Time-averaged, hemisphere-integrated correction:\n")
    
    for lat_idx, lat in enumerate(lat_axis):
        # Average over time and viewing geometry
        # Weight by viewing geometry (emission)
        time_avg_factors = []
        
        for t_idx in range(lut.shape[3]):
            # Hemisphere integral
            integral = 0.0
            norm = 0.0
            
            for e_idx in range(lut.shape[5]):
                for a_idx in range(lut.shape[6]):
                    factor = lut[theta_idx, angle_idx, lat_idx, t_idx, wave_idx, e_idx, a_idx]
                    # Simple average (could weight by sin(e)*cos(e))
                    integral += factor
                    norm += 1.0
            
            time_avg_factors.append(integral / norm if norm > 0 else 1.0)
        
        overall_avg = np.mean(time_avg_factors)
        print(f"  Lat={lat:5.1f}°: {overall_avg:.4f}")

print("="*70)
