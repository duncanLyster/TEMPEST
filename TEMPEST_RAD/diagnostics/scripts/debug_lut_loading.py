#!/usr/bin/env python3
"""
Trace through RoughnessLUT loading step-by-step
"""
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

LUT_PATH =  "roughness_lut_spectral_v1.h5"

print("="*70)
print("DEBUG: RoughnessLUT Loading")
print("="*70)

# Manually replicate what RoughnessLUT.load_subset does
with h5py.File(LUT_PATH, 'r') as f:
    # Load Axes
    theta_axis = f['theta'][:]
    rms_axis = f['opening_angle'][:]
    lat_axis = f['latitude'][:]
    wave_axis = f['wavelength'][:]
    emi_axis = f['emission'][:]
    azi_axis = f['azimuth'][:]
    
    print(f"\nAxis shapes:")
    print(f"  theta: {theta_axis.shape} = {theta_axis}")
    print(f"  rms/angle: {rms_axis.shape} = {rms_axis}")
    print(f"  lat: {lat_axis.shape}, range [{lat_axis.min()}, {lat_axis.max()}]")
    print(f"  wave: {wave_axis.shape} = {wave_axis}")
    print(f"  emission: {emi_axis.shape}, range [{emi_axis.min()}, {emi_axis.max()}]")
    print(f"  azimuth: {azi_axis.shape}, range [{azi_axis.min()}, {azi_axis.max()}]")
    
    # Select indices (as RoughnessLUT does)
    target_theta = 0.02763  # From calculate_theta
    target_rms = 90.0
    
    idx_theta = np.abs(theta_axis - target_theta).argmin()
    idx_rms = np.abs(rms_axis - target_rms).argmin()
    
    print(f"\nSelected indices:")
    print(f"  theta: idx={idx_theta}, value={theta_axis[idx_theta]}")
    print(f"  rms:   idx={idx_rms}, value={rms_axis[idx_rms]}")
    
    # Load full LUT
    lut_full = f['lut'][:]
    print(f"\nFull LUT shape: {lut_full.shape}")
    print(f"  Dimensions: (Theta, Angle, Lat, Time, Wave, Emission, Azimuth)")
    
    # Slice (spectral mode - keep wavelength)
    # [th, ang, :, :, :, :, :]
    lut_subset = lut_full[idx_theta, idx_rms, :, :, :, :, :]
    print(f"\nSubset shape after slicing [theta={idx_theta}, angle={idx_rms}, :, :, :, :, :]:")
    print(f"  {lut_subset.shape}")
    print(f"  Remaining dimensions: (Lat, Time, Wave, Emission, Azimuth)")
    
    # Create sun_phases axis
    n_time = lut_subset.shape[1]
    sun_phases = np.linspace(0, 360, n_time, endpoint=False)
    print(f"\nSun phases axis: {len(sun_phases)} steps")
    print(f"  First 5: {sun_phases[:5]}")
    print(f"  Last 5: {sun_phases[-5:]}")
    
    # Check interpolator setup
    points = (lat_axis, sun_phases, wave_axis, emi_axis, azi_axis)
    print(f"\nInterpolator points (axis lengths):")
    for i, name in enumerate(['lat', 'sun_phase', 'wave', 'emission', 'azimuth']):
        print(f"  {name}: {len(points[i])}")
    
    # Check a specific value
    lat_idx_test = 0
    time_idx_test = 0
    wave_idx_test = 2  # 15 microns
    emi_idx_test = 0
    azi_idx_test = 0
    
    value_direct = lut_subset[lat_idx_test, time_idx_test, wave_idx_test, emi_idx_test, azi_idx_test]
    print(f"\nDirect access to subset[0, 0, 2, 0, 0]:")
    print(f"  Value: {value_direct:.4f}")
    print(f"  (This is lat=0°, time=0°, wave=15μm, emi=0°, azi=0°)")
    
    # Now test interpolator
    interpolator = RegularGridInterpolator(points, lut_subset, bounds_error=False, fill_value=None)
    
    # Query at exact grid point
    query_lat = lat_axis[lat_idx_test]
    query_phase = sun_phases[time_idx_test]
    query_wave = wave_axis[wave_idx_test]
    query_emi = emi_axis[emi_idx_test]
    query_azi = azi_axis[azi_idx_test]
    
    query_point = np.array([[query_lat, query_phase, query_wave, query_emi, query_azi]])
    value_interp = interpolator(query_point)[0]
    
    print(f"\nInterpolator query at same point:")
    print(f"  Query: lat={query_lat}°, phase={query_phase}°, wave={query_wave}μm, emi={query_emi}°, azi={query_azi}°")
    print(f"  Value: {value_interp:.4f}")
    
    if np.abs(value_direct - value_interp) < 0.001:
        print(f"\n✓ Direct and interpolated values match!")
    else:
        print(f"\n✗ MISMATCH: direct={value_direct:.4f}, interp={value_interp:.4f}")

print("="*70)
