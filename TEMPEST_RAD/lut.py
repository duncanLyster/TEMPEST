"""
Roughness Lookup Table (LUT) Handler.

This module manages the loading and interpolation of the 7D High-Fidelity Roughness LUT.
Dimensions: (Theta, OpeningAngle, Latitude, SunPhase, Wavelength, Emission, Azimuth)
"""

import numpy as np
import h5py
import os
from scipy.interpolate import RegularGridInterpolator
from src.utilities.utils import conditional_print

class RoughnessLUT:
    def __init__(self, lut_path, target_theta=None, target_rms=None, target_wavelength=None):
        self.lut_path = lut_path
        self.interpolator = None
        self.axes = {}
        self.is_loaded = False
        
        # Load only the subset if targets are provided
        if os.path.exists(lut_path):
            self.load_subset(target_theta, target_rms, target_wavelength)
        else:
            print(f"Warning: Roughness LUT not found at {lut_path}")

    def load_subset(self, target_theta, target_rms, target_wavelength):
        """
        Load a specific subset of the LUT matching the target parameters.
        This saves RAM by avoiding loading the entire 7D tensor.
        """
        try:
            with h5py.File(self.lut_path, 'r') as f:
                # Load Axes
                theta_axis = f['theta'][:]
                wave_axis = f['wavelength'][:]
                lat_axis = f['latitude'][:]
                emi_axis = f['emission'][:]
                azi_axis = f['azimuth'][:]
                
                # Find indices
                # Theta
                if target_theta is not None:
                    idx_theta = np.abs(theta_axis - target_theta).argmin()
                    print(f"LUT: Selected Theta={theta_axis[idx_theta]} (Target={target_theta})")
                else:
                    idx_theta = 0 # Default
                    
                # RMS / Opening Angle
                if 'opening_angle' in f:
                    rms_axis = f['opening_angle'][:]
                    print("LUT: Using 'opening_angle' axis.")
                elif 'rms' in f:
                    rms_axis = f['rms'][:]
                    print("LUT: Using 'rms' axis.")
                else:
                    raise KeyError("LUT missing 'opening_angle' or 'rms' dataset.")

                if target_rms is not None:
                    idx_rms = np.abs(rms_axis - target_rms).argmin()
                    print(f"LUT: Selected Angle={rms_axis[idx_rms]} (Target={target_rms})")
                else:
                    idx_rms = 0
                    
                # Wavelength
                # Generator Dimensions: (Theta, Angle, Lat, Time, Wave, Emi, Azi)
                #                        0      1      2     3     4     5    6
                
                if target_wavelength is not None:
                    # Find closest wavelength index
                    idx_wave = np.abs(wave_axis - target_wavelength).argmin()
                    print(f"LUT: Selected Wave={wave_axis[idx_wave]} (Target={target_wavelength})")
                    
                    if 'correction_factors' in f:
                        dset_name = 'correction_factors'
                    else:
                        dset_name = 'lut'
                        
                    # Slice: [th, ang, :, :, wave, :, :]
                    # Remaining: (Lat, Time, Emi, Azi)
                    # Note: h5py supports this slicing
                    lut_subset = f[dset_name][idx_theta, idx_rms, :, :, idx_wave, :, :]
                    
                    # Points for Interpolator: (Lat, Time, Emi, Azi)
                    n_time = lut_subset.shape[1]
                    sun_phases = np.linspace(0, 360, n_time, endpoint=False)
                    
                    points = (lat_axis, sun_phases, emi_axis, azi_axis)
                    self.spectral_mode = False
                    
                else:
                    # Spectral Mode: Keep Wavelength
                    # Slice: [th, ang, :, :, :, :, :]
                    # Remaining: (Lat, Time, Wave, Emi, Azi)
                    if 'correction_factors' in f:
                        dset_name = 'correction_factors'
                    else:
                        dset_name = 'lut'
                        
                    lut_subset = f[dset_name][idx_theta, idx_rms, :, :, :, :, :]
                    
                    n_time = lut_subset.shape[1]
                    sun_phases = np.linspace(0, 360, n_time, endpoint=False)
                    
                    points = (lat_axis, sun_phases, wave_axis, emi_axis, azi_axis)
                    self.spectral_mode = True
                    self.axes['wavelength'] = wave_axis

            # Handle NaNs
            if np.isnan(lut_subset).any():
                lut_subset = np.nan_to_num(lut_subset, nan=1.0)
                
            self.interpolator = RegularGridInterpolator(
                points, lut_subset, bounds_error=False, fill_value=None
            )
            self.is_loaded = True
            
        except Exception as e:
            print(f"Error loading Roughness LUT subset: {e}")
            self.is_loaded = False

    def get_correction_factors(self, latitudes, sun_phases, emissions, azimuths, wavelength=None):
        """
        Query the LUT.
        """
        if not self.is_loaded:
            return np.ones_like(emissions)
            
        # Wrap phases
        sun_phases = np.mod(sun_phases, 360.0)
        
        # Mirror Azimuths
        azimuths = np.abs(azimuths)
        azimuths = np.where(azimuths > 180, 360 - azimuths, azimuths)
        
        # Clip Latitudes (just in case)
        latitudes = np.abs(latitudes)
        
        # Cap emission angle to avoid limb divergence.
        # The ratio R = I_rough / I_smooth diverges as emission -> 90° because
        # the Lambertian denominator cos(e) -> 0 while the rough radiance stays
        # finite.  The last LUT grid point (e=89°) absorbs this divergence and
        # has values 10-90x, which are mathematically correct for the ratio but
        # create unphysical per-pixel temperatures.  Capping at 80° keeps us in
        # the well-behaved regime.  Limb facets beyond 80° contribute
        # negligibly to disk-integrated flux (cos 80° = 0.17).
        EMI_CAP = 80.0
        emissions = np.minimum(emissions, EMI_CAP)
        
        if self.spectral_mode:
            if wavelength is None:
                # Default to first wavelength if none provided
                wavelength = self.axes['wavelength'][0]
                
            # Broadcast wavelength to match other arrays
            n = len(latitudes)
            waves = np.full(n, wavelength)
            
            # Query: (Lat, Time, Wave, Emi, Azi)
            query_points = np.column_stack((
                latitudes, sun_phases, waves, emissions, azimuths
            ))
        else:
            # Fixed wavelength mode
            # Query: (Lat, Time, Emi, Azi)
            query_points = np.column_stack((
                latitudes, sun_phases, emissions, azimuths
            ))
            
        return self.interpolator(query_points)
