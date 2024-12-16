# src/model/scattering.py

import numpy as np
from pathlib import Path
from src.utilities.locations import Locations
import os

class BRDFLookupTable:
    def __init__(self, lut_name):
        locations = Locations()
        self.lut_path = locations.get_scattering_lut_path(lut_name)
        
        if not os.path.exists(self.lut_path):
            raise FileNotFoundError(f"BRDF lookup table not found at {self.lut_path}")
                
        try:
            # Load the LUT and angle arrays
            lut_data = np.load(self.lut_path, allow_pickle=True).item()
            
            self.table = lut_data['table'].astype(np.float32)
            self.incidence_angles = lut_data['incidence_angles']
            self.emission_angles = lut_data['emission_angles']
            self.azimuth_angles = lut_data['azimuth_angles']
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BRDF lookup table: {str(e)}")
            
    def query(self, inc, em, az):
        # Ensure angles are within bounds
        az = az % 180  # Use symmetry
        
        # Get interpolation indices
        i = np.searchsorted(self.incidence_angles, inc) - 1
        e = np.searchsorted(self.emission_angles, em) - 1
        a = np.searchsorted(self.azimuth_angles, az) - 1
        
        # Perform trilinear interpolation
        return self._interpolate(inc, em, az, i, e, a)
    
    def _interpolate(self, inc, em, az, i, e, a):
        """Implementation of trilinear interpolation.
        
        Args:
            inc (float): Incidence angle in degrees
            em (float): Emission angle in degrees 
            az (float): Azimuth angle in degrees
            i (int): Lower index for incidence angle interpolation
            e (int): Lower index for emission angle interpolation
            a (int): Lower index for azimuth angle interpolation
            
        Returns:
            float: Interpolated BRDF value
        """
        # Get weights for interpolation
        wi = (inc - self.incidence_angles[i]) / (self.incidence_angles[i+1] - self.incidence_angles[i])
        we = (em - self.emission_angles[e]) / (self.emission_angles[e+1] - self.emission_angles[e])
        wa = (az - self.azimuth_angles[a]) / (self.azimuth_angles[a+1] - self.azimuth_angles[a])
        
        # Get the 8 surrounding points
        c000 = self.table[i, e, a]
        c001 = self.table[i, e, a+1]
        c010 = self.table[i, e+1, a]
        c011 = self.table[i, e+1, a+1]
        c100 = self.table[i+1, e, a]
        c101 = self.table[i+1, e, a+1]
        c110 = self.table[i+1, e+1, a]
        c111 = self.table[i+1, e+1, a+1]
        
        # Perform trilinear interpolation
        c00 = c000 * (1-wi) + c100 * wi
        c01 = c001 * (1-wi) + c101 * wi
        c10 = c010 * (1-wi) + c110 * wi
        c11 = c011 * (1-wi) + c111 * wi
        
        c0 = c00 * (1-we) + c10 * we
        c1 = c01 * (1-we) + c11 * we
        
        return c0 * (1-wa) + c1 * wa