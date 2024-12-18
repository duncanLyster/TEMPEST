# src/model/emission.py

import numpy as np
import os
from src.utilities.locations import Locations

class EPFLookupTable:
    def __init__(self, lut_name):
        locations = Locations()
        self.lut_path = locations.get_emission_lut_path(lut_name)
        
        if not os.path.exists(self.lut_path):
            raise FileNotFoundError(f"EPF lookup table not found at {self.lut_path}")
                
        try:
            # Load the LUT and angle arrays
            lut_data = np.load(self.lut_path, allow_pickle=True).item()
            
            self.table = lut_data['table'].astype(np.float32)
            self.emission_angles = lut_data['emission_angles']
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EPF lookup table: {str(e)}")
            
    def query(self, em):
        """Query the EPF value for a given emission angle."""
        # Get interpolation index
        e = np.searchsorted(self.emission_angles, em) - 1
        
        # Handle edge cases
        if e < 0:
            return self.table[0]
        if e >= len(self.emission_angles) - 1:
            return self.table[-1]
            
        # Linear interpolation
        we = (em - self.emission_angles[e]) / (self.emission_angles[e+1] - self.emission_angles[e])
        return self.table[e] * (1-we) + self.table[e+1] * we 