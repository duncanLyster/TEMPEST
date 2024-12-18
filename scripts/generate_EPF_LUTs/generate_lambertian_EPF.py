# scripts/generate_LUTs/generate_lambertian_lut.py

"""
Generate Lambertian EPF (Emission Phase Function) lookup table.
This script creates and saves a lookup table for Lambertian thermal emission.
For a Lambertian surface, the EPF is constant (1.0).
"""

import numpy as np
from pathlib import Path
import sys

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parents[1] / "src"
sys.path.append(str(src_dir))

from src.utilities.locations import Locations

def create_lambertian_epf_lut(em_resolution=1):
    """
    Create Lambertian emission lookup table.
    For a Lambertian surface, EPF = 1.0 (constant)
    """
    # Create emission angle array
    emission_angles = np.linspace(0, 90, int(90/em_resolution) + 1)
    
    # Create table with constant value of 1.0
    table = np.ones_like(emission_angles, dtype=np.float32)
    
    return {
        'table': table,
        'emission_angles': emission_angles
    }

def main():
    # Initialize locations and ensure directories exist
    locations = Locations()
    locations.ensure_directories_exist()
    
    # Generate Lambertian EPF LUT
    print("Generating Lambertian EPF lookup table...")
    lut_data = create_lambertian_epf_lut(em_resolution=1)
    
    # Save the LUT using locations
    output_path = locations.get_emission_lut_path('lambertian_epf.npy')
    np.save(output_path, lut_data)
    
    # Print memory usage and save location
    memory_mb = lut_data['table'].nbytes / (1024 * 1024)
    print(f"LUT size: {memory_mb:.2f} MB")
    print(f"Saved LUT to {output_path}")

if __name__ == "__main__":
    main()