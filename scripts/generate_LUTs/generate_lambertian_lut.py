# scripts/generate_LUTs/generate_lambertian_lut.py

"""
Generate Lambertian BRDF lookup table. Technically not a BRDF because of a missing factor of pi, but it's a lookup table for the Lambertian case.
This script creates and saves a lookup table for Lambertian scattering.
"""

import numpy as np
from pathlib import Path
import sys

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parents[1] / "src"
sys.path.append(str(src_dir))

from src.utilities.locations import Locations

def create_lambertian_lut(inc_resolution=1, em_resolution=1, az_resolution=1):
    """
    Create Lambertian BRDF lookup table.
    For a Lambertian surface, BRDF = 1/π (constant)
    """
    # Create angle arrays
    incidence_angles = np.linspace(0, 90, int(90/inc_resolution) + 1)
    emission_angles = np.linspace(0, 90, int(90/em_resolution) + 1)
    azimuth_angles = np.linspace(0, 180, int(180/az_resolution) + 1)
    
    # Create table with constant Lambertian BRDF value
    table = np.full((len(incidence_angles),
                    len(emission_angles),
                    len(azimuth_angles)),
                    1.0,
                    dtype=np.float32)
    
    return {
        'table': table,
        'incidence_angles': incidence_angles,
        'emission_angles': emission_angles,
        'azimuth_angles': azimuth_angles
    }

def main():
    # Initialize locations and ensure directories exist
    locations = Locations()
    locations.ensure_directories_exist()
    
    # Generate Lambertian LUT
    print("Generating Lambertian BRDF lookup table...")
    lut_data = create_lambertian_lut(inc_resolution=1,
                                    em_resolution=1,
                                    az_resolution=1)
    
    # Save the LUT using locations
    output_path = locations.get_scattering_lut_path('lambertian.npy')
    np.save(output_path, lut_data)
    
    # Print memory usage and save location
    memory_mb = lut_data['table'].nbytes / (1024 * 1024)
    print(f"LUT size: {memory_mb:.2f} MB")
    print(f"Saved LUT to {output_path}")

if __name__ == "__main__":
    main()