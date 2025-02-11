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

def create_lommel_seeliger_epf_lut(em_resolution=1):
    """
    Create Lommel-Seeliger emission phase function lookup table.
    The Lommel-Seeliger law shows more directional beaming than Lambertian,
    making it suitable for rough surfaces like the Moon or asteroids.
    
    EPF = 2 * cos(e) / (1 + cos(e))
    where e is the emission angle
    """
    # Create emission angle array
    emission_angles = np.linspace(0, 90, int(90/em_resolution) + 1)
    
    # Convert angles to radians for numpy calculations
    emission_radians = np.radians(emission_angles)
    
    # Calculate Lommel-Seeliger EPF
    cos_e = np.cos(emission_radians)
    table = 2 * cos_e / (1 + cos_e)
    
    # Convert to float32 for consistency
    table = table.astype(np.float32)
    
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
    lambertian_lut = create_lambertian_epf_lut(em_resolution=1)
    
    # Generate Lommel-Seeliger EPF LUT
    print("Generating Lommel-Seeliger EPF lookup table...")
    lommel_seeliger_lut = create_lommel_seeliger_epf_lut(em_resolution=1)
    
    # Save the LUTs using locations
    lambertian_path = locations.get_emission_lut_path('lambertian_epf.npy')
    lommel_seeliger_path = locations.get_emission_lut_path('lommel_seeliger_epf.npy')
    
    np.save(lambertian_path, lambertian_lut)
    np.save(lommel_seeliger_path, lommel_seeliger_lut)
    
    # Print memory usage and save locations
    for name, lut, path in [
        ("Lambertian", lambertian_lut, lambertian_path),
        ("Lommel-Seeliger", lommel_seeliger_lut, lommel_seeliger_path)
    ]:
        memory_mb = lut['table'].nbytes / (1024 * 1024)
        print(f"{name} LUT size: {memory_mb:.2f} MB")
        print(f"Saved {name} LUT to {path}")

if __name__ == "__main__":
    main()