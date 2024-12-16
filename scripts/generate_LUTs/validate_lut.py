"""
Validate BRDF lookup tables for energy conservation.
For each incidence angle, integrating over all emission angles and azimuths should give 1.
"""

import numpy as np
from pathlib import Path
import sys

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parents[1] / "src"
sys.path.append(str(src_dir))

from src.utilities.locations import Locations
from src.model.scattering import BRDFLookupTable

def validate_brdf_lut_integration(lut_name):
    """Validate BRDF LUT using direct integration with trapezoidal rule."""
    locations = Locations()
    lut_path = locations.get_scattering_lut_path(lut_name)
    lut_data = np.load(lut_path, allow_pickle=True).item()
    
    # Extract data
    table = lut_data['table']
    emission_angles = lut_data['emission_angles']
    azimuth_angles = lut_data['azimuth_angles']
    
    # Convert angles to radians
    em_rad = np.radians(emission_angles)
    az_rad = np.radians(azimuth_angles)
    
    # Create meshgrid
    em_mesh, az_mesh = np.meshgrid(em_rad, az_rad, indexing='ij')
    
    print("\nValidating using direct integration (doubled):")
    for i, inc in enumerate(lut_data['incidence_angles']):
        # Prepare integrand
        integrand = table[i] * np.cos(em_mesh) * np.sin(em_mesh)
        
        # Double trapz integration (factor of 2 for azimuthal symmetry)
        integral = 2 * np.trapz(
            np.trapz(integrand, x=az_rad, axis=1),
            x=em_rad
        )
        
        print(f"Incidence angle {inc:5.1f}°: Integral = {integral:.6f} (should be 1.0)")

def validate_brdf_lut_monte_carlo(lut_name, n_samples=10000):
    """Validate BRDF LUT using Monte Carlo sampling of random angles."""
    brdf_lut = BRDFLookupTable(lut_name)
    
    print("\nValidating using Monte Carlo sampling:")
    for inc in [0, 30, 45, 60, 75, 89]:
        # Generate random emission and azimuth angles
        em_angles = np.random.uniform(0, 90, n_samples)
        az_angles = np.random.uniform(0, 360, n_samples)
        
        # Convert emission angles to radians for integration
        em_rad = np.radians(em_angles)
        
        # Calculate BRDF values
        brdf_values = np.array([brdf_lut.query(inc, em, az) 
                              for em, az in zip(em_angles, az_angles)])
        
        # Monte Carlo integration
        integrand = brdf_values * np.cos(em_rad) * np.sin(em_rad)
        integral = 2*np.pi * (np.pi/2) * np.mean(integrand)  # multiply by total area (2π * π/2)
        
        print(f"Incidence angle {inc:5.1f}°: Integral = {integral:.6f} (should be 1.0)")

def main():
    lut_name = "lambertian.npy"
    
    # Run both validation methods
    validate_brdf_lut_integration(lut_name)
    validate_brdf_lut_monte_carlo(lut_name)

if __name__ == "__main__":
    main()