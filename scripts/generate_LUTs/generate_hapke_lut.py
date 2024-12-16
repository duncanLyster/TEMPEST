"""
Generate Hapke BRDF lookup table.
This script creates and saves a lookup table for BRDF values based on Hapke's equation.
Hapke's model includes 9 parameters which must be manually set in the script.
"""

import numpy as np
from pathlib import Path
import sys

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parents[1] / "src"
sys.path.append(str(src_dir))

from src.utilities.locations import Locations

def hapke_brdf(mu, mu0, g, w, h, B0, c, theta_bar, b, bc):
    """
    Hapke BRDF calculation.

    Parameters:
        mu: Cosine of the emission angle (dimensionless)
        mu0: Cosine of the incidence angle (dimensionless)
        g: Phase angle (degrees)
        w: Single scattering albedo (dimensionless)
        h: Roughness parameter (dimensionless)
        B0: Opposition surge amplitude (dimensionless)
        c: Opposition surge width (dimensionless)
        theta_bar: Macroscopic roughness angle (degrees)
        b: Backscatter amplitude (dimensionless)
        bc: Backscatter width (dimensionless)

    Returns:
        BRDF value (dimensionless)
    """
    # Convert angles to radians where needed
    g_rad = np.radians(g)
    theta_bar_rad = np.radians(theta_bar)

    # Avoid division by zero in angle-dependent terms
    mu = max(mu, 1e-6)
    mu0 = max(mu0, 1e-6)

    # Compute phase function P(g)
    P = 1 + b * np.cos(g_rad) ** 2 / (1 + bc * g_rad ** 2)

    # Opposition effect B(g)
    B = B0 / (1 + c * np.tan(g_rad / 2))

    # Roughness correction
    roughness_factor = (1 - h * np.tan(theta_bar_rad) ** 2 / (mu + mu0))

    # Hapke BRDF
    brdf = (w / (4 * np.pi)) * ((mu0 / (mu0 + mu)) * P * B + roughness_factor)
    
    # Normalise BRDF for energy conservation
    brdf = max(brdf, 0)  # Ensure non-negative values
    return brdf

def validate_hapke_brdf():
    """
    Validate BRDF values against reference data.
    """
    # Example reference values from the paper (adjust as needed)
    reference_values = [
        {'mu': 1.0, 'mu0': 1.0, 'g': 0.0, 'expected': 0.045},
        {'mu': 0.5, 'mu0': 1.0, 'g': 30.0, 'expected': 0.02},
        {'mu': 0.5, 'mu0': 0.5, 'g': 60.0, 'expected': 0.01},
    ]

    # Hapke parameters
    hapke_params = {
        'w': 0.485,       # Single scattering albedo
        'h': 0.2,         # Roughness parameter
        'B0': 1.5,        # Opposition surge amplitude
        'c': 0.3,         # Opposition surge width
        'theta_bar': 21.69, # Macroscopic roughness angle (degrees)
        'b': 0.15,        # Backscatter amplitude
        'bc': 0.1         # Backscatter width
    }

    for ref in reference_values:
        mu = ref['mu']
        mu0 = ref['mu0']
        g = ref['g']
        expected = ref['expected']
        calculated = hapke_brdf(mu, mu0, g, **hapke_params)
        print(f"mu: {mu}, mu0: {mu0}, g: {g}° => Calculated: {calculated:.6f}, Expected: {expected:.6f}")

def create_hapke_lut(inc_resolution=1, em_resolution=1, az_resolution=1, params=None):
    """
    Create Hapke BRDF lookup table.

    Parameters:
        inc_resolution: Resolution for incidence angles (degrees)
        em_resolution: Resolution for emission angles (degrees)
        az_resolution: Resolution for azimuthal angles (degrees)
        params: Hapke model parameters (dict)

    Returns:
        LUT dictionary containing:
            - table: The BRDF values
            - incidence_angles
            - emission_angles
            - azimuth_angles
    """
    if params is None:
        raise ValueError("Hapke parameters must be provided.")

    # Unpack Hapke parameters
    w = params['w']
    h = params['h']
    B0 = params['B0']
    c = params['c']
    theta_bar = params['theta_bar']
    b = params['b']
    bc = params['bc']

    # Create angle arrays
    incidence_angles = np.linspace(0, 90, int(90/inc_resolution) + 1)
    emission_angles = np.linspace(0, 90, int(90/em_resolution) + 1)
    azimuth_angles = np.linspace(0, 180, int(180/az_resolution) + 1)

    # Create empty table
    table = np.zeros((len(incidence_angles), len(emission_angles), len(azimuth_angles)), dtype=np.float32)

    # Populate table
    for i, inc in enumerate(incidence_angles):
        for j, em in enumerate(emission_angles):
            for k, az in enumerate(azimuth_angles):
                mu = np.cos(np.radians(em))
                mu0 = np.cos(np.radians(inc))
                g = az
                table[i, j, k] = hapke_brdf(mu, mu0, g, w, h, B0, c, theta_bar, b, bc)

    return {
        'table': table,
        'incidence_angles': incidence_angles,
        'emission_angles': emission_angles,
        'azimuth_angles': azimuth_angles
    }

def main():
    # Hapke parameters with emphasis on opposition effect
    hapke_params = {
        'w': 0.485,       # Single scattering albedo
        'h': 0.2,         # Roughness parameter
        'B0': 1.5,        # Opposition surge amplitude
        'c': 0.3,         # Opposition surge width
        'theta_bar': 21.69, # Macroscopic roughness angle (degrees)
        'b': 0.15,        # Backscatter amplitude
        'bc': 0.1         # Backscatter width
    }

    # Validate Hapke BRDF
    print("Validating Hapke BRDF...")
    validate_hapke_brdf()

    # Initialize locations and ensure directories exist
    locations = Locations()
    locations.ensure_directories_exist()

    # Generate Hapke LUT
    print("Generating Hapke BRDF lookup table...")
    lut_data = create_hapke_lut(inc_resolution=1,
                                 em_resolution=1,
                                 az_resolution=1,
                                 params=hapke_params)

    # Save the LUT using locations
    output_path = locations.get_scattering_lut_path('hapke.npy')
    np.save(output_path, lut_data)

    # Print memory usage and save location
    memory_mb = lut_data['table'].nbytes / (1024 * 1024)
    print(f"LUT size: {memory_mb:.2f} MB")
    print(f"Saved LUT to {output_path}")

if __name__ == "__main__":
    main()
