# scripts/generate_LUTs/generate_opposition_brdf_lut.py

"""
Generate BRDF lookup table with Opposition Effect.
This script creates and saves a lookup table for a BRDF that includes the opposition effect.
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parents[1] / "src"
sys.path.append(str(src_dir))

from src.utilities.locations import Locations

def create_opposition_brdf_lut(inc_resolution=1, em_resolution=1, az_resolution=1, e=1.0, alpha=0.05):
    """
    Create BRDF lookup table with Opposition Effect.
    
    Parameters:
    - inc_resolution: Resolution for incidence angles in degrees.
    - em_resolution: Resolution for emission angles in degrees.
    - az_resolution: Resolution for azimuth angles in degrees.
    - e: Strength of the opposition effect.
    - alpha: Width of the opposition peak.
    
    Returns:
    - dict containing the table and angle arrays.
    """
    # Create angle arrays in degrees
    incidence_angles = np.linspace(0, 90, int(90/inc_resolution) + 1)
    emission_angles = np.linspace(0, 90, int(90/em_resolution) + 1)
    azimuth_angles = np.linspace(0, 180, int(180/az_resolution) + 1)
    
    # Convert angles to radians for computation
    inc_rad = np.radians(incidence_angles)
    em_rad = np.radians(emission_angles)
    
    # Initialize the BRDF table
    table = np.zeros((len(incidence_angles),
                      len(emission_angles),
                      len(azimuth_angles)),
                     dtype=np.float32)
    
    # Populate the BRDF table
    for i, theta_i in enumerate(inc_rad):
        for j, theta_e in enumerate(em_rad):
            # Phase angle approximation (sum of incidence and emission angles)
            phase_angle = theta_i + theta_e
            opposition = 1 + e * np.exp(- (alpha * phase_angle / np.pi)**2)
            table[i, j, :] = opposition
    
    # Normalize the BRDF so that it integrates to 1
    # Compute the solid angle element dω
    d_inc = np.radians(inc_resolution)
    d_em = np.radians(em_resolution)
    d_az = np.radians(az_resolution)
    
    # Integral over all directions
    integral = np.sum(table * np.sin(inc_rad[:, np.newaxis, np.newaxis]) * d_inc * d_em * d_az)
    
    # Normalize
    table /= integral
    
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
    
    # Generate Opposition BRDF LUT
    print("Generating BRDF lookup table with Opposition Effect...")
    lut_data = create_opposition_brdf_lut(inc_resolution=1,
                                         em_resolution=1,
                                         az_resolution=1,
                                         e=1.0,       # Adjust strength as needed
                                         alpha=0.05)  # Adjust width as needed
    
    # Save the LUT using locations
    output_path = locations.get_scattering_lut_path('opposition_brdf.npy')
    np.save(output_path, lut_data)
    
    # Print memory usage and save location
    memory_mb = lut_data['table'].nbytes / (1024 * 1024)
    print(f"LUT size: {memory_mb:.2f} MB")
    print(f"Saved LUT to {output_path}")

if __name__ == "__main__":
    main()

# Load the LUT
lut_data = np.load('path/to/opposition_brdf.npy', allow_pickle=True).item()

# Extract data
table = lut_data['table']
inc_angles = lut_data['incidence_angles']
em_angles = lut_data['emission_angles']

# Visualize a slice of the BRDF table
az_index = 0

# Plot BRDF values for different incidence angles at a fixed emission angle
em_index = 0
plt.figure(figsize=(10, 6))
for i, inc_angle in enumerate(inc_angles):
    plt.plot(em_angles, table[i, :, az_index], label=f'Incidence {inc_angle}°')

plt.title('BRDF with Pronounced Opposition Effect')
plt.xlabel('Emission Angle (degrees)')
plt.ylabel('BRDF Value')
plt.legend()
plt.grid(True)
plt.show()

# Compute the integral
inc_rad = np.radians(inc_angles)
em_rad = np.radians(em_angles)
d_inc = inc_rad[1] - inc_rad[0]
d_em = em_rad[1] - em_rad[0]
d_az = np.radians(lut_data['azimuth_angles'][1] - lut_data['azimuth_angles'][0])

integral = np.sum(table * np.sin(inc_rad[:, np.newaxis, np.newaxis]) * d_inc * d_em * d_az)
print(f"BRDF Integral: {integral:.4f}")  # Should be close to 1