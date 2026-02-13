"""
Generate High-Fidelity SPECTRAL Roughness Lookup Tables (LUTs) for TEMPEST.

This script simulates the thermal behavior of a rough surface (modeled as a spherical crater)
across a wide range of conditions to generate a radiometric correction table.

Changes in V2:
- SPECTRAL OUTPUT: Calculates radiance ratios using Planck function instead of T^4.
- High-Resolution internal simulation (720 steps) for stability.
- Uses Dimensionless Thermal Parameter (Theta) for grid.

Dimensions:
1. Thermal Parameter (Theta): Capture thermal memory.
2. Opening Angle: Capture crater shape (90 = Hemisphere).
3. Wavelength (Lambda): Capture spectral non-linearity.
4. Latitude: Capture diurnal history (Equatorial vs Polar).
5. Sun Phase (Time): The rotation angle of the body (0-360).
6. Emission Angle (View): Observer angle (0-90).
7. Azimuth Angle (View): Relative angle between Sun and Observer (0-180).

Output:
A 7D tensor containing the Radiance Factor:
    R(lambda) = Radiance_Rough(lambda) / Radiance_Smooth(lambda)
"""

import os
import sys
import time
import numpy as np
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path

# Ensure root directory is in the Python path to allow 'from src...' imports
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.model.facet import Facet
from src.model.simulation import Simulation, ThermalData
from src.model.solvers import TemperatureSolverFactory
from src.utilities.config import Config
from src.utilities.locations import Locations
from src.utilities.utils import calculate_rotation_matrix

# --- Configuration for the LUT Generation ---

# Grid Definitions
THETA_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0]      # Thermal Parameter (Dimensionless)
OPENING_ANGLES = [90.0]                         # Crater Opening Angles (Degrees). 90 = Hemisphere.
WAVELENGTHS_MICRONS = [5.0, 8.0, 15.0, 50.0, 100.0] # Spectral bands
LATITUDE_VALUES = [0.0, 30.0, 60.0, 85.0]       # Latitudes (degrees)
LUT_TIMESTEPS = 90                              # Steps in the output LUT (4 degree resolution)
SIM_TIMESTEPS = 180                             # Internal simulation steps (2x LUT resolution for fidelity)
EMISSION_ANGLES = np.linspace(0, 89, 10)        # Viewing angles (avoid 90 to prevent div by zero)
AZIMUTH_ANGLES = np.linspace(0, 180, 10)        # Relative azimuths (symmetric)

OUTPUT_FILE = "roughness_lut_spectral_v1.h5"

def planck_function(wavelength_um, temp_k):
    """
    Calculate spectral radiance B(lambda, T).
    Result unit: W / (m^2 sr um)
    
    Constants:
    c1 = 2 * h * c^2 = 1.191042e8 W um^4 / (m^2 sr)
    c2 = h * c / k_B = 1.4387752e4 um K
    """
    c1 = 1.191042e8  
    c2 = 1.4387752e4 
    
    # Avoid overflow/div-by-zero for T=0 or very small
    t_safe = np.maximum(temp_k, 1e-5)
    
    # Calculate
    val = c1 / (wavelength_um**5 * (np.exp(c2 / (wavelength_um * t_safe)) - 1))
    return val

class ReferenceConfig(Config):
    """
    A minimal Config subclass that does not require a file.
    Used to generate the generic LUT under standard reference conditions.
    """
    def __init__(self):
        # Initialize Locations (required by base class, but we won't use paths)
        self.locations = Locations()
        
        # Manually set standard reference values for the Generator
        # We must populate config_data because Simulation class reads from it.
        self.config_data = {
            'solar_distance_au': 1.0,
            'solar_luminosity': 3.828e26,
            'albedo': 0.12,
            'emissivity': 0.95,
            'subsurface_heating_flux': 0.0,
            
            # Solver Settings
            'temp_solver': 'tempest_implicit',
            'convergence_method': 'mean',
            'convergence_target': 0.1,
            'max_days': 100,
            'min_days': 5,
            'n_layers': 40,
            'include_shadowing': True,
            'n_scatters': 0, 
            'include_self_heating': True,
            'vf_rays': 1000,
            'intra_facet_scatters': 2,
            
            # Kernel/Roughness specific
            'apply_kernel_based_roughness': True,
            'roughness_kernel': 'spherical_section',
            'kernel_subfacets_count': 100,
            'kernel_profile_angle_degrees': 90.0,
            'kernel_directional_bins': 36,
            'kernel_dome_radius_factor': 100.0,
            
            # Disable Plotting/Saving
            'plot_temp_curve': [],
            'save_temp_curve_data': False,
            'save_visible_phase_curve_data': False,
            'save_thermal_phase_curve_data': False,
            'animate_subfacets': False,
            'animate_final_day_temp_distribution': False,
            'animate_shadowing': False,
            
            # Shape model settings (dummy)
            'shape_model_file': 'cube_without_roughness.obj', # Dummy filename
            'ra_degrees': 0.0,
            'dec_degrees': 90.0,
            'rotation_period_hours': 10.0, # Will be overwritten by generator
            
            # Physical props (will be overwritten by generator)
            'density': 1000.0,
            'specific_heat_capacity': 1000.0,
            'thermal_inertia': 100.0 
        }
        
        # Also set attributes for Config compatibility if needed by other utils
        self.load_settings()
        
        # Override path to avoid errors
        self.path_to_shape_model_file = None

def simulate_crater_diurnal_cycle(theta, opening_angle, latitude, config, n_timesteps):
    """
    Run a full diurnal simulation for a single crater configuration.
    Uses TEMPEST core logic (Simulation, Facet, Solver).
    """
    # 1. Setup Simulation Object
    simulation = Simulation(config)
    
    simulation.timesteps_per_day = n_timesteps
    simulation.rotation_period_hours = 10.0 
    simulation.density = 1000.0
    simulation.specific_heat_capacity = 1000.0
    omega = 2 * np.pi / (simulation.rotation_period_hours * 3600)
    simulation.angular_velocity = omega
    
    # Calculate Thermal Inertia from Dimensionless Theta
    # Theta = (Gamma * sqrt(omega)) / (epsilon * sigma * Tss^3)
    # Tss = ((1-A) * S / (epsilon * sigma))^0.25
    
    solar_flux = simulation.solar_luminosity / (4 * np.pi * simulation.solar_distance_m**2)
    boltzmann = 5.670374419e-8
    epsilon = simulation.emissivity
    albedo = simulation.albedo
    
    # Sub-solar temperature
    val = ((1 - albedo) * solar_flux) / (epsilon * boltzmann)
    tss = val ** 0.25
    
    # Gamma = Theta * epsilon * sigma * Tss^3 / sqrt(omega)
    # Note: Tss^3 * epsilon * sigma = (1-A) * S / Tss
    # Easier: Gamma = Theta * ((1-A) * S / Tss) / sqrt(omega)
    
    # Avoid division by zero
    if omega <= 0: omega = 1e-9
    
    thermal_inertia = (theta * epsilon * boltzmann * (tss**3)) / np.sqrt(omega)
    
    # Update Simulation with derived thermal inertia
    simulation.thermal_inertia = thermal_inertia
    # Re-run initialization calculations that depend on thermal_inertia
    simulation.thermal_conductivity = (simulation.thermal_inertia**2) / (simulation.density * simulation.specific_heat_capacity)
    simulation.skin_depth = (simulation.thermal_conductivity / (simulation.density * simulation.specific_heat_capacity * simulation.angular_velocity)) ** 0.5
    simulation.layer_thickness = 8 * simulation.skin_depth / simulation.n_layers
    simulation.thermal_diffusivity = simulation.thermal_conductivity / (simulation.density * simulation.specific_heat_capacity)
    
    # Set Geometry
    lat_rad = np.radians(latitude)
    # Normal vector in X-Z plane (Z is pole)
    normal = np.array([np.cos(lat_rad), 0.0, np.sin(lat_rad)])
    
    vertices = [
        np.array([-0.5, -0.288, 0.0]),
        np.array([0.5, -0.288, 0.0]),
        np.array([0.0, 0.577, 0.0])
    ]
    facet = Facet(normal, vertices, simulation.timesteps_per_day, simulation.max_days, simulation.n_layers, False)
    
    # Generate Crater
    config.kernel_profile_angle_degrees = opening_angle
    config.kernel_subfacets_count = 100 # High resolution for accuracy
    facet.generate_spherical_depression(config, simulation)
    
    # 3. Insolation Calculation
    sun_vectors = np.zeros((n_timesteps, 3))
    sun_declination = 0.0 
    if latitude > 80:
        sun_declination = np.radians(5.0) 
        
    for t in range(n_timesteps):
        angle = (2 * np.pi * t) / n_timesteps
        # Sun vector in body frame (Sun rotating around Z)
        x = np.cos(sun_declination) * np.cos(-angle)
        y = np.cos(sun_declination) * np.sin(-angle)
        z = np.sin(sun_declination)
        sun_vectors[t] = np.array([x, y, z])
        
    solar_constant = simulation.solar_luminosity / (4 * np.pi * simulation.solar_distance_m**2)
    
    for t in range(n_timesteps):
        sun_vec = sun_vectors[t]
        cos_inc = np.dot(facet.normal, sun_vec)
        
        if cos_inc > 0:
            facet.parent_incident_energy_packets = [(solar_constant, sun_vec, 'solar')]
            facet.process_intra_depression_energetics(config, simulation)
            facet.depression_thermal_data.insolation[:, t] = facet._last_absorbed_solar
        else:
            facet.depression_thermal_data.insolation[:, t] = 0.0
            
    # 4. Solve Rough Temperatures
    solver = TemperatureSolverFactory.create(config.temp_solver)
    solver.initialize_temperatures(facet.depression_thermal_data, simulation, config)
    res = solver.solve(facet.depression_thermal_data, facet.sub_facets, simulation, config)
    
    if res['final_day_temperatures'] is None:
        raise ValueError(f"Solver failed for Theta={theta}, lat={latitude}")
        
    rough_temps = res['final_day_temperatures'] # (N_sub, T)
    
    # 5. Solve Smooth Reference
    smooth_data = ThermalData(1, n_timesteps, simulation.n_layers, simulation.max_days, False)
    for t in range(n_timesteps):
        cos_inc = np.dot(facet.normal, sun_vectors[t])
        if cos_inc > 0:
            smooth_data.insolation[0, t] = solar_constant * cos_inc * (1 - simulation.albedo)
        else:
            smooth_data.insolation[0, t] = 0.0
            
    solver.initialize_temperatures(smooth_data, simulation, config)
    dummy_facet = Facet(normal, vertices, n_timesteps, simulation.max_days, simulation.n_layers, False)
    dummy_facet.visible_facets = np.array([], dtype=np.int64) 
    res_smooth = solver.solve(smooth_data, [dummy_facet], simulation, config)
    
    if res_smooth['final_day_temperatures'] is None:
        raise ValueError(f"Smooth solver failed for Theta={theta}, lat={latitude}")
        
    smooth_temps = res_smooth['final_day_temperatures'][0] # (T,)
    
    return rough_temps, smooth_temps, facet, sun_vectors

def process_single_case(theta, opening_angle, lat, config):
    """
    Run simulation and compute viewing geometry grid.
    Returns: 4D array (Time, Wave, Emi, Azi)
    """
    try:
        rough_temps_sim, smooth_temps_sim, facet, sun_vectors_sim = simulate_crater_diurnal_cycle(theta, opening_angle, lat, config, SIM_TIMESTEPS)
    except Exception as e:
        print(f"Error in simulation (Th={theta}, ang={opening_angle}, lat={lat}): {e}")
        return np.full((LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS), len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)), np.nan, dtype=np.float32)

    # Resample to LUT grid
    indices = np.linspace(0, SIM_TIMESTEPS-1, LUT_TIMESTEPS, dtype=int)
    rough_temps = rough_temps_sim[:, indices]
    smooth_temps = smooth_temps_sim[indices]
    sun_vectors = sun_vectors_sim[indices]
    
    # Result: Time x Wavelength x Emi x Azi
    result_grid = np.zeros((LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS), len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)), dtype=np.float32)
    
    # Geometry Arrays
    normals = Facet._canonical_normals
    areas = Facet._canonical_areas
    triangles = Facet._canonical_subfacet_triangles
    centers = Facet._canonical_subfacet_centers
    facet_normal = facet.normal / np.linalg.norm(facet.normal)
    
    # Loop over Timesteps
    for t_idx in range(LUT_TIMESTEPS):
        t_sub = rough_temps[:, t_idx] # (N_sub,)
        t_smooth = smooth_temps[t_idx] # scalar
        
        # Sun Basis for Azimuth
        sun_vec = sun_vectors[t_idx]
        sun_vec_norm = sun_vec / np.linalg.norm(sun_vec)
        s_dot_n = np.dot(sun_vec_norm, facet_normal)
        sun_proj = sun_vec_norm - s_dot_n * facet_normal
        if np.linalg.norm(sun_proj) > 1e-6:
            basis_x = sun_proj / np.linalg.norm(sun_proj) 
        else:
            up = np.array([0, 0, 1]) if np.abs(np.dot([0,0,1], facet_normal)) < 0.9 else np.array([1, 0, 0])
            basis_x = np.cross(facet_normal, up)
            basis_x /= np.linalg.norm(basis_x)
        basis_y = np.cross(facet_normal, basis_x)
        
        # Loop Viewing Angles
        for i_e, emi_deg in enumerate(EMISSION_ANGLES):
            emi_rad = np.radians(emi_deg)
            sin_e = np.sin(emi_rad)
            cos_e = np.cos(emi_rad)
            
            for i_a, azi_deg in enumerate(AZIMUTH_ANGLES):
                azi_rad = np.radians(azi_deg)
                
                # View Vector
                v_local_x = sin_e * np.cos(azi_rad)
                v_local_y = sin_e * np.sin(azi_rad)
                v_local_z = cos_e
                view_vec = v_local_x * basis_x + v_local_y * basis_y + v_local_z * facet_normal
                view_vec /= np.linalg.norm(view_vec)
                
                # Ray-Trace Visibility (Effective Projected Area per Subfacet)
                # Flux=1.0 visible packet to measure projected area
                packet = (1.0, view_vec, 'visible') 
                E_vis, _, _, _ = Facet._process_incident_packet(
                    packet, facet.dome_rotation, facet.area, normals, areas, triangles, centers, 0.0, 0.0
                )
                
                # Loop Wavelengths
                for i_w, wave in enumerate(WAVELENGTHS_MICRONS):
                    # Rough Radiance
                    # L_sub(i) = B(wave, T_sub(i))
                    # Flux_rough = Sum( L_sub(i) * A_proj_sub(i) )
                    #            = Sum( B(wave, T_sub(i)) * E_vis[i] )
                    # (Note: E_vis is effectively A_proj because Flux_in was 1.0)
                    
                    rad_sub = planck_function(wave, t_sub)
                    weighted_sum = np.sum(rad_sub * E_vis)
                    
                    # Smooth Radiance
                    # Flux_smooth = B(wave, T_smooth) * A_macro_proj
                    # A_macro_proj = A_facet * cos(e)
                    
                    if t_smooth < 5.0: # Night / Shadow
                        ratio = 1.0
                    else:
                        rad_smooth = planck_function(wave, t_smooth)
                        flux_smooth = rad_smooth * facet.area * cos_e
                        
                        if flux_smooth < 1e-15:
                            ratio = 1.0
                        else:
                            ratio = weighted_sum / flux_smooth
                            
                    result_grid[t_idx, i_w, i_e, i_a] = ratio
                
    return result_grid

def main():
    print("Initializing SPECTRAL LUT Generator...")
    
    # Create Reference Config (Standard Physics, 1 AU)
    # This avoids loading any external config.yaml
    config = ReferenceConfig()
    
    # Tasks
    tasks = []
    for theta in THETA_VALUES:
        for angle in OPENING_ANGLES:
            for lat in LATITUDE_VALUES:
                tasks.append((theta, angle, lat))
                
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {len(THETA_VALUES)}x{len(OPENING_ANGLES)}x{len(LATITUDE_VALUES)}x{LUT_TIMESTEPS}x{len(WAVELENGTHS_MICRONS)}x{len(EMISSION_ANGLES)}x{len(AZIMUTH_ANGLES)}")
    
    n_jobs = 4
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_case)(theta, angle, lat, config) 
        for theta, angle, lat in tqdm(tasks)
    )
    
    # Assemble Tensor
    lut_tensor = np.zeros((len(THETA_VALUES), len(OPENING_ANGLES), len(LATITUDE_VALUES), 
                           LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS), 
                           len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)), dtype=np.float32)
    
    idx = 0
    for i_th, theta in enumerate(THETA_VALUES):
        for i_ang, angle in enumerate(OPENING_ANGLES):
            for i_lat, lat in enumerate(LATITUDE_VALUES):
                if results[idx] is not None:
                    lut_tensor[i_th, i_ang, i_lat, :, :, :, :] = results[idx]
                idx += 1
                
    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset("lut", data=lut_tensor)
        f.create_dataset("theta", data=THETA_VALUES)
        f.create_dataset("opening_angle", data=OPENING_ANGLES)
        f.create_dataset("wavelength", data=WAVELENGTHS_MICRONS)
        f.create_dataset("latitude", data=LATITUDE_VALUES)
        f.create_dataset("emission", data=EMISSION_ANGLES)
        f.create_dataset("azimuth", data=AZIMUTH_ANGLES)
        
    print("Done.")

if __name__ == "__main__":
    main()
