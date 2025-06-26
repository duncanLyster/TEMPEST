# calculate_phase_curve.py

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import math
from typing import Tuple
from src.utilities.utils import rotate_vector, calculate_rotation_matrix
from src.model.scattering import BRDFLookupTable
from src.model.emission import EPFLookupTable

def calculate_phase_curve(
    shape_model,
    simulation,
    thermal_data,
    config,
    phase_curve_type='visible',
    observer_distance=1e9,
    normalized=False,
    plot=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the disc-integrated phase curve.
    Wrapper function that calls the appropriate specialized function.
    """
    phase_angles = np.linspace(0, 180, num=181)
    phase_angles_rad = np.radians(phase_angles)
    brightness_values = []

    # Initialize lookup tables
    brdf_lut = None
    epf_lut = None
    if phase_curve_type == 'visible':
        brdf_lut = BRDFLookupTable(config.scattering_lut)
    elif phase_curve_type == 'thermal':
        epf_lut = EPFLookupTable(config.emission_lut)

    # Load dome flux data for thermal calculations if available
    dome_flux_data = None
    if phase_curve_type == 'thermal':
        dome_flux_data = load_dome_flux_data()

    for phase_angle in phase_angles_rad:
        # Shift phase curve by 90 degrees
        shifted_phase_angle = phase_angle + np.pi/2
        observer_direction = compute_observer_direction(
            simulation.sunlight_direction, simulation.rotation_axis, shifted_phase_angle
        )
        observer_position = observer_direction * observer_distance

        visible_indices = compute_visible_facets_from_observer(
            shape_model, observer_position
        )

        total_brightness = 0.0

        for idx in visible_indices:
            facet = shape_model[idx]
            projected_area = compute_projected_area(facet, observer_position)
            if projected_area <= 0:
                continue

            if phase_curve_type == 'visible':
                total_brightness += compute_visible_brightness(
                    facet, observer_position, simulation, thermal_data,
                    brdf_lut, idx, config
                ) * projected_area
            else:  # thermal
                if dome_flux_data is not None:
                    # Use advanced dome-based thermal calculation
                    total_brightness += compute_advanced_thermal_brightness(
                        facet, observer_position, simulation, thermal_data,
                        epf_lut, idx, config, dome_flux_data
                    ) * projected_area
                else:
                    # Fallback to simple thermal calculation
                    total_brightness += compute_thermal_brightness(
                        thermal_data, shape_model, simulation, idx, 0, observer_direction
                    ) * projected_area

        brightness_values.append(total_brightness)

    brightness_values = np.array(brightness_values)
    if normalized:
        brightness_values /= np.max(brightness_values)

    if plot:
        plot_phase_curve(phase_angles, brightness_values, phase_curve_type)

    return phase_angles, brightness_values

def load_dome_flux_data():
    """Load dome flux data for advanced thermal calculations."""
    dome_flux_path = os.path.join('outputs', 'dome_fluxes.h5')
    if not os.path.exists(dome_flux_path):
        return None
    
    try:
        with h5py.File(dome_flux_path, 'r') as dfh:
            dome_flux_data = {
                'dome_flux_th': dfh['dome_flux_th'][:],      # (n_facets, M, T)
                'dome_bin_normals': dfh['dome_normals'][:],  # (M, 3)
                'dome_bin_areas': dfh['dome_bin_areas'][:],  # (n_facets, M)
                'dome_bin_solid_angles': dfh['dome_bin_solid_angles'][:]  # (M,)
            }
        return dome_flux_data
    except Exception as e:
        print(f"Warning: Could not load dome flux data: {e}")
        return None

def compute_advanced_thermal_brightness(facet, observer_position, simulation, thermal_data,
                                      epf_lut, idx, config, dome_flux_data):
    """
    Compute thermal brightness using advanced dome flux calculations.
    This integrates with the existing roughness and sub-facet thermal modeling.
    """
    # Calculate observer direction
    observer_direction = (observer_position - facet.position)
    observer_direction /= np.linalg.norm(observer_direction)
    
    # Get emission angle for EPF lookup
    cos_emission = np.dot(facet.normal, observer_direction)
    if cos_emission <= 0:
        return 0.0
    
    emission_angle = np.degrees(np.arccos(np.clip(cos_emission, 0, 1)))
    epf_value = epf_lut.query(emission_angle)
    
    # Use dome flux data if available and dome_rotation exists (accounts for sub-facet structure)
    if (dome_flux_data is not None and 
        hasattr(facet, 'dome_rotation') and 
        facet.dome_rotation is not None):
        
        # Transform observer direction to local facet coordinates
        local_observer = facet.dome_rotation.dot(observer_direction)
        
        # Find best dome bin for this viewing direction
        dome_cosines = np.dot(local_observer, dome_flux_data['dome_bin_normals'].T)
        dome_cosines = np.clip(dome_cosines, 0, None)
        
        if np.max(dome_cosines) > 1e-6:  # Check if any bin is visible
            best_bin = np.argmax(dome_cosines)
            
            # Get thermal flux from this bin (average over rotation)
            flux_values = dome_flux_data['dome_flux_th'][idx, best_bin, :]
            avg_flux = np.mean(flux_values)
            
            # Get bin area
            bin_area = dome_flux_data['dome_bin_areas'][idx, best_bin]
            
            if bin_area > 0:
                # Calculate observed thermal brightness with EPF
                # Scale by dome radius factor
                scale = config.kernel_dome_radius_factor ** 2
                thermal_brightness = (avg_flux / bin_area * scale) * epf_value * cos_emission
                return thermal_brightness
    
    # Fallback to simple thermal calculation
    temperature = np.mean(thermal_data.temperatures[idx, :])
    sigma = 5.670374419e-8
    thermal_emission = simulation.emissivity * sigma * (temperature**4)
    
    return thermal_emission * epf_value * cos_emission

def compute_observer_direction(sunlight_direction, rotation_axis, phase_angle):
    """
    Compute the observer's direction vector given the phase angle.
    """
    # Compute a vector perpendicular to the sunlight direction and rotation axis
    perp_vector = np.cross(sunlight_direction, rotation_axis)
    if np.linalg.norm(perp_vector) == 0:
        # sunlight_direction and rotation_axis are parallel, choose any perpendicular vector
        perp_vector = np.array([1, 0, 0])
    else:
        perp_vector /= np.linalg.norm(perp_vector)

    # Rotate the sunlight direction by the phase angle around the perpendicular vector
    observer_direction = rotate_vector(
        sunlight_direction, perp_vector, phase_angle
    )
    observer_direction /= np.linalg.norm(observer_direction)
    return observer_direction

def compute_visible_facets_from_observer(shape_model, observer_position):
    """
    Determine which facets are visible from the observer's position.
    """
    visible_indices = []
    observer_direction = -observer_position / np.linalg.norm(observer_position)

    for idx, facet in enumerate(shape_model):
        # Vector from facet to observer
        to_observer = observer_position - facet.position
        to_observer_norm = to_observer / np.linalg.norm(to_observer)
        # Check if facet is facing the observer
        if np.dot(facet.normal, to_observer_norm) < 0:
            continue  # Facet is facing away from observer
        # Optionally, implement shadowing here
        visible_indices.append(idx)
    return visible_indices

def compute_projected_area(facet, observer_position):
    """
    Compute the projected area of the facet as seen from the observer.
    """
    to_observer = observer_position - facet.position
    distance_squared = np.dot(to_observer, to_observer)
    to_observer_norm = to_observer / np.sqrt(distance_squared)
    cos_theta = np.dot(facet.normal, to_observer_norm)
    if cos_theta <= 0:
        return 0
    # Projected area factor
    projected_area = facet.area * cos_theta / distance_squared
    return projected_area

def compute_thermal_brightness(thermal_data, shape_model, simulation, idx, time_step, observer_direction):
    """
    Compute the thermal brightness contribution from a single facet (simple version)
    """
    # Get surface temperature (now from 2D array)
    temperature = thermal_data.temperatures[idx, time_step]  
    
    # Rest of the function remains the same
    normal = shape_model[idx].normal
    area = shape_model[idx].area
    
    # Calculate emission angle cosine
    cos_emission = np.dot(normal, observer_direction)
    
    if cos_emission <= 0:
        return 0
    
    # Calculate thermal emission using Planck function
    thermal_emission = simulation.emissivity * (5.67e-8) * temperature**4
    
    return thermal_emission * area * cos_emission

def compute_visible_brightness(facet, observer_position, simulation, thermal_data, brdf_lut, idx, config):
    """Compute visible brightness for a single facet."""
    to_observer = observer_position - facet.position
    to_observer_norm = to_observer / np.linalg.norm(to_observer)
    
    # Calculate angles for BRDF
    em_cos = np.dot(facet.normal, to_observer_norm)
    em_deg = np.degrees(np.arccos(np.clip(em_cos, -1.0, 1.0)))
    
    inc_cos = np.dot(simulation.sunlight_direction, facet.normal)
    if inc_cos <= 0:  # Facet not illuminated
        return 0.0
    inc_deg = np.degrees(np.arccos(inc_cos))
    
    # Calculate azimuth angle
    sun_projected = (simulation.sunlight_direction - 
                    np.dot(simulation.sunlight_direction, facet.normal) * facet.normal)
    obs_projected = to_observer_norm - np.dot(to_observer_norm, facet.normal) * facet.normal
    
    if np.linalg.norm(sun_projected) > 1e-10 and np.linalg.norm(obs_projected) > 1e-10:
        sun_projected /= np.linalg.norm(sun_projected)
        obs_projected /= np.linalg.norm(obs_projected)
        az_cos = np.dot(sun_projected, obs_projected)
        az_deg = np.degrees(np.arccos(np.clip(az_cos, -1.0, 1.0)))
    else:
        az_deg = 0.0
    
    # Calculate direct and scattered components
    solar_constant = simulation.solar_luminosity / (4 * np.pi * simulation.solar_distance_m**2)
    direct_flux = solar_constant * inc_cos
    
    brdf = brdf_lut.query(inc_deg, em_deg, az_deg)
    reflected_direct = direct_flux * brdf
    
    reflected_scattered = 0.0
    if config.n_scatters > 0:
        total_flux = thermal_data.insolation[idx, 0]
        scattered_flux = total_flux - direct_flux
        reflected_scattered = scattered_flux * simulation.albedo / np.pi
    
    return reflected_direct + reflected_scattered

def compute_thermal_emission(facet, observer_position, thermal_data, epf_lut, idx, config):
    """Calculate thermal emission from a facet towards an observer."""
    # Calculate emission angle
    direction_to_observer = normalize_vector(observer_position - facet.position)
    cos_emission = np.dot(direction_to_observer, facet.normal)
    
    if cos_emission <= 0:
        return 0.0
        
    emission_angle = np.degrees(np.arccos(cos_emission))
    
    # Get EPF value for this emission angle
    epf = epf_lut.query(emission_angle)
    
    # Calculate thermal emission with directional dependence
    # Use average temperature over the day
    temperature = np.mean(thermal_data.temperatures[idx, :])
    thermal_emission = config.emissivity * 5.670374419e-8 * (temperature**4)
    
    return thermal_emission * epf * cos_emission

def plot_phase_curve(phase_angles, brightness_values, phase_curve_type):
    """
    Plot the phase curve.
    """
    plt.figure()
    plt.plot(phase_angles, brightness_values, label=f'{phase_curve_type.capitalize()} Phase Curve')
    plt.xlabel('Phase Angle (degrees)')
    plt.ylabel('Total Brightness')
    plt.title(f'{phase_curve_type.capitalize()} Disc-Integrated Phase Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def normalize_vector(vector):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

