# calculate_phase_curve.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Import necessary functions from your main script
# Make sure thermophysical_body_model.py is in your PYTHONPATH or same directory
from src.utilities.utils import calculate_black_body_temp, rotate_vector
from src.model.scattering import BRDFLookupTable

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
    Calculate the disc-integrated phase curve using BRDF for visible light.
    Includes both direct and scattered light contributions.
    """
    # Initialize BRDF lookup table if needed for visible light
    brdf_lut = None
    if phase_curve_type == 'visible':
        brdf_lut = BRDFLookupTable(config.scattering_lut)
    
    phase_angles = np.linspace(0, 180, num=181)
    phase_angles_rad = np.radians(phase_angles)
    brightness_values = []

    for phase_angle in phase_angles_rad:
        observer_direction = compute_observer_direction(
            simulation.sunlight_direction, simulation.rotation_axis, phase_angle
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
                # Calculate angles for BRDF
                to_observer = observer_position - facet.position
                to_observer_norm = to_observer / np.linalg.norm(to_observer)
                
                # Emission angle (observer to facet)
                em_cos = np.dot(facet.normal, to_observer_norm)
                em_deg = np.degrees(np.arccos(np.clip(em_cos, -1.0, 1.0)))
                
                # Incidence angle (sun to facet)
                inc_cos = np.dot(simulation.sunlight_direction, facet.normal)
                if inc_cos <= 0:  # Facet not illuminated
                    continue
                inc_deg = np.degrees(np.arccos(inc_cos))
                
                # Calculate azimuth angle between sun and observer in facet plane
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
                
                # Calculate direct insolation component
                inc_cos = np.dot(simulation.sunlight_direction, facet.normal)
                if inc_cos <= 0:  # Facet not illuminated
                    continue
                    
                solar_constant = simulation.solar_luminosity / (4 * np.pi * simulation.solar_distance_m**2)
                direct_flux = solar_constant * inc_cos
                
                # Get BRDF value and calculate reflected flux for direct illumination
                brdf = brdf_lut.query(inc_deg, em_deg, az_deg)
                reflected_direct = direct_flux * brdf
                
                # Calculate scattered component by subtracting direct from total
                if config.n_scatters > 0:
                    total_flux = thermal_data.insolation[idx, 0]  # Total insolation
                    scattered_flux = total_flux - direct_flux
                    reflected_scattered = scattered_flux * simulation.albedo / np.pi
                
                total_brightness += (reflected_direct + reflected_scattered) * projected_area

            elif phase_curve_type == 'thermal':
                # Thermal calculation remains unchanged
                emitted_flux = compute_thermal_emission(
                    idx, simulation, thermal_data.temperatures, 0
                )
                total_brightness += emitted_flux * projected_area

        brightness_values.append(total_brightness)

    brightness_values = np.array(brightness_values)
    if normalized:
        brightness_values /= np.max(brightness_values)

    if plot:
        plot_phase_curve(phase_angles, brightness_values, phase_curve_type)

    return phase_angles, brightness_values

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

def compute_thermal_emission(idx, simulation, temperatures, time_index):
    """
    Compute the thermal emission from a facet.
    """
    emissivity = simulation.emissivity
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    temperature = temperatures[idx, time_index, 0]  # Surface temperature
    emitted_flux = emissivity * sigma * temperature ** 4
    return emitted_flux

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

