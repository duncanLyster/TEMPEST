# calculate_phase_curve.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Import necessary functions from your main script
# Make sure thermophysical_body_model.py is in your PYTHONPATH or same directory
from utils import calculate_black_body_temp, rotate_vector

def calculate_phase_curve(
    shape_model,
    simulation,
    thermal_data,
    phase_curve_type='visible',
    observer_distance=1e9,  # in meters
    normalized=False,
    plot=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the disc-integrated phase curve.

    Parameters:
    - shape_model: List of Facet objects.
    - simulation: Simulation object containing simulation parameters.
    - thermal_data: ThermalData object containing insolation and temperature arrays.
    - phase_curve_type: 'visible' or 'thermal' to specify the type of phase curve.
    - observer_distance: Distance from the observer to the body (in meters).
    - normalized: Boolean flag to normalize the brightness values.
    - plot: Boolean flag to plot the phase curve.

    Returns:
    - phase_angles: Array of phase angles in degrees.
    - brightness_values: Array of brightness values corresponding to the phase angles.
    """
    # Define phase angles from 0 to 180 degrees
    phase_angles = np.linspace(0, 180, num=181)  # 1-degree increments
    phase_angles_rad = np.radians(phase_angles)

    rotation_axis = simulation.rotation_axis
    sunlight_direction = simulation.sunlight_direction

    brightness_values = []

    for phase_angle in phase_angles_rad:
        # Compute observer position
        observer_direction = compute_observer_direction(
            sunlight_direction, rotation_axis, phase_angle
        )
        observer_position = observer_direction * observer_distance

        # Determine visible facets
        visible_indices = compute_visible_facets_from_observer(
            shape_model, observer_position
        )

        # Compute brightness for this phase angle
        total_brightness = 0.0

        for idx in visible_indices:
            facet = shape_model[idx]
            projected_area = compute_projected_area(facet, observer_position)
            if projected_area <= 0:
                continue

            if phase_curve_type == 'visible':
                # Use insolation data
                time_index = 0  # Assuming static insolation, adjust if necessary
                reflected_flux = compute_reflected_light(
                    idx, simulation, thermal_data.insolation, time_index
                )
                total_brightness += reflected_flux * projected_area

            elif phase_curve_type == 'thermal':
                # Use temperature data
                time_index = 0  # Assuming static temperature, adjust if necessary
                emitted_flux = compute_thermal_emission(
                    idx, simulation, thermal_data.temperatures, time_index
                )
                total_brightness += emitted_flux * projected_area

        brightness_values.append(total_brightness)

    brightness_values = np.array(brightness_values)

    # Normalization (if required)
    if normalized:
        brightness_values /= np.max(brightness_values)

    # Plotting (if required)
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
        -sunlight_direction, perp_vector, phase_angle
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

def compute_reflected_light(idx, simulation, insolation, time_index):
    """
    Compute the reflected light from a facet.
    """
    albedo = simulation.albedo
    # Incident solar flux on the facet
    incident_flux = insolation[idx, time_index]
    # Scattered flux using Lambertian scattering
    reflected_flux = incident_flux * albedo / np.pi
    return reflected_flux

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

