# src/model/radiance.py

"""
Radiance Calculation Module for TEMPEST

This module provides functions to calculate observed radiance from a target body
as seen by an observer (spacecraft, telescope, etc.). It supports both thermal
emission and reflected sunlight.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from src.model.emission import EPFLookupTable
from src.model.scattering import BRDFLookupTable


def calculate_observed_radiance(
    shape_model,
    thermal_data,
    simulation,
    config,
    observer,
    timestep: int = 0,
    radiance_type: str = 'thermal',
    wavelength_um: Optional[float] = None
) -> Dict:
    """
    Calculate observed radiance from the target body at a specific timestep.
    
    Args:
        shape_model: List of Facet objects representing the body
        thermal_data: ThermalData object with temperature information
        simulation: Simulation object with geometry information
        config: Config object with model parameters
        observer: Observer object for viewing geometry
        timestep: Timestep index (0 to timesteps_per_day-1)
        radiance_type: Type of radiance to calculate ('thermal', 'reflected', or 'total')
        wavelength_um: Wavelength in microns (for spectral radiance, not yet implemented)
        
    Returns:
        Dictionary with:
            - 'total_radiance': Total integrated radiance (W/m^2/sr)
            - 'facet_contributions': Array of per-facet radiance contributions
            - 'visible_facets': List of visible facet indices
            - 'geometry': Dictionary with viewing geometry information
    """
    # Initialize output
    n_facets = len(shape_model)
    facet_contributions = np.zeros(n_facets)
    
    # Get geometry at this timestep
    geometry = simulation.get_geometry_at_timestep(timestep)
    sun_direction = geometry['sun_direction']
    solar_distance_m = geometry['solar_distance_m']
    
    # Get ephemeris time for SPICE-based observer positions
    et = None
    if simulation.use_spice and simulation.spice_manager is not None:
        # Map timestep to ephemeris time
        if config.spice_update_frequency == 'per_timestep':
            idx = min(timestep, len(simulation.spice_et_times) - 1)
        else:
            idx = 0
        et = simulation.spice_et_times[idx]
    
    # Get visible facets from observer
    visible_facets = observer.get_visible_facets(shape_model, et=et)
    
    # Load lookup tables if needed
    epf_lut = None
    brdf_lut = None
    if 'thermal' in radiance_type or radiance_type == 'total':
        epf_lut = EPFLookupTable(config.emission_lut)
    if 'reflected' in radiance_type or radiance_type == 'total':
        brdf_lut = BRDFLookupTable(config.scattering_lut)
    
    # Calculate radiance for each visible facet
    total_radiance = 0.0
    
    for facet_idx in visible_facets:
        facet = shape_model[facet_idx]
        
        # Get facet temperature at this timestep
        temperature = thermal_data.temperatures[facet_idx, timestep]
        
        # Calculate thermal radiance
        thermal_contrib = 0.0
        if radiance_type in ['thermal', 'total'] and epf_lut is not None:
            thermal_contrib = calculate_facet_thermal_radiance(
                facet, temperature, observer, config.emissivity, epf_lut, et
            )
        
        # Calculate reflected radiance
        reflected_contrib = 0.0
        if radiance_type in ['reflected', 'total'] and brdf_lut is not None:
            reflected_contrib = calculate_facet_reflected_radiance(
                facet, sun_direction, solar_distance_m, observer,
                simulation.solar_luminosity, brdf_lut, et
            )
        
        # Total contribution from this facet
        facet_radiance = thermal_contrib + reflected_contrib
        facet_contributions[facet_idx] = facet_radiance
        total_radiance += facet_radiance
    
    # Compile geometry information
    geometry_info = {
        'sun_direction': sun_direction,
        'solar_distance_m': solar_distance_m,
        'timestep': timestep,
        'observer_position': observer.get_position(et=et)
    }
    
    return {
        'total_radiance': total_radiance,
        'facet_contributions': facet_contributions,
        'visible_facets': visible_facets,
        'geometry': geometry_info
    }


def calculate_facet_thermal_radiance(
    facet,
    temperature: float,
    observer,
    emissivity: float,
    epf_lut: EPFLookupTable,
    et: Optional[float] = None
) -> float:
    """
    Calculate thermal radiance from a single facet.
    
    Args:
        facet: Facet object
        temperature: Surface temperature in Kelvin
        observer: Observer object
        emissivity: Surface emissivity
        epf_lut: Emission phase function lookup table
        et: Ephemeris time (for SPICE-based observer)
        
    Returns:
        Thermal radiance contribution in W/m^2/sr
    """
    # Calculate emission angle
    emission_angle = observer.calculate_emission_angle(
        facet.position, facet.normal, et
    )
    
    # Get EPF value
    epf_value = epf_lut.query(emission_angle)
    
    # Calculate projected area as seen from observer
    projected_area = observer.calculate_projected_area(
        facet.area, facet.position, facet.normal, et
    )
    
    if projected_area <= 0:
        return 0.0
    
    # Stefan-Boltzmann constant
    sigma = 5.670374419e-8
    
    # Thermal emission per unit area
    thermal_flux = emissivity * sigma * (temperature ** 4)
    
    # Apply emission phase function and projected area
    radiance = thermal_flux * epf_value * projected_area
    
    return radiance


def calculate_facet_reflected_radiance(
    facet,
    sun_direction: np.ndarray,
    solar_distance_m: float,
    observer,
    solar_luminosity: float,
    brdf_lut: BRDFLookupTable,
    et: Optional[float] = None
) -> float:
    """
    Calculate reflected solar radiance from a single facet.
    
    Args:
        facet: Facet object
        sun_direction: Unit vector pointing to Sun (in body frame)
        solar_distance_m: Distance to Sun in meters
        observer: Observer object
        solar_luminosity: Solar luminosity in Watts
        brdf_lut: BRDF lookup table
        et: Ephemeris time (for SPICE-based observer)
        
    Returns:
        Reflected radiance contribution in W/m^2/sr
    """
    # Check if facet is illuminated
    cos_incidence = np.dot(sun_direction, facet.normal)
    if cos_incidence <= 0:
        return 0.0
    
    # Calculate incidence angle
    incidence_angle = np.degrees(np.arccos(np.clip(cos_incidence, -1.0, 1.0)))
    
    # Calculate emission angle to observer
    emission_angle = observer.calculate_emission_angle(
        facet.position, facet.normal, et
    )
    
    # Get direction to observer
    to_observer = observer.get_direction_to_observer(facet.position, et)
    
    # Calculate azimuth angle between sun and observer
    # Project both vectors onto the facet plane
    sun_projected = sun_direction - np.dot(sun_direction, facet.normal) * facet.normal
    obs_projected = to_observer - np.dot(to_observer, facet.normal) * facet.normal
    
    # Normalize projected vectors
    sun_proj_norm = np.linalg.norm(sun_projected)
    obs_proj_norm = np.linalg.norm(obs_projected)
    
    if sun_proj_norm > 1e-10 and obs_proj_norm > 1e-10:
        sun_projected /= sun_proj_norm
        obs_projected /= obs_proj_norm
        cos_azimuth = np.dot(sun_projected, obs_projected)
        azimuth_angle = np.degrees(np.arccos(np.clip(cos_azimuth, -1.0, 1.0)))
    else:
        azimuth_angle = 0.0
    
    # Get BRDF value
    brdf_value = brdf_lut.query(incidence_angle, emission_angle, azimuth_angle)
    
    # Calculate solar flux at the body
    solar_flux = solar_luminosity / (4 * np.pi * solar_distance_m ** 2)
    
    # Calculate incident flux on facet
    incident_flux = solar_flux * cos_incidence
    
    # Calculate projected area
    projected_area = observer.calculate_projected_area(
        facet.area, facet.position, facet.normal, et
    )
    
    if projected_area <= 0:
        return 0.0
    
    # Reflected radiance using BRDF
    radiance = incident_flux * brdf_value * projected_area
    
    return radiance


def calculate_radiance_timeseries(
    shape_model,
    thermal_data,
    simulation,
    config,
    observer,
    radiance_type: str = 'thermal',
    timesteps: Optional[List[int]] = None
) -> Dict:
    """
    Calculate observed radiance over multiple timesteps.
    
    Args:
        shape_model: List of Facet objects
        thermal_data: ThermalData object
        simulation: Simulation object
        config: Config object
        observer: Observer object
        radiance_type: Type of radiance ('thermal', 'reflected', or 'total')
        timesteps: List of timesteps to compute (None = all timesteps)
        
    Returns:
        Dictionary with:
            - 'times': Array of timesteps
            - 'radiances': Array of total radiance values
            - 'facet_contributions_array': 2D array (timesteps x facets)
    """
    if timesteps is None:
        timesteps = list(range(simulation.timesteps_per_day))
    
    n_timesteps = len(timesteps)
    n_facets = len(shape_model)
    
    radiances = np.zeros(n_timesteps)
    facet_contributions_array = np.zeros((n_timesteps, n_facets))
    
    for i, t in enumerate(timesteps):
        result = calculate_observed_radiance(
            shape_model, thermal_data, simulation, config,
            observer, timestep=t, radiance_type=radiance_type
        )
        radiances[i] = result['total_radiance']
        facet_contributions_array[i, :] = result['facet_contributions']
    
    return {
        'times': np.array(timesteps),
        'radiances': radiances,
        'facet_contributions_array': facet_contributions_array
    }


def calculate_disk_integrated_flux(
    shape_model,
    thermal_data,
    simulation,
    config,
    observer,
    timestep: int = 0,
    radiance_type: str = 'thermal'
) -> float:
    """
    Calculate disk-integrated flux at the observer location.
    
    This is the total power per unit area received by the observer from
    the entire visible disk of the target body.
    
    Args:
        shape_model: List of Facet objects
        thermal_data: ThermalData object
        simulation: Simulation object
        config: Config object
        observer: Observer object
        timestep: Timestep index
        radiance_type: Type of radiance to calculate
        
    Returns:
        Disk-integrated flux in W/m^2
    """
    result = calculate_observed_radiance(
        shape_model, thermal_data, simulation, config,
        observer, timestep, radiance_type
    )
    
    # The total_radiance already accounts for projected areas and solid angles
    # This is the flux received by the observer
    return result['total_radiance']

