"""
Radiance Retrieval Script for TEMPEST

This script retrieves radiance measurements from TEMPEST temperature output data
using SPICE geometry and FOV-based facet queries.

Usage:
    python retrieve_radiance.py --config private/data/config/radiance_retrievals/Bennu_OTES.yaml
"""

import os
import sys
import argparse
import numpy as np
import h5py
from pathlib import Path
try:
    from scipy.constants import h, c, k  # Planck constant, speed of light, Boltzmann constant
except ImportError:
    # Fallback if scipy not available
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 299792458.0     # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (J/K)
import spiceypy as spice
from stl import mesh as stl_mesh_module

# Add src to path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

from src.utilities.config import Config
from src.utilities.locations import Locations
from src.utilities.utils import rays_triangles_intersection
from tempest import read_shape_model


def planck_radiance(wavelengths, temperature, emissivity=1.0):
    """
    Calculate Planck black body radiance spectrum.
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelengths in meters
    temperature : float or np.ndarray
        Temperature(s) in Kelvin
    emissivity : float
        Surface emissivity (default 1.0)
    
    Returns:
    --------
    np.ndarray
        Spectral radiance in W/(m^2 sr m)
    """
    wavelengths = np.asarray(wavelengths)
    temperature = np.asarray(temperature)
    
    # Avoid division by zero
    wavelengths = np.maximum(wavelengths, 1e-10)
    temperature = np.maximum(temperature, 1e-10)
    
    # Planck's law: B(λ, T) = (2πhc²/λ⁵) / (exp(hc/(λkT)) - 1)
    # Returns radiance in W/(m^2 sr m)
    
    # Handle scalar vs array temperature
    if temperature.ndim == 0 or temperature.size == 1:
        # Scalar temperature
        temp_val = float(temperature) if temperature.ndim == 0 else float(temperature[0])
        # Calculate exponent, handling overflow
        exponent = h * c / (wavelengths * k * temp_val)
        # Clip exponent to prevent overflow (exp(700) ~ 1e304, beyond float64 range)
        exponent_clipped = np.clip(exponent, None, 700.0)
        exp_term = np.exp(exponent_clipped)
        # When exponent is very large, exp_term >> 1, so radiance ≈ 0
        # Use np.where to handle overflow cases gracefully
        radiance = np.where(exponent > 700.0, 0.0, 
                           (2 * np.pi * h * c**2) / (wavelengths**5 * (exp_term - 1)))
        radiance = emissivity * radiance
        return radiance
    else:
        # Array of temperatures
        exponent = h * c / (wavelengths[:, None] * k * temperature[None, :])
        exponent_clipped = np.clip(exponent, None, 700.0)
        exp_term = np.exp(exponent_clipped)
        radiance = np.where(exponent > 700.0, 0.0,
                           (2 * np.pi * h * c**2) / (wavelengths[:, None]**5 * (exp_term - 1)))
        radiance = emissivity * radiance
        return radiance


def load_tempest_output(config):
    """
    Load TEMPEST output data from HDF5 file.
    
    Parameters:
    -----------
    config : Config
        Configuration object with tempest_output_path
    
    Returns:
    --------
    dict
        Dictionary containing:
        - shape_model: list of Facet objects
        - subfacet_temps: array of sub-facet temperatures (n_facets, n_subfacets, timesteps)
        - dome_flux_th: dome thermal flux data
        - simulation_params: dict with simulation parameters
    """
    loc = Locations()
    
    # Get path to TEMPEST output
    if hasattr(config, 'tempest_output_path') and config.tempest_output_path:
        h5_path = tempest_output_path
    else:
        # Try to find most recent output
        remote_outputs = loc.remote_outputs
        animation_dirs = [d for d in os.listdir(remote_outputs) 
                         if d.startswith('animation_outputs_')] if os.path.exists(remote_outputs) else []
        if animation_dirs:
            latest_dir = max(animation_dirs)
            h5_files = [f for f in os.listdir(os.path.join(remote_outputs, latest_dir))
                       if f.startswith('combined_animation_data_rough_')]
            if h5_files:
                h5_path = os.path.join(remote_outputs, latest_dir, h5_files[0])
            else:
                raise FileNotFoundError("No TEMPEST output HDF5 file found")
        else:
            raise FileNotFoundError("No TEMPEST output directory found")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"TEMPEST output file not found: {h5_path}")
    
    print(f"Loading TEMPEST output from: {h5_path}")
    
    data = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Load subfacet data
        if 'subfacet_data' in f:
            subgrp = f['subfacet_data']
            data['subfacet_points'] = subgrp['points'][:]
            data['subfacet_faces'] = subgrp['faces'][:]
            data['subfacet_temps'] = subgrp['temps'][:]  # Final timestep temps (for backward compatibility)
            # Load full timestep sub-facet temperatures if available
            if 'temps_full' in subgrp:
                data['subfacet_temps_full'] = subgrp['temps_full'][:]  # Shape: (n_facets, N_subfacets, T)
                data['n_subfacets'] = int(subgrp.attrs.get('n_subfacets', 0))
        
        # Load dome flux data (contains full timestep data)
        if 'dome_fluxes' in f:
            domegrp = f['dome_fluxes']
            data['dome_flux_th'] = domegrp['dome_flux_th'][:]  # Shape: (n_facets, M, T)
            data['dome_normals'] = domegrp['dome_normals'][:]
            data['dome_bin_areas'] = domegrp['dome_bin_areas'][:]
            if 'dome_bin_solid_angles' in domegrp:
                data['dome_bin_solid_angles'] = domegrp['dome_bin_solid_angles'][:]
        
        # Load simulation parameters
        if 'animation_params' in f:
            paramgrp = f['animation_params']
            data['simulation_params'] = {
                'rotation_axis': paramgrp['rotation_axis'][:],
                'sunlight_direction': paramgrp['sunlight_direction'][:],
                'timesteps_per_day': int(paramgrp['timesteps_per_day'][()]),
                'solar_distance_au': float(paramgrp['solar_distance_au'][()]),
                'rotation_period_hours': float(paramgrp['rotation_period_hours'][()]),
                'emissivity': float(paramgrp['emissivity'][()])
            }
            # Get dome radius factor if available
            if 'dome_radius_factor' in paramgrp.attrs:
                data['dome_radius_factor'] = float(paramgrp.attrs['dome_radius_factor'])
            elif 'kernel_dome_radius_factor' in paramgrp.attrs:
                data['dome_radius_factor'] = float(paramgrp.attrs['kernel_dome_radius_factor'])
        
        # Check for parent facet temperatures in animation_io
        if 'animation_io' in f:
            iogrp = f['animation_io']
            if 'plotted_variable_array' in iogrp:
                # This is the parent facet final day temperatures
                data['parent_temperatures'] = iogrp['plotted_variable_array'][:]  # Shape: (n_facets, T)
    
    # Load shape model from TEMPEST config
    tempest_config_path = config.config_data.get('tempest_config_path')
    if not tempest_config_path:
        raise ValueError("tempest_config_path must be specified in radiance retrieval config")
    tempest_config = Config(config_path=tempest_config_path)
    shape_model = read_shape_model(
        tempest_config.path_to_shape_model_file,
        data['simulation_params']['timesteps_per_day'],
        1,  # n_layers (not needed for radiance)
        1,  # max_days (not needed)
        False  # calculate_energy_terms
    )
    
    # If roughness was used, generate depressions to initialize canonical mesh and dome_rotation
    # This is needed to access Facet._canonical_subfacet_mesh and facet.dome_rotation for geometry reconstruction
    if config.apply_kernel_based_roughness if hasattr(config, 'apply_kernel_based_roughness') else False:
        from src.model.simulation import Simulation
        simulation = Simulation(tempest_config)
        # Generate depressions for all facets to set up dome_rotation matrices
        # Canonical mesh is a class attribute (set once), but each facet needs its own dome_rotation
        for facet in shape_model:
            facet.generate_spherical_depression(config, simulation)
    
    data['shape_model'] = shape_model
    
    return data


def generate_fov_rays(observer_position, pointing_direction, fov_angle_rad, n_rays=1000):
    """
    Generate rays within a circular FOV with uniform sensitivity.
    
    Parameters:
    -----------
    observer_position : np.ndarray(3)
        Position of observer in target body frame (meters)
    pointing_direction : np.ndarray(3)
        Pointing direction (normalized) in target body frame
    fov_angle_rad : float
        Field of view half-angle in radians
    n_rays : int
        Number of rays to generate for FOV sampling
    
    Returns:
    --------
    np.ndarray(n_rays, 3)
        Ray directions in target body frame (normalized)
    """
    # Normalize pointing direction
    pointing_direction = pointing_direction / np.linalg.norm(pointing_direction)
    
    # Create orthonormal basis for FOV plane
    # Use a perpendicular vector to create the basis
    if abs(pointing_direction[0]) < 0.9:
        perp1 = np.array([1.0, 0.0, 0.0])
    else:
        perp1 = np.array([0.0, 1.0, 0.0])
    
    # Gram-Schmidt to create orthonormal basis
    u1 = perp1 - np.dot(perp1, pointing_direction) * pointing_direction
    u1_norm = np.linalg.norm(u1)
    if u1_norm > 1e-10:
        u1 = u1 / u1_norm
    else:
        # If pointing is along perp1, use different vector
        u1 = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), pointing_direction) * pointing_direction
        u1 = u1 / np.linalg.norm(u1)
    
    u2 = np.cross(pointing_direction, u1)
    u2 = u2 / np.linalg.norm(u2)
    
    # Generate uniform random points in circular FOV
    # Use uniform sampling in radius^2 and angle for uniform area density
    r_squared = np.random.rand(n_rays) * (np.sin(fov_angle_rad) ** 2)
    r = np.sqrt(r_squared)
    theta = 2 * np.pi * np.random.rand(n_rays)
    
    # Convert to Cartesian coordinates in FOV plane
    x_local = r * np.cos(theta)
    y_local = r * np.sin(theta)
    # z component ensures unit length (pointing along FOV cone)
    z_local = np.sqrt(1 - r_squared)
    
    # Transform to target body frame
    rays = (x_local[:, None] * u1[None, :] + 
            y_local[:, None] * u2[None, :] + 
            z_local[:, None] * pointing_direction[None, :])
    
    # Normalize rays (should already be unit length, but ensure)
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    rays = rays / norms
    
    return rays


def find_visible_facets_in_fov(shape_model, observer_position, pointing_direction, 
                                fov_angle_rad, n_rays=1000):
    """
    Find facets visible within the observer's FOV.
    
    Parameters:
    -----------
    shape_model : list
        List of Facet objects
    observer_position : np.ndarray(3)
        Observer position in target body frame (meters)
    pointing_direction : np.ndarray(3)
        Pointing direction (normalized) in target body frame
    fov_angle_rad : float
        Field of view half-angle in radians
    n_rays : int
        Number of rays for FOV sampling
    
    Returns:
    --------
    dict
        Dictionary with:
        - visible_facet_indices: list of facet indices
        - facet_contributions: dict mapping facet_idx to (area_fraction, distances)
    """
    # Generate FOV rays
    ray_directions = generate_fov_rays(observer_position, pointing_direction, fov_angle_rad, n_rays)
    
    # Get all facet vertices
    all_vertices = np.array([f.vertices for f in shape_model])
    
    # Check intersections
    # rays_triangles_intersection expects ray_origin as (3,) array, not (1, 3)
    intersections, t_values = rays_triangles_intersection(
        observer_position,  # Shape (3,) - single origin for all rays
        ray_directions,      # Shape (n_rays, 3) - multiple ray directions
        all_vertices         # Shape (n_facets, 3, 3) - triangle vertices
    )
    
    # Find which facets are hit
    visible_facets = {}
    facet_contributions = {}
    
    for i, facet in enumerate(shape_model):
        # Check if any rays hit this facet
        hits = intersections[:, i]
        if np.any(hits):
            distances = t_values[hits, i]
            # Count hits and calculate average distance
            n_hits = np.sum(hits)
            avg_distance = np.mean(distances)
            
            # Calculate solid angle contribution
            # For uniform FOV sensitivity, weight by number of hits
            area_fraction = n_hits / n_rays
            
            visible_facets[i] = {
                'n_hits': n_hits,
                'avg_distance': avg_distance,
                'area_fraction': area_fraction
            }
            facet_contributions[i] = (area_fraction, avg_distance)
    
    return {
        'visible_facet_indices': list(visible_facets.keys()),
        'facet_contributions': facet_contributions
    }


def spice_geometry(config):
    """
    Use SPICE to calculate geometry for radiance retrieval.
    
    Parameters:
    -----------
    config : Config
        Configuration object with SPICE parameters
    
    Returns:
    --------
    dict
        Dictionary containing:
        - timestep: int (TEMPEST timestep index)
        - target_position: np.ndarray(3) in target body frame
        - target_orientation: np.ndarray(3,3) rotation matrix
        - observer_position: np.ndarray(3) in target body frame
        - observer_pointing: np.ndarray(3) pointing direction in target body frame
        - fov_angle_rad: float (FOV half-angle in radians)
    """
    # Load SPICE kernels
    kernel_paths = config.config_data.get('spice_kernels', [])
    for kernel_path in kernel_paths:
        if not os.path.exists(kernel_path):
            raise FileNotFoundError(f"SPICE kernel not found: {kernel_path}")
        spice.furnsh(kernel_path)
    
    try:
        # Parse time string
        observation_time = config.config_data.get('observation_time')
        if not observation_time:
            raise ValueError("observation_time must be specified in config")
        et = spice.str2et(observation_time)
        
        # Get target and observer IDs
        target_id = config.config_data.get('target_id')
        observer_id = config.config_data.get('observer_id')
        target_frame = config.config_data.get('target_frame')
        observer_frame = config.config_data.get('observer_frame', 'J2000')
        
        if not all([target_id, observer_id, target_frame]):
            raise ValueError("target_id, observer_id, and target_frame must be specified in config")
        
        # Get target position and orientation
        # Position of target center in observer frame
        target_state, lt = spice.spkezr(target_id, et, observer_frame, 'NONE', observer_id)
        target_position_obs_frame = target_state[:3]  # km
        
        # Get target orientation (rotation matrix from target frame to observer frame)
        target_rotation_matrix = spice.pxform(target_frame, observer_frame, et)
        
        # Convert observer position to target body frame
        # Observer position in target frame = -target position in observer frame (rotated)
        observer_position_target_frame = -target_rotation_matrix.T @ target_position_obs_frame
        
        # Convert from km to meters
        observer_position_target_frame = observer_position_target_frame * 1000.0
        
        # Get observer pointing direction
        # Priority: pointing_latlon > instrument_id > pointing_target > pointing_vector
        
        # Option 1: Lat/lon on target surface (from observation geometry files)
        pointing_latlon = config.config_data.get('pointing_latlon')
        if pointing_latlon is not None:
            lat_deg, lon_deg = pointing_latlon
            print(f"  Using pointing from surface coordinates: lat={lat_deg}°, lon={lon_deg}°")
            
            # Convert lat/lon to Cartesian in target body-fixed frame
            lat_rad = np.radians(lat_deg)
            lon_rad = np.radians(lon_deg)
            
            # Spherical to Cartesian (assuming spherical body)
            # In IAU body-fixed frame: +X through 0°lat/0°lon, +Z through north pole
            radius = 245.0  # Bennu mean radius in meters (approximate)
            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)
            surface_point_target_frame = np.array([x, y, z])
            
            # Pointing direction: from observer to surface point
            pointing_direction_target_frame = surface_point_target_frame - observer_position_target_frame
            pointing_direction_target_frame = pointing_direction_target_frame / np.linalg.norm(pointing_direction_target_frame)
            
            # Convert to observer frame for consistency with other methods
            pointing_direction_obs_frame = target_rotation_matrix @ pointing_direction_target_frame
            
            print(f"    Surface point (target frame): [{surface_point_target_frame[0]:.1f}, {surface_point_target_frame[1]:.1f}, {surface_point_target_frame[2]:.1f}] m")
            print(f"  ✓ Pointing determined from observation geometry")
            
        # Option 2: Use instrument kernel
        elif config.config_data.get('instrument_id') is not None:
            instrument_id = config.config_data.get('instrument_id')
            print(f"  Getting pointing from instrument kernel: {instrument_id}")
        
            # Check if instrument ID exists
            inst_naif_id = spice.bodn2c(instrument_id)
            print(f"    Instrument NAIF ID: {inst_naif_id}")
            
            # Get instrument frame ID
            inst_frame = spice.frmnam(inst_naif_id)
            print(f"    Instrument frame: {inst_frame}")
            
            if not inst_frame:
                raise ValueError(f"Instrument frame not found for {instrument_id}. Check that IK kernel is loaded.")
            
            # Use GETFOV to get boresight direction directly from instrument kernel
            try:
                shape, frame, boresight_inst_frame, bounds, n = spice.getfov(inst_naif_id, 100, 32, 32)
                print(f"    Using GETFOV: boresight in {frame} frame = [{boresight_inst_frame[0]:.3f}, {boresight_inst_frame[1]:.3f}, {boresight_inst_frame[2]:.3f}]")
                # Transform boresight from instrument frame to observer frame
                inst_to_obs = spice.pxform(frame, observer_frame, et)
                pointing_direction_obs_frame = spice.mxv(inst_to_obs, boresight_inst_frame)
            except Exception as getfov_error:
                # Fallback: use standard +Z axis assumption
                print(f"    GETFOV failed ({str(getfov_error)[:100]}), using +Z axis assumption")
                inst_to_obs = spice.pxform(inst_frame, observer_frame, et)
                boresight_inst = np.array([0.0, 0.0, 1.0])
                pointing_direction_obs_frame = inst_to_obs @ boresight_inst
            
            pointing_direction_obs_frame = pointing_direction_obs_frame / np.linalg.norm(pointing_direction_obs_frame)
            
            # Verify pointing makes sense: if instrument collected data, it should be pointing at target
            # Check angle between pointing and target direction
            target_dir = -target_position_obs_frame / np.linalg.norm(target_position_obs_frame)
            cos_angle = np.dot(pointing_direction_obs_frame, target_dir)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            
            # Diagnostic information
            print(f"    Pointing direction (observer frame): [{pointing_direction_obs_frame[0]:.3f}, {pointing_direction_obs_frame[1]:.3f}, {pointing_direction_obs_frame[2]:.3f}]")
            print(f"    Target direction (observer frame): [{target_dir[0]:.3f}, {target_dir[1]:.3f}, {target_dir[2]:.3f}]")
            print(f"    Angle between pointing and target: {angle_deg:.2f}°")
            
            # If pointing is > 90° from target, warn but continue (CK may not have accurate pointing for all observations)
            if angle_deg > 90.0:
                print(f"    ⚠️  Warning: Instrument pointing {angle_deg:.1f}° from target")
                print(f"    CK kernel may not have accurate pointing for this observation")
                print(f"    Consider using pointing_latlon from observation geometry file instead")
            
            print(f"  ✓ Successfully using pointing from instrument kernel: {instrument_id}")
        
        # Option 3: pointing_target or pointing_vector (handled below)
        else:
            pointing_direction_obs_frame = None
        
        # Convert pointing direction to target body frame (if not already set from lat/lon)
        # Note: pointing_direction_obs_frame points FROM observer TO target
        # In target frame, this should point FROM observer position TO origin
        if pointing_latlon is None:  # Only convert if not already in target frame from lat/lon
            pointing_direction_target_frame = target_rotation_matrix.T @ pointing_direction_obs_frame
        
        # Verify: pointing should point toward origin (negative of observer position direction)
        # If dot product is negative, this indicates a frame transformation error
        obs_dir_to_origin = -observer_position_target_frame / np.linalg.norm(observer_position_target_frame)
        dot_product = np.dot(pointing_direction_target_frame, obs_dir_to_origin)
        if dot_product < -0.1:  # Allow small tolerance for numerical errors
            raise ValueError(
                f"Pointing direction appears reversed after frame transformation.\n"
                f"This indicates a geometry calculation error.\n"
                f"Pointing in target frame: [{pointing_direction_target_frame[0]:.3f}, {pointing_direction_target_frame[1]:.3f}, {pointing_direction_target_frame[2]:.3f}]\n"
                f"Direction to origin: [{obs_dir_to_origin[0]:.3f}, {obs_dir_to_origin[1]:.3f}, {obs_dir_to_origin[2]:.3f}]\n"
                f"Dot product: {dot_product:.6f} (should be ~1.0)"
            )
        
        # Calculate geometry information for output
        observer_distance_km = np.linalg.norm(target_position_obs_frame)
        observer_distance_m = observer_distance_km * 1000.0
        
        # Calculate phase angle (angle between Sun-Target-Observer)
        # Get Sun position relative to target
        try:
            sun_state, _ = spice.spkezr("SUN", et, observer_frame, 'NONE', target_id)
            sun_position_obs_frame = sun_state[:3]  # km
            # Sun direction from target (in observer frame)
            sun_direction = sun_position_obs_frame / np.linalg.norm(sun_position_obs_frame)
            # Observer direction from target (in observer frame)
            observer_direction = -target_position_obs_frame / np.linalg.norm(target_position_obs_frame)
            # Phase angle = angle between sun direction and observer direction
            cos_phase = np.clip(np.dot(sun_direction, observer_direction), -1.0, 1.0)
            phase_angle_deg = np.degrees(np.arccos(cos_phase))
        except Exception as e:
            print(f"  Warning: Could not calculate phase angle: {e}")
            phase_angle_deg = None
        
        # Calculate emission angle (angle between surface normal and observer direction)
        # This will be calculated per-facet in retrieve_radiance, but we can show average here
        emission_angle_deg = None  # Will be calculated per-facet
        
        # Get FOV angle
        fov_angle_degrees = config.config_data.get('fov_angle_degrees')
        if not fov_angle_degrees:
            raise ValueError("fov_angle_degrees must be specified in config")
        fov_angle_rad = np.radians(fov_angle_degrees / 2.0)  # Half-angle
        
        # Print geometry information
        print(f"\n  Geometry Information:")
        print(f"    Observer distance: {observer_distance_km:.2f} km ({observer_distance_m:.0f} m)")
        if phase_angle_deg is not None:
            print(f"    Phase angle: {phase_angle_deg:.2f}°")
        print(f"    Target position (observer frame): [{target_position_obs_frame[0]:.2f}, {target_position_obs_frame[1]:.2f}, {target_position_obs_frame[2]:.2f}] km")
        print(f"    Observer pointing direction (observer frame): [{pointing_direction_obs_frame[0]:.3f}, {pointing_direction_obs_frame[1]:.3f}, {pointing_direction_obs_frame[2]:.3f}]")
        print(f"    FOV half-angle: {np.degrees(fov_angle_rad):.2f}° (full angle: {fov_angle_degrees:.2f}°)")
        
        # Calculate TEMPEST timestep from rotation phase
        # If timestep is explicitly provided, use it
        if hasattr(config, 'timestep') and config.timestep is not None:
            timestep = config.timestep
        elif config.config_data.get('rotation_phase') is not None:
            # Calculate timestep from rotation phase (0-1)
            # This requires timesteps_per_day from TEMPEST output
            # For now, we'll calculate it after loading TEMPEST data
            timestep = None  # Will be calculated later
        else:
            # Default to first timestep
            timestep = 0
        
        # Target position in target body frame is at origin (body-centered)
        target_position_target_frame = np.array([0.0, 0.0, 0.0])
        
        # Target orientation is identity in body frame
        target_orientation_target_frame = np.eye(3)
        
        return {
            'timestep': timestep,
            'target_position': target_position_target_frame,
            'target_orientation': target_orientation_target_frame,
            'observer_position': observer_position_target_frame,
            'observer_pointing': pointing_direction_target_frame,
            'fov_angle_rad': fov_angle_rad,
            'observer_distance_m': observer_distance_m,
            'phase_angle_deg': phase_angle_deg
        }
    
    finally:
        # Unload kernels
        spice.kclear()


def retrieve_radiance(timestep, target_position, target_orientation, observer_position,
                     observer_pointing, fov_angle_rad, tempest_data, wavelengths,
                     use_roughness=True):
    """
    Retrieve radiance spectrum from TEMPEST data for given geometry.
    
    Parameters:
    -----------
    timestep : int
        TEMPEST timestep index
    target_position : np.ndarray(3)
        Target position in target body frame (not used, kept for API consistency)
    target_orientation : np.ndarray(3,3)
        Target orientation matrix (not used, kept for API consistency)
    observer_position : np.ndarray(3)
        Observer position in target body frame (meters)
    observer_pointing : np.ndarray(3)
        Observer pointing direction (normalized) in target body frame
    fov_angle_rad : float
        Field of view half-angle in radians
    tempest_data : dict
        TEMPEST output data from load_tempest_output()
    wavelengths : np.ndarray
        Wavelengths for radiance spectrum (meters)
    use_roughness : bool
        Whether to use roughness sub-facet temperatures (default True)
    
    Returns:
    --------
    np.ndarray
        Spectral radiance in W/(m^2 sr m) for each wavelength
    """
    shape_model = tempest_data['shape_model']
    simulation_params = tempest_data['simulation_params']
    emissivity = simulation_params['emissivity']
    
    # Find visible facets in FOV
    fov_results = find_visible_facets_in_fov(
        shape_model,
        observer_position,
        observer_pointing,
        fov_angle_rad,
        n_rays=1000
    )
    
    visible_indices = fov_results['visible_facet_indices']
    facet_contributions = fov_results['facet_contributions']
    
    if len(visible_indices) == 0:
        print("Warning: No facets visible in FOV")
        return np.zeros_like(wavelengths)
    
    print(f"Found {len(visible_indices)} visible facets in FOV")
    
    # Calculate average emission angle for visible facets
    if len(visible_indices) > 0:
        emission_angles = []
        for facet_idx in visible_indices[:min(10, len(visible_indices))]:  # Sample first 10
            facet = shape_model[facet_idx]
            facet_to_observer = observer_position - facet.position
            facet_to_observer_norm = facet_to_observer / np.linalg.norm(facet_to_observer)
            cos_emission = np.dot(facet.normal, facet_to_observer_norm)
            if cos_emission > 0:
                emission_angles.append(np.degrees(np.arccos(np.clip(cos_emission, 0, 1))))
        if emission_angles:
            avg_emission_angle = np.mean(emission_angles)
            print(f"  Average emission angle: {avg_emission_angle:.2f}°")
    
    # Initialize radiance spectrum
    radiance_spectrum = np.zeros_like(wavelengths)
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant
    
    if use_roughness:
        # Use sub-facet temperatures directly (domes are for self-heating, not radiance)
        if 'subfacet_temps_full' not in tempest_data:
            raise ValueError(
                "Full timestep sub-facet temperatures not found in TEMPEST output. "
                "This is required for roughness radiance calculation. "
                "Ensure TEMPEST was run with roughness enabled and saved sub-facet data. "
                "The output file should contain 'subfacet_data/temps_full' dataset."
            )
        
        subfacet_temps_full = tempest_data['subfacet_temps_full']  # Shape: (n_facets, N_subfacets, T)
        n_subfacets = tempest_data.get('n_subfacets', subfacet_temps_full.shape[1])
        dome_radius_factor = tempest_data.get('dome_radius_factor', 100.0)
        
        # Get canonical sub-facet mesh (needed to reconstruct geometry)
        from src.model.facet import Facet
        
        # Ensure canonical mesh is initialized
        if not hasattr(Facet, '_canonical_subfacet_mesh') or Facet._canonical_subfacet_mesh is None:
            # Initialize mesh if not already done (should have been done in load_tempest_output)
            if len(shape_model) > 0 and hasattr(shape_model[0], 'dome_rotation'):
                # Mesh should already be initialized, but check anyway
                raise ValueError(
                    "Canonical sub-facet mesh not initialized. "
                    "This should be set when loading TEMPEST output. "
                    "Ensure TEMPEST config matches the output file."
                )
            else:
                raise ValueError(
                    "Shape model not properly initialized with roughness. "
                    "Ensure TEMPEST was run with roughness enabled."
                )
        canonical_mesh = Facet._canonical_subfacet_mesh
        
        # For each visible facet, query visible sub-facets and calculate radiance
        for facet_idx in visible_indices:
            area_fraction, avg_distance = facet_contributions[facet_idx]
            facet = shape_model[facet_idx]
            
            # Get sub-facet temperatures for this facet at this timestep
            facet_subfacet_temps = subfacet_temps_full[facet_idx, :, timestep]  # Shape: (N_subfacets,)
            
            # Reconstruct sub-facet world coordinates from canonical mesh
            # Scale factor: dome_radius_factor * parent_radius
            parent_radius = np.sqrt(facet.area / np.pi)
            scale = dome_radius_factor * parent_radius
            
            # Calculate direction from facet center to observer
            facet_to_observer = observer_position - facet.position
            facet_to_observer_norm = facet_to_observer / np.linalg.norm(facet_to_observer)
            
            facet_radiance_contribution = np.zeros_like(wavelengths)
            
            # Check each sub-facet for visibility
            # Ensure we don't exceed the number of stored sub-facets
            n_stored_subfacets = len(facet_subfacet_temps)
            n_canonical_subfacets = len(canonical_mesh)
            
            max_subfacets = min(n_canonical_subfacets, n_stored_subfacets)
            
            # Warn once if there's a mismatch (canonical mesh regenerated with different params)
            if n_canonical_subfacets != n_stored_subfacets and facet_idx == visible_indices[0]:
                print(f"Note: Canonical mesh has {n_canonical_subfacets} sub-facets but TEMPEST output has {n_stored_subfacets}")
                print(f"  Using {max_subfacets} sub-facets (minimum) for radiance calculation")
            
            for subfacet_idx in range(max_subfacets):
                mesh_entry = canonical_mesh[subfacet_idx]
                # Transform canonical mesh to world coordinates
                local_tri = mesh_entry['vertices'] * scale
                world_tri = (facet.dome_rotation.dot(local_tri.T)).T + facet.position
                
                # Calculate sub-facet normal and center
                v1, v2, v3 = world_tri[0], world_tri[1], world_tri[2]
                subfacet_normal = np.cross(v2 - v1, v3 - v1)
                subfacet_normal_norm = np.linalg.norm(subfacet_normal)
                if subfacet_normal_norm > 1e-12:
                    subfacet_normal = subfacet_normal / subfacet_normal_norm
                else:
                    continue
                
                # Check if sub-facet is visible to observer
                subfacet_center = np.mean(world_tri, axis=0)
                subfacet_to_observer = observer_position - subfacet_center
                subfacet_to_observer_norm = subfacet_to_observer / np.linalg.norm(subfacet_to_observer)
                
                cos_emission = np.dot(subfacet_normal, subfacet_to_observer_norm)
                if cos_emission <= 0:
                    continue  # Sub-facet facing away from observer
                
                # Get sub-facet temperature and area
                subfacet_temp = facet_subfacet_temps[subfacet_idx]
                # Area calculation: canonical area (normalized) * parent area
                # The geometry is scaled by dome_radius_factor * parent_radius,
                # so area scales as (dome_radius_factor)^2
                # This matches TEMPEST's sub-facet area calculation for flux
                subfacet_area = mesh_entry['area'] * facet.area * (dome_radius_factor ** 2)
                
                # Calculate Planck radiance directly from temperature
                subfacet_radiance = planck_radiance(wavelengths, subfacet_temp, emissivity)
                
                # Weight by emission angle, solid angle, and area
                subfacet_distance = np.linalg.norm(subfacet_to_observer)
                solid_angle_factor = cos_emission / (subfacet_distance ** 2) if subfacet_distance > 0 else 0
                
                facet_radiance_contribution += subfacet_radiance * cos_emission * solid_angle_factor * subfacet_area
            
            # Weight by FOV area fraction and add to total spectrum
            radiance_spectrum += facet_radiance_contribution * area_fraction
    
    else:
        # Use parent facet temperatures (smooth surface mode)
        if 'parent_temperatures' not in tempest_data:
            raise ValueError(
                "parent_temperatures not found in TEMPEST output. "
                "This is required when use_roughness=False. "
                "Either run TEMPEST with roughness enabled, or ensure parent temperatures are saved."
            )
        parent_temps = tempest_data['parent_temperatures']  # Shape: (n_facets, T)
        
        for facet_idx in visible_indices:
            area_fraction, avg_distance = facet_contributions[facet_idx]
            facet = shape_model[facet_idx]
            
            facet_to_observer = observer_position - facet.position
            facet_to_observer_norm = facet_to_observer / np.linalg.norm(facet_to_observer)
            
            # Emission angle cosine
            cos_emission = np.dot(facet.normal, facet_to_observer_norm)
            if cos_emission <= 0:
                continue
            
            # Get temperature for this facet at this timestep
            if parent_temps is not None:
                temp = parent_temps[facet_idx, timestep]
            else:
                temp = 200.0  # Placeholder
            
            # Calculate Planck radiance
            facet_radiance = planck_radiance(wavelengths, temp, emissivity)
            
            # Weight by area fraction, solid angle, and emission angle
            solid_angle_factor = cos_emission / (avg_distance ** 2) if avg_distance > 0 else 0
            radiance_spectrum += facet_radiance * area_fraction * solid_angle_factor * facet.area
    
    return radiance_spectrum


def main():
    parser = argparse.ArgumentParser(description='Retrieve radiance from TEMPEST output')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--plot', action='store_true',
                       help='Plot the radiance spectrum after calculation')
    args = parser.parse_args()
    
    # Load configuration
    config = Config(config_path=args.config)
    
    # Load TEMPEST output data
    print("Loading TEMPEST output data...")
    tempest_data = load_tempest_output(config)
    
    # Get geometry from SPICE
    print("Calculating geometry from SPICE...")
    geometry = spice_geometry(config)
    
    # If timestep needs to be calculated from rotation phase, do it now
    rotation_phase = config.config_data.get('rotation_phase')
    if geometry['timestep'] is None and rotation_phase is not None:
        timesteps_per_day = tempest_data['simulation_params']['timesteps_per_day']
        geometry['timestep'] = int(rotation_phase * timesteps_per_day) % timesteps_per_day
    
    # Define wavelength range (e.g., for OTES: ~5-50 microns)
    wavelengths_config = config.config_data.get('wavelengths')
    if wavelengths_config is not None:
        wavelengths = np.array(wavelengths_config)
        # If wavelengths are < 1e-5, assume they're in microns and convert to meters
        if len(wavelengths) > 0 and np.all(wavelengths < 1e-5):
            wavelengths = wavelengths * 1e-6
        # If only 2 values provided, create array
        if len(wavelengths) == 2:
            wavelengths = np.linspace(wavelengths[0], wavelengths[1], 100)
    else:
        # Default: OTES wavelength range
        wavelengths = np.linspace(5e-6, 50e-6, 100)  # 5-50 microns in meters
    
    # Retrieve radiance
    print("Retrieving radiance...")
    radiance_spectrum = retrieve_radiance(
        geometry['timestep'],
        geometry['target_position'],
        geometry['target_orientation'],
        geometry['observer_position'],
        geometry['observer_pointing'],
        geometry['fov_angle_rad'],
        tempest_data,
        wavelengths,
        use_roughness=config.apply_kernel_based_roughness if hasattr(config, 'apply_kernel_based_roughness') else True
    )
    
    # Save results
    output_dir = os.path.join('output', 'radiance_retrievals')
    os.makedirs(output_dir, exist_ok=True)
    
    observation_time = config.config_data.get('observation_time', 'unknown_time')
    output_file = os.path.join(output_dir, f"radiance_{observation_time.replace(':', '-').replace(' ', '_')}.npz")
    np.savez(output_file,
             wavelengths=wavelengths,
             radiance=radiance_spectrum,
             geometry=geometry)
    
    print(f"\nRadiance retrieval complete!")
    print(f"Output saved to: {output_file}")
    print(f"Wavelength range: {wavelengths[0]*1e6:.1f} - {wavelengths[-1]*1e6:.1f} microns")
    print(f"Radiance range: {np.min(radiance_spectrum):.2e} - {np.max(radiance_spectrum):.2e} W/(m^2 sr m)")
    
    # Plot if requested
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Convert wavelengths to wavenumbers (cm⁻¹)
            # wavenumber = 1 / wavelength (in cm)
            wavenumbers = 1.0 / (wavelengths * 100)  # Convert meters to cm, then invert
            
            plt.figure(figsize=(10, 6))
            plt.plot(wavenumbers, radiance_spectrum, 'b-', linewidth=2)
            plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
            plt.ylabel('Radiance (W/(m² sr m))', fontsize=12)
            plt.title(f'Radiance Spectrum\nObservation: {observation_time}', fontsize=14)
            plt.grid(True, alpha=0.3)
            # Reverse x-axis so higher wavenumbers (shorter wavelengths) are on the right
            plt.xlim(wavenumbers[-1], wavenumbers[0])
            
            # Save plot
            plot_file = output_file.replace('.npz', '.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {plot_file}")
            plt.close()
        except ImportError:
            print("Warning: matplotlib not available, skipping plot")
        except Exception as e:
            print(f"Warning: Failed to create plot: {e}")


if __name__ == "__main__":
    main()
