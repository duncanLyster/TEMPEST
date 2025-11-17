"""
Spatial Selection Utilities for TEMPEST

This module provides functions to select facets based on latitude/longitude coordinates
and visualize the selected regions on the shape model.

Author: Duncan Lyster
Created: 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


def latlon_to_cartesian(lat_deg: float, lon_deg: float, radius: float, 
                        center: np.ndarray = None) -> np.ndarray:
    """
    Convert latitude/longitude coordinates to Cartesian coordinates.
    
    Convention:
    - Latitude: -90° (South Pole) to +90° (North Pole)
    - Longitude: 0° to 360° (Eastward)
    - Z-axis points to North Pole
    - X-axis points to 0° longitude
    - Y-axis completes right-handed coordinate system
    
    Args:
        lat_deg: Latitude in degrees (-90 to 90)
        lon_deg: Longitude in degrees (0 to 360)
        radius: Radius of the body in meters
        center: Center offset of the shape model (default: origin [0,0,0])
        
    Returns:
        Cartesian position as numpy array [x, y, z]
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return np.array([x, y, z]) + center


def cartesian_to_latlon(position: np.ndarray, center: np.ndarray = None) -> Tuple[float, float]:
    """
    Convert Cartesian coordinates to latitude/longitude.
    
    Args:
        position: Cartesian position as numpy array [x, y, z]
        center: Center offset of the shape model (default: origin [0,0,0])
        
    Returns:
        Tuple of (latitude_deg, longitude_deg)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    # Shift position relative to center
    relative_pos = position - center
    x, y, z = relative_pos
    r = np.linalg.norm(relative_pos)
    
    if r < 1e-10:
        return 0.0, 0.0
    
    lat_deg = np.degrees(np.arcsin(z / r))
    lon_deg = np.degrees(np.arctan2(y, x))
    
    # Normalize longitude to 0-360 range
    if lon_deg < 0:
        lon_deg += 360.0
        
    return lat_deg, lon_deg


def calculate_shape_model_center(shape_model) -> np.ndarray:
    """
    Calculate the center of mass of the shape model.
    
    Args:
        shape_model: List of Facet objects
        
    Returns:
        Center of mass as numpy array [x, y, z]
    """
    positions = np.array([f.position for f in shape_model])
    return np.mean(positions, axis=0)


def find_closest_facet(shape_model, lat_deg: float, lon_deg: float, 
                       body_radius: float, center: np.ndarray = None) -> Tuple[int, float]:
    """
    Find the facet whose center is closest to the given lat/lon position.
    
    Args:
        shape_model: List of Facet objects
        lat_deg: Target latitude in degrees
        lon_deg: Target longitude in degrees
        body_radius: Mean radius of the body in meters
        center: Center of the shape model (default: origin [0,0,0])
        
    Returns:
        Tuple of (facet_id, distance_in_meters)
    """
    target_pos = latlon_to_cartesian(lat_deg, lon_deg, body_radius, center)
    
    min_distance = float('inf')
    closest_facet_id = -1
    
    for i, facet in enumerate(shape_model):
        distance = np.linalg.norm(facet.position - target_pos)
        if distance < min_distance:
            min_distance = distance
            closest_facet_id = i
            
    return closest_facet_id, min_distance


def find_facets_in_radius(shape_model, lat_deg: float, lon_deg: float,
                          body_radius: float, search_radius_m: float, 
                          center: np.ndarray = None) -> List[int]:
    """
    Find all facets whose centers fall within a circular region around a lat/lon.
    
    The search first finds the closest facet to the target lat/lon, then uses that
    facet's actual surface position as the reference point. This accounts for
    irregular body shapes where the surface may deviate from the mean radius.
    
    Args:
        shape_model: List of Facet objects
        lat_deg: Center latitude in degrees
        lon_deg: Center longitude in degrees
        body_radius: Mean radius of the body in meters
        search_radius_m: Radius of search circle in meters
        center: Center of the shape model (default: origin [0,0,0])
        
    Returns:
        List of facet IDs within the search radius, sorted by distance
    """
    # First, find the closest facet to the target lat/lon
    # This gives us the actual surface position at that location
    closest_id, _ = find_closest_facet(shape_model, lat_deg, lon_deg, body_radius, center)
    reference_pos = shape_model[closest_id].position
    
    # Now find all facets within the radius from this actual surface position
    facets_in_radius = []
    
    for i, facet in enumerate(shape_model):
        distance = np.linalg.norm(facet.position - reference_pos)
        if distance <= search_radius_m:
            facets_in_radius.append((i, distance))
    
    # Sort by distance (closest first)
    facets_in_radius.sort(key=lambda x: x[1])
    
    # Return just the facet IDs
    return [fid for fid, _ in facets_in_radius]


def get_site_statistics(shape_model, facet_ids: List[int], 
                        body_radius: float) -> Dict:
    """
    Calculate statistics for a selected site.
    
    Args:
        shape_model: List of Facet objects
        facet_ids: List of selected facet IDs
        body_radius: Mean radius of the body in meters
        
    Returns:
        Dictionary with site statistics
    """
    if not facet_ids:
        return {
            'n_facets': 0,
            'total_area_m2': 0.0,
            'center_lat_deg': None,
            'center_lon_deg': None,
            'mean_lat_deg': None,
            'mean_lon_deg': None,
        }
    
    # Calculate center position (mean of all facet positions)
    positions = np.array([shape_model[i].position for i in facet_ids])
    center_pos = np.mean(positions, axis=0)
    center_lat, center_lon = cartesian_to_latlon(center_pos)
    
    # Calculate area-weighted mean position
    areas = np.array([shape_model[i].area for i in facet_ids])
    total_area = np.sum(areas)
    weighted_pos = np.sum(positions * areas[:, np.newaxis], axis=0) / total_area
    mean_lat, mean_lon = cartesian_to_latlon(weighted_pos)
    
    return {
        'n_facets': len(facet_ids),
        'total_area_m2': total_area,
        'center_lat_deg': center_lat,
        'center_lon_deg': center_lon,
        'mean_lat_deg': mean_lat,
        'mean_lon_deg': mean_lon,
        'facet_areas_m2': areas.tolist(),
    }


def select_landing_sites(shape_model, landing_sites: Dict, body_radius: float,
                         site_radius_m: float) -> Dict:
    """
    Select facets for multiple landing sites.
    
    Args:
        shape_model: List of Facet objects
        landing_sites: Dictionary of landing site specifications
            Each site should have 'lat' and 'lon' keys
        body_radius: Mean radius of the body in meters
        site_radius_m: Radius around each site in meters
        
    Returns:
        Dictionary mapping site names to selected facet IDs and statistics
    """
    results = {}
    
    for site_name, site_info in landing_sites.items():
        lat = site_info['lat']
        lon = site_info['lon']
        
        # Find facets in radius
        facet_ids = find_facets_in_radius(shape_model, lat, lon, 
                                          body_radius, site_radius_m)
        
        # Get statistics
        stats = get_site_statistics(shape_model, facet_ids, body_radius)
        
        results[site_name] = {
            'target_lat': lat,
            'target_lon': lon,
            'facet_ids': facet_ids,
            'stats': stats,
            'description': site_info.get('description', '')
        }
        
    return results


def print_site_summary(site_results: Dict):
    """
    Print a formatted summary of site selection results.
    
    Args:
        site_results: Output from select_landing_sites()
    """
    print("\n" + "="*80)
    print("LANDING SITE FACET SELECTION SUMMARY")
    print("="*80)
    
    for site_name, data in site_results.items():
        print(f"\n{site_name}:")
        print(f"  Description: {data['description']}")
        print(f"  Target Location: {data['target_lat']:.2f}°N, {data['target_lon']:.2f}°E")
        print(f"  Number of Facets: {data['stats']['n_facets']}")
        print(f"  Total Area: {data['stats']['total_area_m2']:.2f} m²")
        
        if data['stats']['n_facets'] > 0:
            print(f"  Geometric Center: {data['stats']['center_lat_deg']:.2f}°N, "
                  f"{data['stats']['center_lon_deg']:.2f}°E")
            print(f"  Area-Weighted Mean: {data['stats']['mean_lat_deg']:.2f}°N, "
                  f"{data['stats']['mean_lon_deg']:.2f}°E")
            print(f"  Facet IDs: {data['facet_ids'][:10]}{'...' if len(data['facet_ids']) > 10 else ''}")
    
    print("\n" + "="*80 + "\n")


def calculate_t4_mean_temperature(temperatures: np.ndarray) -> np.ndarray:
    """
    Calculate the T^4 mean temperature (effective temperature).
    
    This is used for bolometric observations where the observed flux is
    proportional to the sum of T^4 values.
    
    Args:
        temperatures: Array of temperatures (can be 1D or 2D)
            If 2D, shape should be (n_facets, n_timesteps)
            
    Returns:
        T^4 mean temperature(s)
    """
    t4_mean = np.mean(temperatures**4, axis=0 if temperatures.ndim > 1 else None)
    return t4_mean ** 0.25


def extract_site_temperatures(thermal_data, facet_ids: List[int]) -> np.ndarray:
    """
    Extract temperature data for selected facets.
    
    Args:
        thermal_data: ThermalData object from simulation
        facet_ids: List of facet IDs to extract
        
    Returns:
        Array of temperatures with shape (n_facets, n_timesteps)
    """
    n_timesteps = thermal_data.temperatures.shape[1]
    n_facets = len(facet_ids)
    
    site_temps = np.zeros((n_facets, n_timesteps))
    
    for i, facet_id in enumerate(facet_ids):
        # Extract surface temperature (layer 0) for all timesteps
        site_temps[i, :] = thermal_data.temperatures[facet_id, :, 0]
    
    return site_temps


def convert_to_local_time(timesteps_per_day: int, facet_normal: np.ndarray,
                         sunlight_direction: np.ndarray, rotation_axis: np.ndarray) -> np.ndarray:
    """
    Convert timestep indices to local time (in hours) for a facet.
    
    Local time 12:00 corresponds to local noon (sun directly overhead).
    
    Args:
        timesteps_per_day: Number of timesteps in one rotation
        facet_normal: Normal vector of the facet
        sunlight_direction: Direction vector to the sun
        rotation_axis: Rotation axis of the body
        
    Returns:
        Array of local times in hours (0-24) for each timestep
    """
    local_times = np.zeros(timesteps_per_day)
    
    for t in range(timesteps_per_day):
        # Calculate rotation angle at this timestep
        angle = (t / timesteps_per_day) * 2 * np.pi
        
        # Rotate facet normal to current orientation
        # Using Rodrigues' rotation formula
        k = rotation_axis / np.linalg.norm(rotation_axis)
        rotated_normal = (facet_normal * np.cos(angle) +
                         np.cross(k, facet_normal) * np.sin(angle) +
                         k * np.dot(k, facet_normal) * (1 - np.cos(angle)))
        
        # Calculate angle between rotated normal and sun direction
        cos_angle = np.dot(rotated_normal, sunlight_direction)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Project onto plane perpendicular to rotation axis
        proj_normal = rotated_normal - np.dot(rotated_normal, k) * k
        proj_sun = sunlight_direction - np.dot(sunlight_direction, k) * k
        
        # Normalize projections
        proj_normal_norm = np.linalg.norm(proj_normal)
        proj_sun_norm = np.linalg.norm(proj_sun)
        
        if proj_normal_norm > 1e-6 and proj_sun_norm > 1e-6:
            proj_normal = proj_normal / proj_normal_norm
            proj_sun = proj_sun / proj_sun_norm
            
            # Calculate local time angle
            cos_local = np.clip(np.dot(proj_normal, proj_sun), -1.0, 1.0)
            local_angle = np.arccos(cos_local)
            
            # Determine sign using cross product
            cross = np.cross(proj_sun, proj_normal)
            if np.dot(cross, k) < 0:
                local_angle = 2 * np.pi - local_angle
                
            # Convert to hours (0 = midnight, 12 = noon)
            local_times[t] = (local_angle / (2 * np.pi)) * 24.0
        else:
            # Facet is at pole or sun is along rotation axis
            local_times[t] = 12.0
    
    return local_times


def plot_site_temperatures(site_results: Dict, thermal_data, simulation,
                          output_dir: str = "plots/landing_sites/",
                          use_t4_mean: bool = True):
    """
    Plot temperature vs local time for landing sites.
    
    Args:
        site_results: Output from select_landing_sites()
        thermal_data: ThermalData object from simulation
        simulation: Simulation object
        output_dir: Directory to save plots
        use_t4_mean: If True, plot T^4 mean; if False, plot arithmetic mean
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    site_names = list(site_results.keys())
    
    for idx, site_name in enumerate(site_names):
        if idx >= 4:  # Only plot first 4 sites
            break
            
        ax = axes[idx]
        data = site_results[site_name]
        facet_ids = data['facet_ids']
        
        if not facet_ids:
            ax.text(0.5, 0.5, 'No facets selected', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(site_name)
            continue
        
        # Extract temperatures
        site_temps = extract_site_temperatures(thermal_data, facet_ids)
        
        # Calculate mean temperature
        if use_t4_mean:
            mean_temp = calculate_t4_mean_temperature(site_temps)
            temp_label = r'$\langle T^4 \rangle^{1/4}$'
        else:
            mean_temp = np.mean(site_temps, axis=0)
            temp_label = 'Mean Temperature'
        
        # Convert to local time (use first facet for reference)
        facet_normal = thermal_data.shape_model[facet_ids[0]].normal if hasattr(thermal_data, 'shape_model') else None
        
        if facet_normal is not None and hasattr(simulation, 'rotation_axis'):
            local_times = convert_to_local_time(
                simulation.timesteps_per_day,
                facet_normal,
                simulation.sunlight_direction,
                simulation.rotation_axis
            )
        else:
            # Fallback to timestep-based x-axis
            local_times = np.linspace(0, 24, len(mean_temp))
        
        # Plot
        ax.plot(local_times, mean_temp, 'b-', linewidth=2, label=temp_label)
        
        # Add individual facets (semi-transparent)
        for i, fid in enumerate(facet_ids[:10]):  # Plot max 10 individual facets
            ax.plot(local_times, site_temps[i], alpha=0.2, color='gray', linewidth=0.5)
        
        ax.set_xlabel('Local Time (hours)', fontsize=11)
        ax.set_ylabel('Temperature (K)', fontsize=11)
        ax.set_title(f"{site_name}\n({data['stats']['n_facets']} facets, "
                    f"{data['stats']['total_area_m2']:.1f} m²)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
        
        # Add local time markers
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axvline(x=6, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axvline(x=12, color='orange', linestyle='--', alpha=0.5, linewidth=0.5, label='Local Noon')
        ax.axvline(x=18, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        
        if idx == 0:
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'landing_sites_temperatures.png'), dpi=300)
    plt.close()
    
    print(f"Temperature plot saved to {output_dir}landing_sites_temperatures.png")
