import json
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from stl import mesh
from datetime import datetime
import h5py
from src.utilities.locations import Locations

def rotation_matrix(axis, theta):
    """Create rotation matrix around given axis by angle theta."""
    axis = [a / math.sqrt(sum(a**2 for a in axis)) for a in axis]
    a = math.cos(theta / 2.0)
    b, c, d = [-axis[i] * math.sin(theta / 2.0) for i in range(3)]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return [[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]

def calculate_observed_temperature(facet_id, view_dir_world, rotation_axis, 
                                 dome_flux_th, dome_bin_normals, dome_bin_areas, 
                                 facet_normals, timesteps_per_day, emissivity, 
                                 dome_radius_factor, specific_phase=None):
    """
    Calculate observed temperature for a facet from a specific viewing direction.
    
    Args:
        facet_id: ID of the facet to analyze
        view_dir_world: Viewing direction in world coordinates (3D vector)
        rotation_axis: Rotation axis of the body
        dome_flux_th: Dome thermal flux data (n_facets, M, T)
        dome_bin_normals: Dome bin normal vectors (M, 3)
        dome_bin_areas: Dome bin areas (n_facets, M)
        facet_normals: Facet normal vectors
        timesteps_per_day: Number of timesteps per day
        emissivity: Surface emissivity
        dome_radius_factor: Dome radius scaling factor
        specific_phase: If provided, only calculate for this phase (0-1), otherwise full rotation
        
    Returns:
        phases, temperatures: Arrays of rotation phases and corresponding temperatures
    """
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant
    epsilon = emissivity
    scale = dome_radius_factor ** 2
    horizon_tolerance = 1e-6
    
    # Normalize viewing direction
    view_dir_world = view_dir_world / np.linalg.norm(view_dir_world)
    
    if specific_phase is not None:
        # Calculate for a specific phase only
        phases = np.array([specific_phase])
        timesteps = [int(specific_phase * timesteps_per_day)]
    else:
        # Calculate for full rotation
        n_t = dome_flux_th.shape[2]
        phases = np.linspace(0, 1, n_t)
        timesteps = range(n_t)
    
    temperatures = np.zeros(len(phases))
    
    for i, j in enumerate(timesteps):
        # Body rotation at timestep j
        angle_j = 2 * math.pi * j / timesteps_per_day
        rot_j = np.array(rotation_matrix(rotation_axis, angle_j))
        
        # Rotate view direction into mesh frame (world->body)
        view_mesh = rot_j.T.dot(view_dir_world)
        
        # Find best dome bin for this viewing direction
        coss = np.clip(np.dot(view_mesh, dome_bin_normals.T), 0, None)
        b_j = np.argmax(coss)
        
        # Get flux and area for this bin
        F_j = dome_flux_th[facet_id, b_j, j]
        area_j = dome_bin_areas[facet_id, b_j]
        
        # Check if patch is visible and facet facing camera
        facet_norm_j = facet_normals[facet_id].dot(rot_j.T)
        horizon_val = np.dot(facet_norm_j, view_dir_world)
        
        # Calculate temperature if visible
        if area_j > 0 and coss[b_j] > horizon_tolerance and horizon_val < -horizon_tolerance:
            temperatures[i] = ((F_j / area_j * scale) / (epsilon * sigma)) ** 0.25
        else:
            temperatures[i] = 0.0  # Not visible
    
    return phases, temperatures

def generate_spherical_sweeps(rotation_axis, n_positions=36):
    """
    Generate viewing positions for realistic east-west (azimuth) and north-south (elevation) sweeps.
    
    Args:
        rotation_axis: The rotation axis vector
        n_positions: Number of viewing positions to generate for each sweep
        
    Returns:
        ew_directions: Array of east-west sweep viewing directions (n_positions, 3)
        ns_directions: Array of north-south sweep viewing directions (n_positions, 3)
        angles: Array of angles (n_positions,)
    """
    # Normalize rotation axis
    rot_axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)
    
    # Create a coordinate system where rot_axis is the "north pole"
    # Find two orthogonal vectors in the plane perpendicular to rotation axis
    if abs(rot_axis[2]) < 0.9:  # If not close to z-axis
        temp = np.array([0, 0, 1])
    else:  # If close to z-axis, use x-axis
        temp = np.array([1, 0, 0])
    
    # First vector in plane (call this "east" direction)
    east_dir = temp - np.dot(temp, rot_axis) * rot_axis
    east_dir = east_dir / np.linalg.norm(east_dir)
    
    # Second vector in plane (call this "north" direction)  
    north_dir = np.cross(rot_axis, east_dir)
    north_dir = north_dir / np.linalg.norm(north_dir)
    
    # Generate angles - use a range that covers most viewing angles
    angles = np.linspace(-90, 90, n_positions)  # From -90° to +90°
    
    ew_directions = np.zeros((n_positions, 3))
    ns_directions = np.zeros((n_positions, 3))
    
    for i, angle_deg in enumerate(angles):
        angle_rad = np.radians(angle_deg)
        
        # East-West sweep: Keep elevation constant (looking horizontally) 
        # and sweep azimuth from east to west
        # Use spherical coordinates: (r, theta, phi) where theta is azimuth, phi is elevation
        elevation = 30  # degrees above horizon - adjust this for good viewing
        azimuth = angle_deg + 90  # Offset so 0° corresponds to east
        
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)
        
        # Convert spherical to cartesian in our coordinate system
        # In our system: rot_axis is "up", east_dir is "east", north_dir is "north"
        x = np.cos(elev_rad) * np.cos(azim_rad)  # east component
        y = np.cos(elev_rad) * np.sin(azim_rad)  # north component  
        z = np.sin(elev_rad)                     # up component
        
        ew_directions[i] = x * east_dir + y * north_dir + z * rot_axis
        ew_directions[i] = ew_directions[i] / np.linalg.norm(ew_directions[i])
        
        # North-South sweep: Keep azimuth constant (looking in one direction)
        # and sweep elevation from south (low) to north (high)
        azimuth_fixed = 45  # degrees - looking northeast-ish for good facet visibility
        elevation_varying = angle_deg  # This varies from -90 to +90
        
        elev_rad = np.radians(elevation_varying)
        azim_rad = np.radians(azimuth_fixed)
        
        x = np.cos(elev_rad) * np.cos(azim_rad)
        y = np.cos(elev_rad) * np.sin(azim_rad)
        z = np.sin(elev_rad)
        
        ns_directions[i] = x * east_dir + y * north_dir + z * rot_axis
        ns_directions[i] = ns_directions[i] / np.linalg.norm(ns_directions[i])
    
    return ew_directions, ns_directions, angles

def plot_viewing_angle_analysis(json_file, npz_file, facet_id, n_positions=36, 
                               specific_phase=0.5, output_dir=None):
    """
    Create plots of observed temperature vs viewing angle for a specific facet.
    
    Args:
        json_file: Path to animation parameters JSON file
        npz_file: Path to animation parameters NPZ file
        facet_id: ID of the facet to analyze
        n_positions: Number of viewing positions around the rotation plane
        specific_phase: Phase of rotation to analyze (0-1, where 0.5 is noon)
        output_dir: Directory to save plots (if None, uses current directory)
    """
    # Load animation data
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    npz_data = np.load(npz_file, allow_pickle=True)
    
    # Extract parameters
    path_to_shape_model_file = json_data['args'][0]
    rotation_axis = npz_data['rotation_axis']
    kwargs = json_data['kwargs']
    timesteps_per_day = kwargs['timesteps_per_day']
    emissivity = kwargs.get('emissivity', 0.5)
    dome_radius_factor = kwargs.get('dome_radius_factor', 1.0)
    
    # Load shape model to get facet normals
    try:
        shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file)
    except Exception as e:
        print(f"Failed to load shape model: {e}")
        return
    
    # Compute facet normals
    facet_normals = np.zeros((shape_mesh.vectors.shape[0], 3))
    for i in range(shape_mesh.vectors.shape[0]):
        v1, v2, v3 = shape_mesh.vectors[i]
        normal = np.cross(v2 - v1, v3 - v1)
        facet_normals[i] = normal / np.linalg.norm(normal)
    
    # Check if facet_id is valid
    if facet_id >= len(facet_normals):
        print(f"Error: facet_id {facet_id} is out of range. Max facet ID: {len(facet_normals)-1}")
        return
    
    # Load dome flux data
    script_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    dome_flux_path = os.path.join(project_root, 'outputs', 'dome_fluxes.h5')
    
    if not os.path.exists(dome_flux_path):
        print(f"Error: Dome flux file not found at {dome_flux_path}")
        print("This analysis requires precomputed dome thermal flux data.")
        return
    
    with h5py.File(dome_flux_path, 'r') as dfh:
        dome_flux_th = dfh['dome_flux_th'][:]      # (n_facets, M, T)
        dome_bin_normals = dfh['dome_normals'][:]  # (M, 3)
        dome_bin_areas = dfh['dome_bin_areas'][:]  # (n_facets, M)
    
    # Generate viewing positions for east-west and north-south sweeps
    ew_directions, ns_directions, angles = generate_spherical_sweeps(rotation_axis, n_positions)
    
    # Calculate observed temperatures for each viewing direction
    print(f"Calculating observed temperatures for facet {facet_id} at phase {specific_phase}")
    print(f"Using {n_positions} viewing positions for each sweep (-90° to +90°)...")
    
    temperatures_east_west = np.zeros(n_positions)
    temperatures_north_south = np.zeros(n_positions)
    
    for i in range(n_positions):
        # East-West sweep
        _, temp_ew = calculate_observed_temperature(
            facet_id, ew_directions[i], rotation_axis, dome_flux_th, dome_bin_normals, 
            dome_bin_areas, facet_normals, timesteps_per_day, emissivity, 
            dome_radius_factor, specific_phase
        )
        temperatures_east_west[i] = temp_ew[0]
        
        # North-South sweep
        _, temp_ns = calculate_observed_temperature(
            facet_id, ns_directions[i], rotation_axis, dome_flux_th, dome_bin_normals, 
            dome_bin_areas, facet_normals, timesteps_per_day, emissivity, 
            dome_radius_factor, specific_phase
        )
        temperatures_north_south[i] = temp_ns[0]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: East-West sweep
    ax1.plot(angles, temperatures_east_west, 'b-o', markersize=4, linewidth=2)
    ax1.set_xlabel('Viewing Angle (degrees)')
    ax1.set_ylabel('Observed Temperature (K)')
    ax1.set_title(f'Facet {facet_id} - East to West Sweep\n(Phase = {specific_phase:.2f})')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-90, 90)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Center view')
    
    # Mark zero temperatures (not visible)
    zero_mask = temperatures_east_west == 0
    if np.any(zero_mask):
        ax1.scatter(angles[zero_mask], temperatures_east_west[zero_mask], 
                   c='red', s=20, marker='x', label='Not visible')
        ax1.legend()
    
    # Plot 2: North-South sweep
    ax2.plot(angles, temperatures_north_south, 'r-o', markersize=4, linewidth=2)
    ax2.set_xlabel('Elevation Angle (degrees)')
    ax2.set_ylabel('Observed Temperature (K)')
    ax2.set_title(f'Facet {facet_id} - North to South Sweep\n(Phase = {specific_phase:.2f})')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-90, 90)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Horizon level')
    
    # Mark zero temperatures (not visible)
    zero_mask = temperatures_north_south == 0
    if np.any(zero_mask):
        ax2.scatter(angles[zero_mask], temperatures_north_south[zero_mask], 
                   c='red', s=20, marker='x', label='Not visible')
        ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = os.getcwd()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'viewing_angle_analysis_facet{facet_id}_phase{specific_phase:.2f}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filepath}")
    
    # Also save data as CSV
    import pandas as pd
    data_dict = {
        'viewing_angle_deg': angles,
        'temperature_east_west_K': temperatures_east_west,
        'temperature_north_south_K': temperatures_north_south
    }
    df = pd.DataFrame(data_dict)
    csv_filename = f'viewing_angle_data_facet{facet_id}_phase{specific_phase:.2f}_{timestamp}.csv'
    csv_filepath = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_filepath, index=False)
    print(f"Data saved as: {csv_filepath}")
    
    plt.show()
    
    return angles, temperatures_east_west, temperatures_north_south

def interactive_analysis():
    """Interactive function to select facet and run viewing angle analysis."""
    # Get available animation folders from data output directory
    loc = Locations()
    base_dir = loc.remote_outputs
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return
    
    # Get folders sorted by modification time
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folders.sort(key=lambda f: os.path.getmtime(os.path.join(base_dir, f)), reverse=True)
    
    if not folders:
        print(f"No folders found in {base_dir}.")
        return
    
    # Show available folders
    print("Available animation folders:")
    for i, folder in enumerate(folders):
        print(f"{i}: {folder}")
    
    # Get folder selection
    while True:
        try:
            choice = int(input(f"Select folder (0-{len(folders)-1}): "))
            if 0 <= choice < len(folders):
                selected_folder = folders[choice]
                break
            else:
                print(f"Invalid selection. Please select 0-{len(folders)-1}.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Set file paths
    json_file = os.path.join(base_dir, selected_folder, 'animation_params.json')
    npz_file = os.path.join(base_dir, selected_folder, 'animation_params.npz')
    
    if not os.path.exists(json_file) or not os.path.exists(npz_file):
        print(f"Required files not found in {selected_folder}.")
        return
    
    # Get facet ID
    while True:
        try:
            facet_id = int(input("Enter facet ID to analyze: "))
            if facet_id >= 0:
                break
            else:
                print("Facet ID must be non-negative.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get phase
    while True:
        try:
            phase = float(input("Enter rotation phase (0-1, where 0.5 is noon): "))
            if 0 <= phase <= 1:
                break
            else:
                print("Phase must be between 0 and 1.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of viewing positions
    while True:
        try:
            n_pos = int(input("Enter number of viewing positions (default 36): ") or "36")
            if n_pos > 0:
                break
            else:
                print("Number of positions must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Run analysis
    output_dir = os.path.join(loc.project_root, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_viewing_angle_analysis(json_file, npz_file, facet_id, n_pos, phase, output_dir)

if __name__ == "__main__":
    interactive_analysis() 