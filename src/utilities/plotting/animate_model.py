"""
Animate Model Script

This script creates an interactive 3D animation of a shape model, allowing the user to visualize various properties over time (e.g., temperature, illumination). It allows the user to pause, select individual facets, and display their properties over time on a separate graph.

Known Issues:
1. Segmentation Fault: The script crashes with a segmentation fault the second time it is called by the model script. 

TODO: 
Option to make rotation axis and sun invisible 
"""

from stl import mesh
import time
import math
import matplotlib.pyplot as plt
import gc
import numpy as np
import datetime
import os
import pandas as pd
from matplotlib.widgets import Button
from src.utilities.locations import Locations
from src.utilities.utils import conditional_print
import h5py
from src.model.facet import Facet

def debug_print(state, *args, **kwargs):
    """Print debug messages only if animation_debug_mode is enabled."""
    if getattr(state, 'animation_debug_mode', False):
        print(*args, **kwargs)

class AnimationState:
    def __init__(self):
        self.is_paused = False
        self.current_frame = 0
        self.camera_phi = math.pi / 2
        self.camera_theta = math.pi / 2
        self.camera_radius = None
        self.fig, self.ax = None, None
        self.pause_time = None
        self.time_line = None
        self.cumulative_rotation = 0
        self.timesteps_per_day = None
        self.highlighted_cell_ids = []
        self.cell_colors = {}
        self.highlight_colors = {}
        self.highlight_mesh = None
        self.color_cycle = plt.cm.tab10.colors
        self.color_index = 0
        self.initial_camera_position = None
        self.initial_camera_focal_point = None
        self.initial_camera_up = None
        self.shape_model_name = None
        self.use_local_time = False  # Toggle between global and local time
        self.facet_normals = None
        self.sunlight_direction = None
        self.rotation_axis = None
        self.current_min = None
        self.current_max = None
        self.view_mode = False  # Toggle between raw and view-based shading (default off)
        self.dome_flux_th = None
        self.dome_bin_normals = None
        self.dome_bin_areas = None
        self.dome_bin_solid_angles = None
        self.dome_rotations = None
        self.simulation_emissivity = None
        self.facet_areas = None
        self.last_camera_angles = None  # Only print camera direction when it changes
        self.dome_radius_factor = None
        self.highlights_need_update = False

def get_next_color(state):
    color = state.color_cycle[state.color_index % len(state.color_cycle)]
    state.color_index += 1
    return color[:3]

def on_press(state):
    state.is_paused = not state.is_paused
    state.pause_time = state.current_frame / state.timesteps_per_day if state.is_paused else None
    debug_print(state, f"[DEBUG] Toggled pause to {state.is_paused}")

def move_forward(state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label):
    state.current_frame = (state.current_frame + 1) % state.timesteps_per_day
    state.cumulative_rotation = (state.current_frame / state.timesteps_per_day) * 2 * math.pi
    update(None, None, state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label)
    plotter.render()

def move_backward(state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label):
    state.current_frame = (state.current_frame - 1) % state.timesteps_per_day
    state.cumulative_rotation = (state.current_frame / state.timesteps_per_day) * 2 * math.pi
    update(None, None, state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label)
    plotter.render()

def reset_camera(state, plotter):
    if state.initial_camera_position is not None:
        plotter.camera.position = state.initial_camera_position
        plotter.camera.focal_point = state.initial_camera_focal_point
        plotter.camera.up = state.initial_camera_up
        plotter.render()

def round_up_to_nearest(x, base):
    return base * math.ceil(x / base)

def round_down_to_nearest(x, base):
    return base * math.floor(x / base)

def rotation_matrix(axis, theta):
    axis = [a / math.sqrt(sum(a**2 for a in axis)) for a in axis]
    a = math.cos(theta / 2.0)
    b, c, d = [-axis[i] * math.sin(theta / 2.0) for i in range(3)]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return [[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]

def clear_selections(state, plotter):
    state.highlighted_cell_ids.clear()
    state.highlight_colors.clear()
    state.color_index = 0  # Reset color index
    # Mark highlights as needing update
    state.highlights_need_update = True
    
    # Remove old single highlight mesh if it exists
    if hasattr(state, 'highlight_mesh') and state.highlight_mesh is not None:
        plotter.remove_actor(state.highlight_mesh)
        state.highlight_mesh = None
    
    # Remove new multi-mesh highlighting system
    if hasattr(state, 'highlight_meshes') and state.highlight_meshes:
        for mesh_actor in state.highlight_meshes:
            plotter.remove_actor(mesh_actor)
        state.highlight_meshes.clear()
    
    if state.fig is not None:
        for line in state.cell_colors.values():
            line.remove()
        state.cell_colors.clear()
        state.ax.legend()  # Update the legend
        state.fig.canvas.draw()
        state.fig.canvas.flush_events()
    plotter.render()

def plot_picked_cell_over_time(state, cell_id, plotter, pv_mesh, plotted_variable_array, axis_label):
    if cell_id in state.highlighted_cell_ids:
        # Remove the cell if it's already highlighted
        state.highlighted_cell_ids.remove(cell_id)
        if cell_id in state.cell_colors:
            line = state.cell_colors.pop(cell_id)
            line.remove()
        if cell_id in state.highlight_colors:
            color = state.highlight_colors.pop(cell_id)
            state.color_index = (state.color_index - 1) % len(state.color_cycle)
        # Mark highlights as needing update
        state.highlights_need_update = True
    else:
        # Check selection limit
        if len(state.highlighted_cell_ids) >= 5:
            print(f"Maximum 5 facets can be selected at once for performance. Clear selections (press 'c') to select new facets.")
            return
            
        # Add the cell if it's not highlighted
        state.highlighted_cell_ids.append(cell_id)
        color = get_next_color(state)
        state.highlight_colors[cell_id] = color
        # Mark highlights as needing update
        state.highlights_need_update = True

        if state.fig is None or state.ax is None:
            plt.ion()
            state.fig, state.ax = plt.subplots()

            def on_key(event):
                if event.key == 'd':  # Download functionality
                    try:
                        # Get the output directory using Locations class
                        locations = Locations()
                        
                        # Create base directory for user saved data
                        user_data_dir = os.path.join(locations.outputs, 'user_saved_data')
                        os.makedirs(user_data_dir, exist_ok=True)
                        
                        # Get or create run-specific directory using shape model name and timestamp
                        if not hasattr(state, 'run_folder'):
                            model_name = os.path.splitext(os.path.basename(state.shape_model_name))[0]
                            timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')
                            state.run_folder = f"{model_name}_{timestamp}"
                        
                        run_dir = os.path.join(user_data_dir, state.run_folder)
                        os.makedirs(run_dir, exist_ok=True)
                            
                        # Prepare data dictionary
                        data = {}
                        n_timesteps = plotted_variable_array.shape[1]
                        # Use local time (0-24 hours) instead of fraction of day
                        time_steps = np.linspace(0, 24, n_timesteps)
                        
                        # Check if we need to calculate view-dependent temperatures
                        if state.view_mode and state.dome_flux_th is not None:
                            print("Calculating view-dependent temperatures for all timesteps...")
                            # Calculate view-dependent temperatures for all timesteps
                            view_dependent_temps = calculate_view_dependent_temperatures_all_timesteps(
                                state, plotter, rotation_matrix, state.rotation_axis)
                            
                            if view_dependent_temps is not None:
                                for facet_id in state.highlighted_cell_ids:
                                    # Convert to local time (shift data so noon is at 12:00)
                                    local_time_data = convert_to_local_time(
                                        state, view_dependent_temps[facet_id, :],
                                        state.facet_normals[facet_id],
                                        state.sunlight_direction,
                                        state.rotation_axis,
                                        facet_id)
                                    data[f'Facet_{facet_id}_view_dependent'] = local_time_data
                                property_suffix = "_view_dependent"
                            else:
                                # Fallback to regular data
                                for facet_id in state.highlighted_cell_ids:
                                    # Convert to local time (shift data so noon is at 12:00)
                                    local_time_data = convert_to_local_time(
                                        state, plotted_variable_array[facet_id, :],
                                        state.facet_normals[facet_id],
                                        state.sunlight_direction,
                                        state.rotation_axis,
                                        facet_id)
                                    data[f'Facet_{facet_id}'] = local_time_data
                                property_suffix = ""
                        else:
                            # Save regular temperature data converted to local time
                            for facet_id in state.highlighted_cell_ids:
                                # Convert to local time (shift data so noon is at 12:00)
                                local_time_data = convert_to_local_time(
                                    state, plotted_variable_array[facet_id, :],
                                    state.facet_normals[facet_id],
                                    state.sunlight_direction,
                                    state.rotation_axis,
                                    facet_id)
                                data[f'Facet_{facet_id}'] = local_time_data
                            property_suffix = ""
                        
                        # Create DataFrame
                        df = pd.DataFrame(data, index=time_steps)
                        df.index.name = 'Local Time (hours)'
                        
                        # Create a valid filename from the axis label
                        property_name = axis_label.lower().replace(' ', '_')
                        timestamp = datetime.datetime.now().strftime("%H%M%S")
                        filename = os.path.join(run_dir, f'{property_name}{property_suffix}_vs_timestep_{timestamp}.csv')
                        df.to_csv(filename)
                        conditional_print(False, f"Data saved to {filename}")
                        
                    except Exception as e:
                        print(f"Error saving data: {e}")
                elif event.key == 't':  # Add time toggle functionality
                    state.use_local_time = not state.use_local_time
                    # Update all existing lines
                    for facet_id, line in state.cell_colors.items():
                        values = plotted_variable_array[facet_id, :]
                        if state.use_local_time:
                            values = convert_to_local_time(state, values, 
                                                         state.facet_normals[facet_id],
                                                         state.sunlight_direction, 
                                                         state.rotation_axis,
                                                         facet_id)
                            time_steps = np.linspace(0, 24, len(values))  # 24-hour time
                        else:
                            time_steps = np.linspace(0, 1, len(values))
                        line.set_xdata(time_steps)
                        line.set_ydata(values)
                    
                    state.ax.set_xlabel("Local Time (hours)" if state.use_local_time else "Fractional angle of rotation")
                    if state.use_local_time:
                        state.ax.set_xlim(0, 24)
                        state.ax.set_xticks(np.linspace(0, 24, 13))
                    else:
                        state.ax.set_xlim(0, 1)
                        state.ax.set_xticks(np.linspace(0, 1, 5))
                    state.fig.canvas.draw()

                state.fig.canvas.flush_events()

            state.fig.canvas.mpl_connect('key_press_event', on_key)
            state.ax.set_xlabel("Fractional angle of rotation")
            state.ax.set_ylabel(axis_label)
            state.ax.set_title(f"{axis_label} Over Time for Selected Facets\nPress 'd' to save data, 't' to toggle local/global time")

        values_over_time = plotted_variable_array[cell_id, :]
        if state.use_local_time:
            values_over_time = convert_to_local_time(state, values_over_time, 
                                                   state.facet_normals[cell_id],
                                                   state.sunlight_direction, 
                                                   state.rotation_axis,
                                                   cell_id)
            time_steps = np.linspace(0, 24, len(values_over_time))
        else:
            time_steps = np.linspace(0, 1, len(values_over_time))
        line, = state.ax.plot(time_steps, values_over_time, label=f'Cell {cell_id}', color=color)
        state.cell_colors[cell_id] = line

    # Ensure the graph and highlight mesh are updated
    update_highlight_mesh(state, plotter, pv_mesh)
    if state.fig is not None:
        state.ax.legend()
        state.fig.canvas.draw()
        state.fig.canvas.flush_events()

def calculate_facet_lat_lon(vertices, cell_id):
    """
    Calculate latitude and longitude of a facet from its centroid position.
    Uses original unrotated vertex positions to get consistent lat/lon regardless of rotation.
    
    Args:
        vertices: Original unrotated vertex array
        cell_id: ID of the cell/facet
        
    Returns:
        tuple: (latitude, longitude) in degrees
    """
    # Each facet has 3 vertices at indices 3*cell_id, 3*cell_id+1, 3*cell_id+2
    v1 = vertices[3 * cell_id]
    v2 = vertices[3 * cell_id + 1]
    v3 = vertices[3 * cell_id + 2]
    
    # Calculate the centroid of the triangle
    centroid = (v1 + v2 + v3) / 3.0
    
    # Convert to spherical coordinates
    x, y, z = centroid
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Latitude: angle from equatorial plane (-90 to +90 degrees)
    latitude = np.degrees(np.arcsin(z / r)) if r > 0 else 0.0
    
    # Longitude: angle in equatorial plane (0 to 360 degrees)
    longitude = np.degrees(np.arctan2(y, x))
    # Convert from -180 to +180 range to 0 to 360 range
    if longitude < 0:
        longitude += 360
    
    return latitude, longitude

def on_pick(state, picked_mesh, plotter, pv_mesh, plotted_variable_array, axis_label):
    if picked_mesh is not None:
        cell_id = picked_mesh['vtkOriginalCellIds'][0]
        
        # Calculate and print latitude/longitude using original unrotated vertices
        latitude, longitude = calculate_facet_lat_lon(state.original_vertices, cell_id)
        print(f"\n=== Facet {cell_id} ===")
        print(f"Latitude:  {latitude:7.2f}°")
        print(f"Longitude: {longitude:7.2f}°")
        
        plot_picked_cell_over_time(state, cell_id, plotter, pv_mesh, plotted_variable_array, axis_label)

def convert_to_local_time(state, global_time_data, facet_normal, sunlight_direction, rotation_axis, facet_id):
    """
    Converts global time data to local time by finding when the facet normal is most aligned with the sun.
    
    Args:
        global_time_data: Array of data points over one rotation
        facet_normal: Normal vector of the facet
        sunlight_direction: Direction vector of sunlight
        rotation_axis: Rotation axis vector
        facet_id: ID of the facet being processed
    
    Returns:
        Array shifted to local time
    """
    timesteps = len(global_time_data)
    max_alignment = -float('inf')
    noon_frame = 0
    
    # For each timestep, calculate the alignment between rotated normal and sun
    for t in range(timesteps):
        angle = (2 * np.pi * t) / timesteps
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Create rotation matrix
        rotation_matrix = np.array([
            [cos_angle + rotation_axis[0]**2*(1-cos_angle), 
             rotation_axis[0]*rotation_axis[1]*(1-cos_angle) - rotation_axis[2]*sin_angle,
             rotation_axis[0]*rotation_axis[2]*(1-cos_angle) + rotation_axis[1]*sin_angle],
            [rotation_axis[1]*rotation_axis[0]*(1-cos_angle) + rotation_axis[2]*sin_angle,
             cos_angle + rotation_axis[1]**2*(1-cos_angle),
             rotation_axis[1]*rotation_axis[2]*(1-cos_angle) - rotation_axis[0]*sin_angle],
            [rotation_axis[2]*rotation_axis[0]*(1-cos_angle) - rotation_axis[1]*sin_angle,
             rotation_axis[2]*rotation_axis[1]*(1-cos_angle) + rotation_axis[0]*sin_angle,
             cos_angle + rotation_axis[2]**2*(1-cos_angle)]
        ])
        
        # Rotate the normal
        rotated_normal = np.dot(rotation_matrix, facet_normal)
        
        # Calculate alignment with sun direction
        alignment = np.dot(rotated_normal, sunlight_direction)
        
        if alignment > max_alignment:
            max_alignment = alignment
            noon_frame = t
    
    # Calculate shift needed to center on maximum alignment
    shift = (timesteps // 2) - noon_frame  # Shift to put noon at middle of day
    
    print(f"Facet {facet_id} shifted by {shift} timesteps ({(shift/timesteps * 360):.1f} degrees)")
    
    return np.roll(global_time_data, shift)

def calculate_view_dependent_temperatures_all_timesteps(state, plotter, rot_mat_func, rotation_axis):
    """Calculate view-dependent temperatures for all timesteps given current viewing direction."""
    if not state.view_mode or state.dome_flux_th is None:
        return None
        
    # Get current camera viewing direction
    cam_pos = np.array(plotter.camera.position)
    focal = np.array(plotter.camera.focal_point)
    view_dir_world = (focal - cam_pos)
    view_dir_world /= np.linalg.norm(view_dir_world)
    
    n_facets = len(state.facet_normals)
    n_timesteps = state.dome_flux_th.shape[2]
    view_dependent_temps = np.zeros((n_facets, n_timesteps))
    
    # Constants for temperature calculation
    scale = state.dome_radius_factor ** 2
    sigma = 5.670374419e-8
    epsilon = state.simulation_emissivity
    
    for timestep in range(n_timesteps):
        # Calculate rotation matrix for this timestep
        rotation_angle = (timestep / state.timesteps_per_day) * 2 * math.pi
        rot_mat = np.array(rot_mat_func(rotation_axis, rotation_angle))
        
        # Transform view direction to body coordinates for this timestep
        view_dir_body = rot_mat.T.dot(view_dir_world)
        
        # Find best dome bins for each facet
        cosines = np.dot(view_dir_body, state.dome_bin_normals.T).reshape(1, -1)
        cosines = np.broadcast_to(cosines, (n_facets, len(state.dome_bin_normals)))
        cosines = np.clip(cosines, 0, None)
        best_bins = np.argmax(cosines, axis=1)
        selected_cosines = cosines[np.arange(len(best_bins)), best_bins]
        
        # Compute facet horizon test
        normals_t = np.dot(state.facet_normals, rot_mat.T)
        facet_horizons = np.dot(normals_t, view_dir_world)
        
        # Extract flux and area data for selected bins
        F_values = state.dome_flux_th[np.arange(len(best_bins)), best_bins, timestep]
        area_values = state.dome_bin_areas[np.arange(len(best_bins)), best_bins]
        
        # Compute temperatures
        T_raw = np.zeros(n_facets)
        valid_areas = area_values > 0
        T_raw[valid_areas] = ((F_values[valid_areas] / area_values[valid_areas] * scale) / (epsilon * sigma)) ** 0.25
        
        # Apply horizon and dome visibility masks (same logic as real-time rendering)
        horizon_tolerance = 1e-6
        mask_dome = selected_cosines > horizon_tolerance  # Dome patch visible
        mask_facet = facet_horizons < -horizon_tolerance  # Facet front-facing
        mask_combined = mask_dome & mask_facet
        
        # Set final temperatures: 0K if not visible, raw temperature if visible
        T_final = np.where(mask_combined, T_raw, 0.0)
        
        view_dependent_temps[:, timestep] = T_final
    
    return view_dependent_temps

def convert_to_view_time(global_time_data, facet_normal, rotation_axis, view_dir):
    """Convert global time data to view-based apparent data over one rotation."""
    T = len(global_time_data)
    result = np.zeros_like(global_time_data)
    for t in range(T):
        angle = (2 * math.pi * t) / T
        rot = np.array(rotation_matrix(rotation_axis, angle))
        n_rot = np.dot(rot, facet_normal)
        cos_val = np.dot(n_rot, view_dir)
        result[t] = global_time_data[t] * max(cos_val, 0)
    return result

def animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction, 
                  timesteps_per_day, solar_distance_au, rotation_period_hr, emissivity, plot_title, axis_label, animation_frames, 
                  save_animation, save_animation_name, background_colour, dome_radius_factor=1.0, colour_map='coolwarm', apply_kernel_based_roughness=False, pre_selected_facets=[1220, 845], animation_debug_mode=False):
    """
    Animate a 3D model with temperature or other variable data.
    
    Args:
        path_to_shape_model_file (str): Path to the STL file
        plotted_variable_array (np.ndarray): Array of values to plot
        rotation_axis (np.ndarray): Axis of rotation
        sunlight_direction (np.ndarray): Direction of sunlight
        timesteps_per_day (int): Number of timesteps per day
        solar_distance_au (float): Solar distance in AU
        rotation_period_hr (float): Rotation period in hours
        emissivity (float): Emissivity of the surface
        plot_title (str): Title of the plot
        axis_label (str): Label for the colorbar
        animation_frames (int): Number of frames in the animation
        save_animation (bool): Whether to save the animation
        save_animation_name (str): Name of the saved animation file
        background_colour (str): Background color of the plot
        colour_map (str): Name of the colormap to use
        pre_selected_facets (list): List of facet IDs to pre-select
    """
    # Lazy import of pyvista and vtk - only import when animation is actually needed
    # This prevents slow startup when animation features aren't being used
    import pyvista as pv
    import vtk

    start_time = time.time()
    # Radiance mode: if HDF5 subfacet data exists, load it instead of STL
    base, _ = os.path.splitext(path_to_shape_model_file)
    h5_path = base + '_subfacets.h5'
    if os.path.exists(h5_path):
        # Load subfacet mesh and temperatures
        with h5py.File(h5_path, 'r') as hf:
            pts = hf['points'][:]
            faces = hf['faces'][:]
            temps = hf['temps'][:]
        pv_mesh = pv.PolyData(pts, faces)
        plotted_variable_array = temps
    else:
        # Default parent-facet mode
        pass
    try:
        shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file)
    except Exception as e:
        print(f"Failed to load shape model: {e}")
        return
    if shape_mesh.vectors.shape[0] != plotted_variable_array.shape[0]:
        print("The plotted variable array must have the same number of rows as the number of cells in the shape model.")
        return
    vertices = shape_mesh.points.reshape(-1, 3)
    faces = [[3, 3*i, 3*i+1, 3*i+2] for i in range(shape_mesh.vectors.shape[0])]
    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh.cell_data[axis_label] = plotted_variable_array[:, 0]

    # Store facet normals when loading the mesh
    facet_normals = np.zeros((shape_mesh.vectors.shape[0], 3))
    for i in range(shape_mesh.vectors.shape[0]):
        v1, v2, v3 = shape_mesh.vectors[i]
        normal = np.cross(v2 - v1, v3 - v1)
        facet_normals[i] = normal / np.linalg.norm(normal)
    
    # Compute parent facet areas for scaling dome bins
    # shape_mesh.vectors has shape (n_facets,3,3): extract triangle vertices
    tri = shape_mesh.vectors  # shape (nF,3,3)
    # Compute triangle areas: ||(v2-v1)x(v3-v1)||/2
    facet_areas = np.linalg.norm(np.cross(tri[:,1] - tri[:,0], tri[:,2] - tri[:,0]), axis=1) / 2
    # Initialize state before any debug prints
    state = AnimationState()
    state.animation_debug_mode = animation_debug_mode
    debug_print(state, f"[DEBUG] Computed facet_areas shape: {facet_areas.shape}")
    state.dome_radius_factor = dome_radius_factor  # store scale factor for exitance calculation
    # Store facet areas on state for dome area scaling
    state.facet_areas = facet_areas
    state.timesteps_per_day = timesteps_per_day
    state.shape_model_name = path_to_shape_model_file
    state.facet_normals = facet_normals
    state.original_vertices = vertices  # Store original unrotated vertices for lat/lon calculation
    state.sunlight_direction = sunlight_direction
    state.rotation_axis = rotation_axis
    if emissivity is None:
        raise ValueError("emissivity must be specified - it is a critical physical parameter")
    state.simulation_emissivity = emissivity
    debug_print(state, f"[DEBUG] Using emissivity: {state.simulation_emissivity}")

    # Load precomputed dome thermal flux arrays only if roughness is enabled
    if apply_kernel_based_roughness:
        # Look for dome flux data in the most recent animation output
        from src.utilities.locations import Locations
        loc = Locations()
        remote_outputs_dir = loc.remote_outputs
        
        # Find the most recent animation output directory
        animation_dirs = [d for d in os.listdir(remote_outputs_dir) if d.startswith('animation_outputs_')]
        if animation_dirs:
            latest_dir = max(animation_dirs)
            dome_flux_file = [f for f in os.listdir(os.path.join(remote_outputs_dir, latest_dir)) 
                             if f.startswith('combined_animation_data_rough_')][0]
            dome_flux_path = os.path.join(remote_outputs_dir, latest_dir, dome_flux_file)
        else:
            dome_flux_path = os.path.join('data', 'output', 'dome_fluxes.h5')  # Fallback to old path
            
        if os.path.exists(dome_flux_path):
            with h5py.File(dome_flux_path, 'r') as dfh:
                # Try to load from dome_fluxes group first (new format), then root level (old format)
                if 'dome_fluxes' in dfh:
                    dome_group = dfh['dome_fluxes']
                    state.dome_flux_th = dome_group['dome_flux_th'][:]
                    state.dome_bin_normals = dome_group['dome_normals'][:]
                    state.dome_bin_areas = dome_group['dome_bin_areas'][:]
                    state.dome_bin_solid_angles = dome_group['dome_bin_solid_angles'][:]
                else:
                    # Fallback to old root-level format
                    state.dome_flux_th = dfh['dome_flux_th'][:]
                    state.dome_bin_normals = dfh['dome_normals'][:]
                    state.dome_bin_areas = dfh['dome_bin_areas'][:]
                    state.dome_bin_solid_angles = dfh['dome_bin_solid_angles'][:]
                debug_print(state, f"[DEBUG] Loaded dome_flux_th with shape {state.dome_flux_th.shape}, dome_bin_normals shape {state.dome_bin_normals.shape}, dome_bin_areas shape {state.dome_bin_areas.shape}")
                # Compute world->local rotation matrices for each facet
                up = np.array([0.0, 0.0, 1.0])
                nF = facet_normals.shape[0]
                dome_rots = np.zeros((nF, 3, 3))
                for i in range(nF):
                    nvec = facet_normals[i]
                    axis = np.cross(up, nvec)
                    if np.linalg.norm(axis) < 1e-8:
                        R_l2w = np.eye(3) if np.dot(up, nvec) > 0 else \
                                 np.array(rotation_matrix(np.array([1.0, 0.0, 0.0]), math.pi))
                    else:
                        axis_norm = axis / np.linalg.norm(axis)
                        angle = math.acos(np.clip(np.dot(up, nvec), -1.0, 1.0))
                        R_l2w = np.array(rotation_matrix(axis_norm, angle))
                    dome_rots[i] = R_l2w.T
                state.dome_rotations = dome_rots
                debug_print(state, f"[DEBUG] Initialized dome_rotations with shape {state.dome_rotations.shape}")
        else:
            debug_print(state, "[DEBUG] No dome_fluxes.h5 found, setting all dome-related arrays to None")
            state.dome_flux_th = None
            state.dome_bin_normals = None
            state.dome_bin_solid_angles = None
            state.dome_rotations = None
            state.dome_bin_areas = None
    else:
        # Roughness disabled: skip loading any dome-related data
        debug_print(state, "[DEBUG] Roughness disabled, skipping dome data loading")
        state.dome_flux_th = None
        state.dome_bin_normals = None
        state.dome_bin_solid_angles = None
        state.dome_rotations = None
        state.dome_bin_areas = None

    text_color = 'white' if background_colour == 'black' else 'black' 
    bar_color = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

    bounding_box = pv_mesh.bounds
    max_dimension = max(bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2], bounding_box[5] - bounding_box[4])
    state.camera_radius = max_dimension * 5

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    plotter.enable_lightkit()  # Enable dynamic lighting for shading
    plotter.add_key_event('space', lambda: (setattr(state, 'is_paused', not state.is_paused), setattr(state, 'pause_time', (state.current_frame / state.timesteps_per_day) if state.is_paused else None), debug_print(state, f"[DEBUG] Toggled pause to {state.is_paused}")))
    plotter.add_key_event('Left', lambda: move_backward(state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label))
    plotter.add_key_event('Right', lambda: move_forward(state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label))
    plotter.add_key_event('c', lambda: clear_selections(state, plotter))
    plotter.add_key_event('r', lambda: reset_camera(state, plotter))  # Add the new key event for camera reset
    # Unified view-mode toggle: update both 3D mesh shading and 2D diurnal plot
    def _toggle_view_event():
        state.view_mode = not state.view_mode
        debug_print(state, f"[DEBUG] Toggled view_mode to {state.view_mode}")
        # Save current mesh geometry to preserve orientation relative to the camera
        saved_points = pv_mesh.points.copy()
        # Update 3D mesh shading for new view mode (this may overwrite points)
        update(None, None, state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label)
        # Restore geometry so the mesh doesn't rotate under the camera
        pv_mesh.points = saved_points
        plotter.render()
        # Also update 2D diurnal plot if open
        if state.fig and hasattr(state, 'ax'):
            debug_print(state, "[DEBUG] Updating diurnal plot from _toggle_view_event")
            # Compute time-series lines depending on view_mode
            if state.view_mode and state.dome_flux_th is not None:
                # Dome-based temperatures across full rotation
                # Compute fixed view direction in world coords
                cam_pos = np.array(plotter.camera.position)
                focal = np.array(plotter.camera.focal_point)
                view_dir_world = focal - cam_pos
                view_dir_world /= np.linalg.norm(view_dir_world)
                # Debug: print camera direction spherical angles only when changed
                theta = math.acos(np.clip(view_dir_world[2], -1.0, 1.0))
                phi = math.atan2(view_dir_world[1], view_dir_world[0])
                if state.last_camera_angles is None or abs(theta - state.last_camera_angles[0]) > 1e-3 or abs(phi - state.last_camera_angles[1]) > 1e-3:
                    debug_print(state, f"[DEBUG] Camera direction angles: theta={theta:.3f}, phi={phi:.3f}")
                    state.last_camera_angles = (theta, phi)
                # Compute global rotation for facet normals
                rot_mat = np.array(rotation_matrix(state.rotation_axis, state.cumulative_rotation))
                # Transform view direction to body coordinates 
                view_dir_body = rot_mat.T.dot(view_dir_world)
                # Compute rotated facet normals for horizon test
                normals_t = np.dot(state.facet_normals, rot_mat.T)
                # Compute facet horizon mask: rotated facet normal dot view direction
                facet_horizons = np.dot(normals_t, view_dir_world)
                # Use body-frame view direction with world-coordinate dome normals
                cosines = np.dot(view_dir_body, state.dome_bin_normals.T).reshape(1, -1)
                cosines = np.broadcast_to(cosines, (len(state.facet_normals), len(state.dome_bin_normals)))
                cosines = np.clip(cosines, 0, None)
                best_bins = np.argmax(cosines, axis=1)
                # Prepare time axis
                sigma = 5.670374419e-8
                epsilon = state.simulation_emissivity
                n_t = state.dome_flux_th.shape[2]
                x = np.linspace(0, 1, n_t)
                # Update each selected facet's diurnal line with dynamic bin selection
                for facet_id, line in state.cell_colors.items():
                    T_series = np.zeros(n_t)
                    for j in range(n_t):
                        # Body rotation at timestep j
                        angle_j = 2 * math.pi * j / state.timesteps_per_day
                        rot_j = np.array(rotation_matrix(state.rotation_axis, angle_j))
                        # Rotate view direction into mesh frame (world->body)
                        view_mesh = rot_j.T.dot(view_dir_world)
                        # FIXED: Use view direction in body coordinates directly with dome normals (both in world coordinates)
                        coss = np.clip(np.dot(view_mesh, state.dome_bin_normals.T), 0, None)
                        b_j = np.argmax(coss)
                        F_j = state.dome_flux_th[facet_id, b_j, j]
                        area_j = state.dome_bin_areas[facet_id, b_j]
                        scale = state.dome_radius_factor ** 2
                        # Check if patch is visible and facet facing camera
                        facet_norm_j = state.facet_normals[facet_id].dot(rot_j.T)
                        horizon_val = np.dot(facet_norm_j, view_dir_world)
                        # Add small tolerance to horizon tests
                        horizon_tolerance = 1e-6
                        if area_j > 0 and coss[b_j] > horizon_tolerance and horizon_val < -horizon_tolerance:
                            T_series[j] = ((F_j / area_j * scale) / (epsilon * sigma)) ** 0.25
                        else:
                            T_series[j] = 0.0
                    line.set_xdata(x)
                    line.set_ydata(T_series)
            else:
                # Parent facet temperatures
                n_t = plotted_variable_array.shape[1]
                x = np.linspace(0, 1, n_t)
                for facet_id, line in state.cell_colors.items():
                    y = plotted_variable_array[facet_id, :]
                    line.set_xdata(x)
                    line.set_ydata(y)
            # Refresh legend to show bin IDs
            state.ax.legend()
            # Common update for axes
            state.ax.relim()
            state.ax.autoscale_view()
            state.ax.set_ylabel("Viewed Temperature (K)" if state.view_mode else axis_label)
            state.fig.canvas.draw()
            state.fig.canvas.flush_events()
    plotter.add_key_event('v', _toggle_view_event)

    cylinder = pv.Cylinder(center=(0, 0, 0), direction=rotation_axis, height=max_dimension, radius=max_dimension/200)
    plotter.add_mesh(cylinder, color='green')

    sunlight_start = [sunlight_direction[i] * max_dimension for i in range(3)]
    sunlight_arrow = pv.Arrow(start=sunlight_start, direction=[-d for d in sunlight_direction], scale=max_dimension * 0.3)
    plotter.add_mesh(sunlight_arrow, color='yellow')

    plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False, lighting=True, smooth_shading=True)

    plotter.enable_element_picking(callback=lambda picked_mesh: on_pick(state, picked_mesh, plotter, pv_mesh, plotted_variable_array, axis_label), mode='cell', show=False, show_message=False)

    text_color_rgb = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

    state.initial_camera_position = plotter.camera.position
    state.initial_camera_focal_point = plotter.camera.focal_point
    state.initial_camera_up = plotter.camera.up

    plotter.scalar_bar.GetLabelTextProperty().SetColor(bar_color)
    plotter.scalar_bar.SetPosition(0.2, 0.05)
    plotter.scalar_bar.GetLabelTextProperty().SetJustificationToCentered()
    plotter.scalar_bar.SetTitle(axis_label)
    plotter.scalar_bar.GetTitleTextProperty().SetColor(text_color_rgb)

    min_val = np.min(plotted_variable_array)
    max_val = np.max(plotted_variable_array)

    num_labels = 5

    def round_to_nice_number(x):
        if x == 0:
            return 0
        exponent = math.floor(math.log10(abs(x)))
        coeff = x / (10**exponent)
        if coeff < 1.5:
            nice_coeff = 1
        elif coeff < 3:
            nice_coeff = 2
        elif coeff < 7:
            nice_coeff = 5
        else:
            nice_coeff = 10
        return nice_coeff * (10**exponent)

    nice_min_val = round_to_nice_number(min_val)
    nice_max_val = round_to_nice_number(max_val)

    labels = np.linspace(nice_min_val, nice_max_val, num_labels)

    vtk_labels = vtk.vtkDoubleArray()
    vtk_labels.SetNumberOfValues(len(labels))
    for i, label in enumerate(labels):
        vtk_labels.SetValue(i, label)

    plotter.scalar_bar.SetCustomLabels(vtk_labels)
    plotter.scalar_bar.SetUseCustomLabels(True)

    # After creating the plotter and before adding text elements
    # Get the initial data range
    data_min = np.min(plotted_variable_array)
    data_max = np.max(plotted_variable_array)
    range_padding = (data_max - data_min) * 0.1  # Add 10% padding
    
    # Store the current range in the state
    state.current_min = data_min
    state.current_max = data_max
    
    def update_min(value):
        state.current_min = value
        plotter.update_scalar_bar_range((state.current_min, state.current_max))
        pv_mesh.set_active_scalars(axis_label)
        plotter.render()
    
    def update_max(value):
        state.current_max = value
        plotter.update_scalar_bar_range((state.current_min, state.current_max))
        pv_mesh.set_active_scalars(axis_label)
        plotter.render()
    
    # Add sliders for min and max values
    plotter.add_slider_widget(
        callback=update_min,
        rng=[data_min - range_padding, data_max],
        value=data_min,
        title=f"Min {axis_label}",
        pointa=(0.025, 0.1),
        pointb=(0.225, 0.1),
        style='modern'
    )
    
    plotter.add_slider_widget(
        callback=update_max,
        rng=[data_min, data_max + range_padding],
        value=data_max,
        title=f"Max {axis_label}",
        pointa=(0.025, 0.15),
        pointb=(0.225, 0.15),
        style='modern'
    )

    plotter.add_text(plot_title, position='upper_edge', font_size=12, color=text_color)
    plotter.add_text("'Spacebar' - Pause/play\n'Right click' - Select facet\n'C' - Clear selections\n'R' - Reset camera\n'V' - Toggle view mode\n(rough models only)", position='upper_left', font_size=10, color=text_color)

    info_text = f"Solar Distance: {solar_distance_au} AU\nRotation Period: {rotation_period_hr} hours"
    plotter.add_text(info_text, position='upper_right', font_size=10, color=text_color)

    plotter.background_color = background_colour

    def update_callback(caller, event):
        # Always update display (advances time if not paused, recomputes shading)
        update(None, None, state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label)

    if save_animation:
        plotter.open_gif(save_animation_name)
        for _ in range(animation_frames):
            update_callback(None, None)
            plotter.write_frame()
        plotter.close()
    else:
        end_time = time.time()
        print(f'Animation setup took {end_time - start_time:.2f} seconds.')
        
        # Print camera position after any interactive camera movement
        plotter.iren.add_observer('EndInteractionEvent', lambda caller, event: debug_print(state, f"[DEBUG] Camera pos: {plotter.camera.position}, focal: {plotter.camera.focal_point}, up: {plotter.camera.up}"))
        # Update 2D diurnal plot on camera move when in view_mode
        def _camera_moved_event(caller, event):
            if state.view_mode and state.fig and hasattr(state, 'ax'):
                debug_print(state, "[DEBUG] Camera moved event: updating diurnal plot")
                sigma = 5.670374419e-8
                epsilon = state.simulation_emissivity
                # Compute fixed view direction in world coords
                cam_pos = np.array(plotter.camera.position)
                focal = np.array(plotter.camera.focal_point)
                view_dir_world = focal - cam_pos
                view_dir_world /= np.linalg.norm(view_dir_world)
                # Debug: print camera direction spherical angles only when changed
                theta = math.acos(np.clip(view_dir_world[2], -1.0, 1.0))
                phi = math.atan2(view_dir_world[1], view_dir_world[0])
                if state.last_camera_angles is None or abs(theta - state.last_camera_angles[0]) > 1e-3 or abs(phi - state.last_camera_angles[1]) > 1e-3:
                    debug_print(state, f"[DEBUG] Camera direction angles: theta={theta:.3f}, phi={phi:.3f}")
                    state.last_camera_angles = (theta, phi)
                # Compute global rotation for facet normals
                rot_mat = np.array(rotation_matrix(state.rotation_axis, state.cumulative_rotation))
                # Transform view direction to body coordinates 
                view_dir_body = rot_mat.T.dot(view_dir_world)
                # Compute rotated facet normals for horizon test
                normals_t = np.dot(state.facet_normals, rot_mat.T)
                # Compute facet horizon mask: rotated facet normal dot view direction
                facet_horizons = np.dot(normals_t, view_dir_world)
                # Use body-frame view direction with world-coordinate dome normals
                cosines = np.dot(view_dir_body, state.dome_bin_normals.T).reshape(1, -1)
                cosines = np.broadcast_to(cosines, (len(state.facet_normals), len(state.dome_bin_normals)))
                cosines = np.clip(cosines, 0, None)
                best_bins = np.argmax(cosines, axis=1)
                # Prepare time axis
                n_t = state.dome_flux_th.shape[2]
                x = np.linspace(0, 1, n_t)
                # Update each selected facet's diurnal line with dynamic bin selection
                for facet_id, line in state.cell_colors.items():
                    T_series = np.zeros(n_t)
                    for j in range(n_t):
                        # Body rotation at timestep j
                        angle_j = 2 * math.pi * j / state.timesteps_per_day
                        rot_j = np.array(rotation_matrix(state.rotation_axis, angle_j))
                        # Rotate view direction into mesh frame (world->body)
                        view_mesh = rot_j.T.dot(view_dir_world)
                        # FIXED: Use view direction in body coordinates directly with dome normals (both in world coordinates)
                        coss = np.clip(np.dot(view_mesh, state.dome_bin_normals.T), 0, None)
                        b_j = np.argmax(coss)
                        F_j = state.dome_flux_th[facet_id, b_j, j]
                        area_j = state.dome_bin_areas[facet_id, b_j]
                        scale = state.dome_radius_factor ** 2
                        # Check if patch is visible and facet facing camera
                        facet_norm_j = state.facet_normals[facet_id].dot(rot_j.T)
                        horizon_val = np.dot(facet_norm_j, view_dir_world)
                        # Add small tolerance to horizon tests
                        horizon_tolerance = 1e-6
                        if area_j > 0 and coss[b_j] > horizon_tolerance and horizon_val < -horizon_tolerance:
                            T_series[j] = ((F_j / area_j * scale) / (epsilon * sigma)) ** 0.25
                        else:
                            T_series[j] = 0.0
                    line.set_xdata(x)
                    line.set_ydata(T_series)
                # Refresh plot
                state.ax.relim()
                state.ax.autoscale_view()
                state.fig.canvas.draw()
                state.fig.canvas.flush_events()

                # Diagnostic consistency check (camera moved): ensure PyVista shading matches plotted line
                for facet_id, line in state.cell_colors.items():
                    shading_val = pv_mesh.cell_data[axis_label][facet_id]
                    ydata = line.get_ydata()
                    if state.current_frame < len(ydata):
                        plotted_val = ydata[state.current_frame]
                        
                        # Print comprehensive debug info for selected facets
                        debug_print(state, f"\n[FACET DEBUG] Facet {facet_id} at Frame {state.current_frame}:")
                        debug_print(state, f"  3D Shading: {shading_val:.2f}K, 2D Plot: {plotted_val:.2f}K, Diff: {abs(shading_val - plotted_val):.2f}K")
                        
                        # Camera info
                        cam_pos = np.array(plotter.camera.position)
                        focal = np.array(plotter.camera.focal_point)
                        view_dir_world = (focal - cam_pos) / np.linalg.norm(focal - cam_pos)
                        debug_print(state, f"  View Direction: [{view_dir_world[0]:.3f}, {view_dir_world[1]:.3f}, {view_dir_world[2]:.3f}]")
                        
                        # Facet info
                        current_rot_mat = np.array(rotation_matrix(state.rotation_axis, state.cumulative_rotation))
                        facet_normal_rotated = state.facet_normals[facet_id].dot(current_rot_mat.T)
                        horizon_val = np.dot(facet_normal_rotated, view_dir_world)
                        debug_print(state, f"  Facet Normal: [{facet_normal_rotated[0]:.3f}, {facet_normal_rotated[1]:.3f}, {facet_normal_rotated[2]:.3f}]")
                        debug_print(state, f"  Horizon Value: {horizon_val:.6f} (front-facing: {horizon_val < 0})")
                        
                        # View mode dome analysis
                        if state.view_mode and state.dome_flux_th is not None:
                            view_dir_body = current_rot_mat.T.dot(view_dir_world)
                            # FIXED: Use view direction in body coordinates directly with dome normals (both in world coordinates)
                            dome_cosines = np.dot(view_dir_body, state.dome_bin_normals.T)
                            positive_bins = np.sum(dome_cosines > 0)
                            best_bin = np.argmax(np.clip(dome_cosines, 0, None))
                            debug_print(state, f"  View Dir (body): [{view_dir_body[0]:.3f}, {view_dir_body[1]:.3f}, {view_dir_body[2]:.3f}]")
                            debug_print(state, f"  Dome cosines: min={dome_cosines.min():.6f}, max={dome_cosines.max():.6f}")
                            debug_print(state, f"  Positive bins: {positive_bins}/{len(dome_cosines)}, Best bin: {best_bin}")
                    else:
                        debug_print(state, f"[ERROR] Frame {state.current_frame} out of bounds for facet {facet_id}")

        plotter.iren.add_observer('EndInteractionEvent', _camera_moved_event)
        plotter.iren.add_observer('TimerEvent', update_callback)
        plotter.iren.create_timer(100)  # Creates a repeating timer that triggers every 100 ms
        
        plotter.show()

    # Clean up
    plotter.close()
    plotter.deep_clean()

    if state.fig:
        plt.close(state.fig)
    
    pv.close_all()
    plt.close('all')

    if 'vtk_labels' in locals():
        vtk_labels.RemoveAllObservers()
        del vtk_labels

    state.fig = None
    state.ax = None

    gc.collect()

def update_highlight_mesh(state, plotter, pv_mesh):
    # Handle both initial creation and updates
    if state.highlighted_cell_ids:
        # Initialize highlight system if needed
        if not hasattr(state, 'highlight_meshes'):
            state.highlight_meshes = []
        if not hasattr(state, 'highlight_cell_mapping'):
            state.highlight_cell_mapping = {}
        
        # Remove highlights for cells that are no longer selected
        cells_to_remove = []
        for i, cell_id in enumerate(state.highlight_cell_mapping.keys()):
            if cell_id not in state.highlighted_cell_ids:
                cells_to_remove.append(cell_id)
        
        for cell_id in cells_to_remove:
            mesh_index = state.highlight_cell_mapping.pop(cell_id)
            if mesh_index < len(state.highlight_meshes):
                try:
                    plotter.remove_actor(state.highlight_meshes[mesh_index])
                    state.highlight_meshes[mesh_index] = None
                except:
                    pass
        
        # Clean up None entries
        state.highlight_meshes = [m for m in state.highlight_meshes if m is not None]
        # Rebuild mapping
        state.highlight_cell_mapping = {list(state.highlighted_cell_ids)[i]: i 
                                       for i in range(len(state.highlighted_cell_ids))}
        
        # Color mapping
        color_map = {
            'red': [1.0, 0.0, 0.0], 'blue': [0.0, 0.0, 1.0], 'green': [0.0, 1.0, 0.0],
            'orange': [1.0, 0.5, 0.0], 'purple': [0.5, 0.0, 1.0], 'cyan': [0.0, 1.0, 1.0],
            'yellow': [1.0, 1.0, 0.0], 'magenta': [1.0, 0.0, 1.0], 'white': [1.0, 1.0, 1.0]
        }
        
        # Update or create highlight meshes  
        while len(state.highlight_meshes) < len(state.highlighted_cell_ids):
            state.highlight_meshes.append(None)
            
        for i, cell_id in enumerate(state.highlighted_cell_ids):
            if cell_id < pv_mesh.n_cells:
                # Extract cell with current rotation
                single_cell = pv_mesh.extract_cells([cell_id])
                color = state.highlight_colors.get(cell_id, 'white')
                
                if isinstance(color, str):
                    rgb_color = color_map.get(color, [1.0, 1.0, 1.0])
                else:
                    rgb_color = color[:3] if len(color) >= 3 else [1.0, 1.0, 1.0]
                
                # Remove old mesh if exists
                if state.highlight_meshes[i] is not None:
                    try:
                        plotter.remove_actor(state.highlight_meshes[i])
                    except:
                        pass
                
                # Add new mesh with current rotation
                mesh_actor = plotter.add_mesh(single_cell, color=rgb_color, 
                                           opacity=1.0, style='wireframe', line_width=4,
                                           show_scalar_bar=False)
                state.highlight_meshes[i] = mesh_actor
    else:
        # Clear all highlights if no cells selected
        if hasattr(state, 'highlight_meshes') and state.highlight_meshes:
            for mesh_actor in state.highlight_meshes:
                if mesh_actor is not None:
                    try:
                        plotter.remove_actor(mesh_actor)
                    except:
                        pass
            state.highlight_meshes.clear()
        if hasattr(state, 'highlight_cell_mapping'):
            state.highlight_cell_mapping.clear()

def update(caller, event, state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label):
    """Update mesh rotation, shading, and debugging information."""
    try:
        # Update cumulative rotation and current frame
        if not state.is_paused:
            state.current_frame = (state.current_frame + 1) % state.timesteps_per_day
            state.cumulative_rotation = (state.current_frame / state.timesteps_per_day) * 2 * math.pi

        # Apply rotation
        rot_mat = np.array(rotation_matrix(rotation_axis, state.cumulative_rotation))
        rotated_vertices = vertices.dot(rot_mat.T)
        pv_mesh.points = rotated_vertices
        
        # Highlight meshes will be updated after mesh rotation via update_highlight_mesh()
        
        # Handle shading based on view mode
        if state.view_mode and state.dome_flux_th is not None:
            try:
                # Dome-based shading for view-dependent temperature display
                cam_pos = np.array(plotter.camera.position)
                focal = np.array(plotter.camera.focal_point)
                view_dir_world = (focal - cam_pos)
                view_dir_world /= np.linalg.norm(view_dir_world)
                
                # Transform view direction to body coordinates 
                view_dir_body = rot_mat.T.dot(view_dir_world)
                
                # Use view direction in body coordinates directly with dome normals (both in world coordinates)
                # Note: dome_bin_normals are in world coordinates, so we compare with view_dir_body
                cosines = np.dot(view_dir_body, state.dome_bin_normals.T).reshape(1, -1)
                cosines = np.broadcast_to(cosines, (len(state.facet_normals), len(state.dome_bin_normals)))
                cosines = np.clip(cosines, 0, None)
                best_bins = np.argmax(cosines, axis=1)
                selected_cosines = cosines[np.arange(len(best_bins)), best_bins]
                
                # Compute facet horizon test: rotated facet normal dot view direction
                normals_t = np.dot(state.facet_normals, rot_mat.T)
                facet_horizons = np.dot(normals_t, view_dir_world)
                
                # Extract flux and area data for selected bins
                F_values = state.dome_flux_th[np.arange(len(best_bins)), best_bins, state.current_frame]
                area_values = state.dome_bin_areas[np.arange(len(best_bins)), best_bins]
                
                # Apply scaling factor and compute temperatures
                scale = state.dome_radius_factor ** 2
                sigma = 5.670374419e-8
                epsilon = state.simulation_emissivity
                
                # Compute raw temperatures
                T_raw = np.zeros(len(F_values))
                valid_areas = area_values > 0
                T_raw[valid_areas] = ((F_values[valid_areas] / area_values[valid_areas] * scale) / (epsilon * sigma)) ** 0.25
                
                # Apply horizon and dome visibility masks
                horizon_tolerance = 1e-6
                mask_dome = selected_cosines > horizon_tolerance  # Dome patch visible
                mask_facet = facet_horizons < -horizon_tolerance  # Facet front-facing
                mask_combined = mask_dome & mask_facet
                
                # Set final temperatures: 0K if not visible, raw temperature if visible
                T = np.where(mask_combined, T_raw, 0.0)
                
                # Debug dome coverage issues every 30 frames
                if state.current_frame % 30 == 0:
                    front_facing = facet_horizons < -horizon_tolerance
                    dome_visible = selected_cosines > horizon_tolerance
                    problematic = front_facing & ~dome_visible & (T_raw > 100)
                    
                    if np.any(problematic):
                        prob_count = np.sum(problematic)
                        debug_print(state, f"\n[ERROR] DOME COVERAGE ISSUE - Frame {state.current_frame}:")
                        debug_print(state, f"  {prob_count} front-facing facets have no suitable dome bins")
                        
                        prob_ids = np.where(problematic)[0]
                        for pid in prob_ids[:2]:  # Show details for first 2 problematic facets
                            debug_print(state, f"  Facet {pid}:")
                            debug_print(state, f"    facet_horizon={facet_horizons[pid]:.6f} (front-facing: {front_facing[pid]})")
                            debug_print(state, f"    selected_bin={best_bins[pid]}, selected_cosine={selected_cosines[pid]:.6f}")
                            debug_print(state, f"    T_raw={T_raw[pid]:.1f}K -> will show 0K")
                            debug_print(state, f"    view_dir_body={view_dir_body}")
                            debug_print(state, f"    unclipped_dome_cosines: min={selected_cosines[pid]:.6f}, max={selected_cosines[pid]:.6f}")
                            
                            # Check unclipped cosines to see if dome actually covers the hemisphere
                            unclipped_cosines = np.dot(view_dir_body, state.dome_bin_normals.T)
                            debug_print(state, f"    unclipped_dome_cosines: min={unclipped_cosines.min():.6f}, max={unclipped_cosines.max():.6f}")
                            
                            positive_bins = np.sum(unclipped_cosines > 0)
                            debug_print(state, f"    {positive_bins}/{len(unclipped_cosines)} dome bins have positive unclipped cosines")
                            
                            if positive_bins == 0:
                                debug_print(state, f"    ERROR: No dome bins face the view direction - dome coverage incomplete!")
                            else:
                                debug_print(state, f"    ERROR: Dome bins exist but being incorrectly filtered out!")
                    
                    # Also check for the reverse: dome visible but facet not front-facing  
                    reverse_problematic = ~front_facing & dome_visible & (T_raw > 100)
                    if np.any(reverse_problematic):
                        rev_count = np.sum(reverse_problematic)
                        debug_print(state, f"[DEBUG] Frame {state.current_frame}: {rev_count} back-facing facets with visible dome patches")
                
                pv_mesh.cell_data[axis_label] = T
                plotter.update_scalar_bar_range((T.min(), T.max()))
            except Exception as e:
                debug_print(state, f"[DEBUG] Error in dome-based shading: {e}")
                debug_print(state, f"[DEBUG] Error occurred at line: {e.__traceback__.tb_lineno}")
                import traceback
                debug_print(state, f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
                # Fall back to default view-based shading
                state.view_mode = False
                pv_mesh.cell_data[axis_label] = plotted_variable_array[:, state.current_frame]
                plotter.update_scalar_bar_range((state.current_min, state.current_max))
        else:
            # Standard mode - no view-dependent shading
            pv_mesh.cell_data[axis_label] = plotted_variable_array[:, state.current_frame]
            plotter.update_scalar_bar_range((state.current_min, state.current_max))
        
        # Update highlights only when NOT paused (to avoid flashing) or when selections changed
        if state.highlighted_cell_ids and (not state.is_paused or getattr(state, 'highlights_need_update', False)):
            # Remove all existing highlight meshes
            if hasattr(state, 'highlight_meshes') and state.highlight_meshes:
                for mesh_actor in state.highlight_meshes:
                    try:
                        plotter.remove_actor(mesh_actor)
                    except:
                        pass
                state.highlight_meshes.clear()
            
            # Recreate highlights from rotated mesh
            if not hasattr(state, 'highlight_meshes'):
                state.highlight_meshes = []
            
            color_map = {
                'red': [1.0, 0.0, 0.0], 'blue': [0.0, 0.0, 1.0], 'green': [0.0, 1.0, 0.0],
                'orange': [1.0, 0.5, 0.0], 'purple': [0.5, 0.0, 1.0], 'cyan': [0.0, 1.0, 1.0],
                'yellow': [1.0, 1.0, 0.0], 'magenta': [1.0, 0.0, 1.0], 'white': [1.0, 1.0, 1.0]
            }
            
            for cell_id in state.highlighted_cell_ids:
                if cell_id < pv_mesh.n_cells:
                    single_cell = pv_mesh.extract_cells([cell_id])
                    color = state.highlight_colors.get(cell_id, 'white')
                    
                    if isinstance(color, str):
                        rgb_color = color_map.get(color, [1.0, 1.0, 1.0])
                    else:
                        rgb_color = color[:3] if len(color) >= 3 else [1.0, 1.0, 1.0]
                    
                    mesh_actor = plotter.add_mesh(single_cell, color=rgb_color, 
                                               opacity=1.0, style='wireframe', line_width=4,
                                               show_scalar_bar=False)
                    state.highlight_meshes.append(mesh_actor)
            
            # Reset the update flag
            state.highlights_need_update = False

    except Exception as e:
        debug_print(state, f"Error updating mesh: {e}")
        debug_print(state, f"Debug info: current_frame={state.current_frame}, shape of plotted_variable_array={plotted_variable_array.shape}")
        # Don't return here, just fall back to raw data
        try:
            pv_mesh.cell_data[axis_label] = plotted_variable_array[:, state.current_frame]
            plotter.update_scalar_bar_range((state.current_min, state.current_max))
        except:
            pass
        return

    if state.fig is not None and state.ax is not None:
        try:
            if state.time_line is None:
                state.time_line = state.ax.axvline(x=state.current_frame / state.timesteps_per_day, color='r', linestyle='--', label='Current Time')
            else:
                state.time_line.set_xdata([state.current_frame / state.timesteps_per_day])
            state.fig.canvas.draw()
            state.fig.canvas.flush_events()
        except Exception as e:
            debug_print(state, f"[WARNING] Matplotlib update failed: {e}")

    # Comprehensive debugging for all selected facets at every frame
    for facet_id, line in state.cell_colors.items():
        shading_val = pv_mesh.cell_data[axis_label][facet_id]
        ydata = line.get_ydata()
        if state.current_frame < len(ydata):
            plotted_val = ydata[state.current_frame]
            
            # Always print debug info for selected facets, regardless of mismatch
            debug_print(state, f"\n[FACET DEBUG] Facet {facet_id} at Frame {state.current_frame}:")
            debug_print(state, f"  3D Shading Value: {shading_val:.2f}K")
            debug_print(state, f"  2D Plotted Value: {plotted_val:.2f}K")
            debug_print(state, f"  Difference: {abs(shading_val - plotted_val):.2f}K")
            
            # Get camera and view information
            cam_pos = np.array(plotter.camera.position)
            focal = np.array(plotter.camera.focal_point)
            view_dir_world = (focal - cam_pos)
            view_dir_world /= np.linalg.norm(view_dir_world)
            debug_print(state, f"  Camera Position: [{cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}]")
            debug_print(state, f"  View Direction (world): [{view_dir_world[0]:.3f}, {view_dir_world[1]:.3f}, {view_dir_world[2]:.3f}]")
            
            # Current rotation information
            current_rot_mat = np.array(rotation_matrix(state.rotation_axis, state.cumulative_rotation))
            debug_print(state, f"  Body Rotation Angle: {state.cumulative_rotation:.3f} rad ({np.degrees(state.cumulative_rotation):.1f}°)")
            
            # Facet information
            facet_normal_world = state.facet_normals[facet_id]
            facet_normal_rotated = facet_normal_world.dot(current_rot_mat.T)
            debug_print(state, f"  Facet Normal (world): [{facet_normal_world[0]:.3f}, {facet_normal_world[1]:.3f}, {facet_normal_world[2]:.3f}]")
            debug_print(state, f"  Facet Normal (rotated): [{facet_normal_rotated[0]:.3f}, {facet_normal_rotated[1]:.3f}, {facet_normal_rotated[2]:.3f}]")
            
            # Horizon test
            horizon_val = np.dot(facet_normal_rotated, view_dir_world)
            front_facing = horizon_val < 0
            debug_print(state, f"  Horizon Value: {horizon_val:.6f} (front-facing: {front_facing})")
            
            # View mode specific debugging
            if state.view_mode and state.dome_flux_th is not None:
                debug_print(state, f"  === VIEW MODE DEBUGGING ===")
                
                # Transform view direction to body frame (world coordinates)
                view_dir_body = current_rot_mat.T.dot(view_dir_world)
                debug_print(state, f"  View Direction (body): [{view_dir_body[0]:.3f}, {view_dir_body[1]:.3f}, {view_dir_body[2]:.3f}]")
                
                # Dome bin analysis - FIXED: Use same calculation as 3D mesh (both in world coordinates)
                dome_cosines_unclipped = np.dot(view_dir_body, state.dome_bin_normals.T)
                dome_cosines_clipped = np.clip(dome_cosines_unclipped, 0, None)
                best_bin = np.argmax(dome_cosines_clipped)
                selected_cosine = dome_cosines_clipped[best_bin]
                
                debug_print(state, f"  Dome Cosines (unclipped): min={dome_cosines_unclipped.min():.6f}, max={dome_cosines_unclipped.max():.6f}")
                debug_print(state, f"  Positive Bins: {np.sum(dome_cosines_unclipped > 0)}/{len(dome_cosines_unclipped)}")
                debug_print(state, f"  Selected Bin: {best_bin}, Cosine: {selected_cosine:.6f}")
                debug_print(state, f"  Selected Bin Normal: [{state.dome_bin_normals[best_bin][0]:.3f}, {state.dome_bin_normals[best_bin][1]:.3f}, {state.dome_bin_normals[best_bin][2]:.3f}]")
                
                # Flux and area information
                F_val = state.dome_flux_th[facet_id, best_bin, state.current_frame]
                area_val = state.dome_bin_areas[facet_id, best_bin]
                debug_print(state, f"  Flux: {F_val:.3e}, Area: {area_val:.6f}")
                
                # Final temperature calculation
                scale = state.dome_radius_factor ** 2
                horizon_tolerance = 1e-6
                mask_dome = selected_cosine > horizon_tolerance
                mask_facet = horizon_val < -horizon_tolerance
                mask_combined = mask_dome and mask_facet
                
                if area_val > 0 and mask_combined:
                    T_calc = ((F_val / area_val * scale) / (state.simulation_emissivity * 5.670374419e-8)) ** 0.25
                else:
                    T_calc = 0.0
                
                debug_print(state, f"  Mask Dome Visible: {mask_dome}, Mask Facet Front: {mask_facet}, Combined: {mask_combined}")
                debug_print(state, f"  Calculated Temperature: {T_calc:.2f}K")
                debug_print(state, f"  Raw Temperature (no masking): {((F_val / area_val * scale) / (state.simulation_emissivity * 5.670374419e-8)) ** 0.25 if area_val > 0 else 0.0:.2f}K")
                
                if not np.isclose(shading_val, T_calc, atol=1e-6):
                    debug_print(state, f"  *** CALCULATION MISMATCH: Expected {T_calc:.2f}K, Got {shading_val:.2f}K ***")
                
            else:
                debug_print(state, "  === NORMAL MODE (non-view-dependent) ===")
                debug_print(state, "  Using raw temperature data from simulation")
                
        else:
            debug_print(state, f"[ERROR] Frame {state.current_frame} out of bounds for plotted data length {len(ydata)} (facet {facet_id})")

    plotter.render()