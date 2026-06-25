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
import re
from src.utilities.locations import Locations
from src.utilities.utils import conditional_print
import h5py

def sanitize_filename(filename):
    """Remove or replace special characters in filename to ensure cross-platform compatibility."""
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove special characters that can cause filesystem issues
    filename = re.sub(r'[^\w\-_]', '', filename)
    # Replace multiple underscores with single underscore
    filename = re.sub(r'_+', '_', filename)
    return filename

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
        self.simulation_emissivity = None
        self.facet_areas = None
        self.last_camera_angles = None  # Only print camera direction when it changes
        self.highlights_need_update = False
        self.rotate_mesh = True

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
                        property_name = sanitize_filename(axis_label.lower())
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

def animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction,
                  timesteps_per_day, solar_distance_au, rotation_period_hr, emissivity, plot_title, axis_label, animation_frames,
                  save_animation, save_animation_name, background_colour, dome_radius_factor=1.0, colour_map='coolwarm',
                  apply_kernel_based_roughness=False, pre_selected_facets=[1220, 845], animation_debug_mode=False,
                  output_dir=None, center_mesh_at_origin=False, rotate_mesh=True):
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
        center_mesh_at_origin (bool): Translate the mesh so its bounding-box center is at the origin.
        rotate_mesh (bool): If False, keep the mesh geometry fixed and only update scalars per frame.
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

    # Downsample timesteps for animation if there are too many
    max_animation_timesteps = 360
    if plotted_variable_array.shape[1] > max_animation_timesteps:
        indices = np.linspace(0, plotted_variable_array.shape[1] - 1, max_animation_timesteps, dtype=int)
        plotted_variable_array = plotted_variable_array[:, indices]
        timesteps_per_day = max_animation_timesteps
        print(f"Downsampled animation to {max_animation_timesteps} timesteps for display.")

    vertices = shape_mesh.points.reshape(-1, 3).copy()
    original_vertices = vertices.copy()

    if center_mesh_at_origin:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        mesh_center = 0.5 * (mins + maxs)
        vertices = vertices - mesh_center
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
    # Store facet areas on state for dome area scaling
    state.facet_areas = facet_areas
    state.timesteps_per_day = timesteps_per_day
    state.shape_model_name = path_to_shape_model_file
    state.facet_normals = facet_normals
    state.original_vertices = original_vertices  # Store original unrotated, untranslated vertices for lat/lon calculation
    state.sunlight_direction = sunlight_direction
    state.rotation_axis = rotation_axis
    state.rotate_mesh = rotate_mesh
    if emissivity is None:
        raise ValueError("emissivity must be specified - it is a critical physical parameter")
    state.simulation_emissivity = emissivity
    debug_print(state, f"[DEBUG] Using emissivity: {state.simulation_emissivity}")

    text_color = 'white' if background_colour == 'black' else 'black' 
    bar_color = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

    bounding_box = pv_mesh.bounds
    max_dimension = max(bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2], bounding_box[5] - bounding_box[4])
    state.camera_radius = max_dimension * 5

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    # No lightkit: use flat lighting so colormap values match the colorbar exactly
    plotter.add_key_event('space', lambda: (setattr(state, 'is_paused', not state.is_paused), setattr(state, 'pause_time', (state.current_frame / state.timesteps_per_day) if state.is_paused else None), debug_print(state, f"[DEBUG] Toggled pause to {state.is_paused}")))
    plotter.add_key_event('Left', lambda: move_backward(state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label))
    plotter.add_key_event('Right', lambda: move_forward(state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label))
    plotter.add_key_event('c', lambda: clear_selections(state, plotter))
    plotter.add_key_event('r', lambda: reset_camera(state, plotter))  # Add the new key event for camera reset

    cylinder = pv.Cylinder(center=(0, 0, 0), direction=rotation_axis, height=max_dimension, radius=max_dimension/200)
    plotter.add_mesh(cylinder, color='green')

    sunlight_start = [sunlight_direction[i] * max_dimension for i in range(3)]
    sunlight_arrow = pv.Arrow(start=sunlight_start, direction=[-d for d in sunlight_direction], scale=max_dimension * 0.3)
    plotter.add_mesh(sunlight_arrow, color='yellow')

    plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False, lighting=False, smooth_shading=True)

    plotter.enable_element_picking(callback=lambda picked_mesh: on_pick(state, picked_mesh, plotter, pv_mesh, plotted_variable_array, axis_label), mode='cell', show=False, show_message=False)

    text_color_rgb = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

    state.initial_camera_position = plotter.camera.position
    state.initial_camera_focal_point = plotter.camera.focal_point
    state.initial_camera_up = plotter.camera.up

    plotter.scalar_bar.GetLabelTextProperty().SetColor(bar_color)
    plotter.scalar_bar.SetPosition(0.3, 0.05)
    plotter.scalar_bar.GetLabelTextProperty().SetJustificationToCentered()
    plotter.scalar_bar.SetTitle(axis_label)
    plotter.scalar_bar.GetTitleTextProperty().SetColor(text_color_rgb)

    min_val = np.min(plotted_variable_array)
    max_val = np.max(plotted_variable_array)

    # Create 7 labels: min + 5 intermediate + max
    num_labels = 7
    labels = np.linspace(min_val, max_val, num_labels)
    
    # Determine decimal places based on range - use consistent precision
    value_range = max_val - min_val
    if value_range > 0:
        exponent = math.floor(math.log10(value_range))
        # Calculate decimal places so we show meaningful variation
        decimal_places = max(1, -exponent + 1)
    else:
        decimal_places = 1
    
    # Set up scalar bar with custom labels
    scalar_bar = plotter.scalar_bar
    
    # Create VTK array for label values
    vtk_labels = vtk.vtkDoubleArray()
    vtk_labels.SetNumberOfValues(len(labels))
    for i, label in enumerate(labels):
        vtk_labels.SetValue(i, label)
    
    scalar_bar.SetCustomLabels(vtk_labels)
    scalar_bar.SetUseCustomLabels(True)
    
    # Set label format string for consistent decimal places
    format_string = f'%.{decimal_places}f'
    scalar_bar.SetLabelFormat(format_string)
    
    # Configure scalar bar for better visibility
    scalar_bar.SetNumberOfLabels(num_labels)
    scalar_bar.GetTitleTextProperty().SetColor(text_color_rgb)

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
        pointa=(0.025, 0.22),
        pointb=(0.225, 0.22),
        style='modern'
    )

    plotter.add_text(plot_title, position='upper_edge', font_size=12, color=text_color)
    plotter.add_text("'Spacebar' - Pause/play\n'Right click' - Select facet\n'C' - Clear selections\n'R' - Reset camera", position='upper_left', font_size=10, color=text_color)

    info_text = f"Solar Distance: {solar_distance_au} AU\nRotation Period: {rotation_period_hr} hours"
    plotter.add_text(info_text, position='upper_right', font_size=10, color=text_color)

    plotter.background_color = background_colour

    def update_callback(caller, event):
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

        # Apply rotation (optional)
        if getattr(state, 'rotate_mesh', True):
            rot_mat = np.array(rotation_matrix(rotation_axis, state.cumulative_rotation))
            rotated_vertices = vertices.dot(rot_mat.T)
            pv_mesh.points = rotated_vertices
        else:
            # Keep geometry fixed; only update scalars over time
            pv_mesh.points = vertices
        
        # Highlight meshes will be updated after mesh rotation via update_highlight_mesh()
        
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
        print(f"Animation update error: {e}")
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
            
        else:
            debug_print(state, f"[ERROR] Frame {state.current_frame} out of bounds for plotted data length {len(ydata)} (facet {facet_id})")

    plotter.render()
def plot_static_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction,
                     solar_distance_au, rotation_period_hr, emissivity, plot_title, axis_label, 
                     background_colour='black', dome_radius_factor=1.0, colour_map='coolwarm',
                     apply_kernel_based_roughness=False):
    """
    Create a static 3D visualization of a shape model with scalar values (e.g., peak temperatures).
    
    Args:
        path_to_shape_model_file (str): Path to the STL file
        plotted_variable_array (np.ndarray): 1D array of values to plot (one per facet)
        rotation_axis (np.ndarray): Axis of rotation
        sunlight_direction (np.ndarray): Direction of sunlight
        solar_distance_au (float): Solar distance in AU
        rotation_period_hr (float): Rotation period in hours
        emissivity (float): Emissivity of the surface
        plot_title (str): Title of the plot
        axis_label (str): Label for the colorbar
        background_colour (str): Background color of the plot
        dome_radius_factor (float): Scale factor for dome radius
        colour_map (str): Name of the colormap to use
        apply_kernel_based_roughness (bool): Whether kernel-based roughness is applied
    
    User-Editable Parameters (modify these values in the code for advanced customization):
        - use_orthographic: Set to True for orthographic projection, False for perspective
        - camera_direction: Camera view direction as [x, y, z]. Default: [0, 0, -1] (view from -z)
        - show_rotation_axis: Set to False to hide rotation axis cylinder visualization
        - show_sunlight: Set to False to hide sunlight arrow visualization
    """
    # ============================================================================
    # USER-EDITABLE PARAMETERS (modify these for advanced customization)
    # ============================================================================
    use_orthographic = True          # Use orthographic projection (True) or perspective (False)
    camera_direction = np.array([0, 0, -1])  # View direction (normalize to unit vector)
    show_rotation_axis = False       # Show/hide rotation axis visualization
    show_sunlight = False            # Show/hide sunlight direction visualization
    # ============================================================================
    
    # Lazy import of pyvista and vtk
    import pyvista as pv
    import vtk
    
    start_time = time.time()
    
    try:
        # Load the shape model
        shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file)
    except Exception as e:
        print(f"Failed to load shape model: {e}")
        return
    
    # Validate array dimensions
    if shape_mesh.vectors.shape[0] != plotted_variable_array.shape[0]:
        print("The plotted variable array must have the same number of rows as the number of cells in the shape model.")
        return
    
    # Create PyVista mesh
    vertices = shape_mesh.points.reshape(-1, 3).copy()
    faces = [[3, 3*i, 3*i+1, 3*i+2] for i in range(shape_mesh.vectors.shape[0])]
    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh.cell_data[axis_label] = plotted_variable_array
    
    # Set up the plotter
    background_colour = 'white'  # Enforce white background
    text_color = 'black'
    bar_color = (0, 0, 0)
    
    bounding_box = pv_mesh.bounds
    max_dimension = max(bounding_box[1] - bounding_box[0], 
                       bounding_box[3] - bounding_box[2], 
                       bounding_box[5] - bounding_box[4])
    
    # Normalize camera direction
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    
    # Calculate camera position to view from specified direction
    # Position camera far enough to see the entire object
    center = np.array(pv_mesh.center)
    camera_pos = center + camera_direction * max_dimension * 3
    
    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    plotter.background_color = background_colour
    
    # Set orthographic or perspective projection
    if use_orthographic:
        plotter.camera.parallel_projection = True
    else:
        plotter.camera.parallel_projection = False
    
    # Set camera position and orientation
    plotter.camera.position = camera_pos
    plotter.camera.focal_point = center
    plotter.camera.up = np.array([0, -1, 0])  # Y-axis points up
    
    # For orthographic projection, set parallel_scale to fit the entire model
    if use_orthographic:
        # Calculate parallel_scale based on bounding box
        # parallel_scale is the height of the viewport in world units
        x_range = bounding_box[1] - bounding_box[0]
        y_range = bounding_box[3] - bounding_box[2]
        z_range = bounding_box[5] - bounding_box[4]
        
        # For -z view, we care about x and y extents; add 30% padding
        viewport_height = max(x_range, y_range) * 1.3 / 2
        plotter.camera.parallel_scale = viewport_height
    
    # Add the mesh with the scalar values
    plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False, 
                    lighting=False, smooth_shading=True)
    
    # Add visualization aids (rotation axis and sunlight direction)
    if show_rotation_axis:
        cylinder = pv.Cylinder(center=center, direction=rotation_axis, 
                              height=max_dimension, radius=max_dimension/200)
        plotter.add_mesh(cylinder, color='green', opacity=0.3)
    
    if show_sunlight:
        sunlight_start = center + np.array(sunlight_direction) * max_dimension
        sunlight_arrow = pv.Arrow(start=sunlight_start, direction=[-d for d in sunlight_direction], 
                                 scale=max_dimension * 0.3)
        plotter.add_mesh(sunlight_arrow, color='yellow')
    
    # Set up scalar bar
    text_color_rgb = (0, 0, 0)
    plotter.scalar_bar.GetLabelTextProperty().SetColor(bar_color)
    plotter.scalar_bar.SetPosition(0.3, 0.05)
    plotter.scalar_bar.GetLabelTextProperty().SetJustificationToCentered()
    plotter.scalar_bar.SetTitle(axis_label)
    plotter.scalar_bar.GetTitleTextProperty().SetColor(text_color_rgb)
    
    # Set up scalar bar labels
    min_val = np.min(plotted_variable_array)
    max_val = np.max(plotted_variable_array)
    
    num_labels = 7
    labels = np.linspace(min_val, max_val, num_labels)
    
    # Determine decimal places based on range
    value_range = max_val - min_val
    if value_range > 0:
        exponent = math.floor(math.log10(value_range))
        decimal_places = max(1, -exponent + 1)
    else:
        decimal_places = 1
    
    format_string = f'%.{decimal_places}f'
    scalar_bar = plotter.scalar_bar
    
    vtk_labels = vtk.vtkDoubleArray()
    vtk_labels.SetNumberOfValues(len(labels))
    for i, label in enumerate(labels):
        vtk_labels.SetValue(i, label)
    
    scalar_bar.SetCustomLabels(vtk_labels)
    scalar_bar.SetUseCustomLabels(True)
    scalar_bar.SetLabelFormat(format_string)
    scalar_bar.SetNumberOfLabels(num_labels)
    
    # Add title and information text
    plotter.add_text(plot_title, position='upper_edge', font_size=12, color=text_color)
    
    projection_type = "Orthographic" if use_orthographic else "Perspective"
    info_text = f"View: {projection_type} from {tuple(np.round(camera_direction, 2))}\nSolar Distance: {solar_distance_au} AU\nRotation Period: {rotation_period_hr} hours"
    plotter.add_text(info_text, position='upper_right', font_size=10, color=text_color)
    
    # Add sliders for min/max scaling
    data_min = min_val
    data_max = max_val
    range_padding = (data_max - data_min) * 0.1
    
    def update_min(value):
        plotter.update_scalar_bar_range((value, data_max))
        pv_mesh.set_active_scalars(axis_label)
        plotter.render()
    
    def update_max(value):
        plotter.update_scalar_bar_range((data_min, value))
        pv_mesh.set_active_scalars(axis_label)
        plotter.render()
    
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
        pointa=(0.025, 0.22),
        pointb=(0.225, 0.22),
        style='modern'
    )
    
    # Display the plot
    end_time = time.time()
    print(f'Static plot setup took {end_time - start_time:.2f} seconds.')
    print(f'Camera position: {camera_pos}, Focal point: {center}')
    
    plotter.show()
    
    # Clean up
    plotter.close()
    plotter.deep_clean()
    pv.close_all()
    
    if 'vtk_labels' in locals():
        vtk_labels.RemoveAllObservers()
        del vtk_labels
    
    gc.collect()
