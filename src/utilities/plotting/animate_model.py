"""
Animate Model Script

This script creates an interactive 3D animation of a shape model, allowing the user to visualize various properties over time (e.g., temperature, illumination). It allows the user to pause, select individual facets, and display their properties over time on a separate graph.

Known Issues:
1. Segmentation Fault: The script crashes with a segmentation fault the second time it is called by the model script. 

TODO: 
Option to make rotation axis and sun invisible 
"""

import pyvista as pv
import vtk
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

def get_next_color(state):
    color = state.color_cycle[state.color_index % len(state.color_cycle)]
    state.color_index += 1
    return color[:3]

def on_press(state):
    state.is_paused = not state.is_paused
    if state.is_paused:
        state.pause_time = state.current_frame / state.timesteps_per_day
    else:
        state.pause_time = None

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
    if state.highlight_mesh is not None:
        plotter.remove_actor(state.highlight_mesh)
        state.highlight_mesh = None
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
    else:
        # Add the cell if it's not highlighted
        state.highlighted_cell_ids.append(cell_id)
        color = get_next_color(state)
        state.highlight_colors[cell_id] = color

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
                        time_steps = [i / state.timesteps_per_day for i in range(plotted_variable_array.shape[1])]
                        
                        for facet_id in state.highlighted_cell_ids:
                            data[f'Facet_{facet_id}'] = plotted_variable_array[facet_id, :]
                        
                        # Create DataFrame
                        df = pd.DataFrame(data, index=time_steps)
                        df.index.name = 'Time (fraction of day)'
                        
                        # Create a valid filename from the axis label
                        property_name = axis_label.lower().replace(' ', '_')
                        timestamp = datetime.datetime.now().strftime("%H%M%S")
                        filename = os.path.join(run_dir, f'{property_name}_vs_timestep_{timestamp}.csv')
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
                            values = convert_to_local_time(values, 
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

            state.fig.canvas.mpl_connect('key_press_event', on_key)
            state.ax.set_xlabel("Fractional angle of rotation")
            state.ax.set_ylabel(axis_label)
            state.ax.set_title(f"{axis_label} Over Time for Selected Facets\nPress 'd' to save data, 't' to toggle local/global time")

        values_over_time = plotted_variable_array[cell_id, :]
        if state.use_local_time:
            values_over_time = convert_to_local_time(values_over_time, 
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

def get_next_color(state):
    color = state.color_cycle[state.color_index]
    state.color_index = (state.color_index + 1) % len(state.color_cycle)
    return color[:3]

def update_highlight_mesh(state, plotter, pv_mesh):
    if state.highlighted_cell_ids:
        highlight_mesh = pv.PolyData()
        cell_colors = []
        for cell_id in state.highlighted_cell_ids:
            color = state.highlight_colors[cell_id]
            cell = pv_mesh.extract_cells([cell_id])
            edges = cell.extract_feature_edges(feature_angle=0, boundary_edges=True, non_manifold_edges=False, manifold_edges=False)
            n_edges = edges.n_cells
            highlight_mesh += edges
            cell_colors.extend([color] * n_edges)
        
        cell_colors = np.array(cell_colors)
        cell_colors = cell_colors[::-1]

        if state.highlight_mesh is None:
            state.highlight_mesh = plotter.add_mesh(highlight_mesh, scalars=cell_colors, rgb=True, line_width=5, render_lines_as_tubes=True, opacity=1)
        else:
            # Remove the old highlight mesh
            plotter.remove_actor(state.highlight_mesh)
            # Add the new highlight mesh
            state.highlight_mesh = plotter.add_mesh(highlight_mesh, scalars=cell_colors, rgb=True, line_width=5, render_lines_as_tubes=True, opacity=1)
    elif state.highlight_mesh is not None:
        plotter.remove_actor(state.highlight_mesh)
        state.highlight_mesh = None
    
    plotter.render()

def update(caller, event, state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label):
    if not state.is_paused:
        state.current_frame = (state.current_frame + 1) % state.timesteps_per_day
        state.cumulative_rotation = (state.current_frame / state.timesteps_per_day) * 2 * math.pi

    rot_mat = np.array(rotation_matrix(rotation_axis, state.cumulative_rotation))

    try:
        # Apply rotation to mesh
        rotated_vertices = np.dot(vertices, rot_mat.T)
        pv_mesh.points = rotated_vertices
        
        # Update cell data with current range
        pv_mesh.cell_data[axis_label] = plotted_variable_array[:, state.current_frame]
        plotter.update_scalar_bar_range((state.current_min, state.current_max))
        
        # Update highlights only if there are highlighted cells
        if state.highlighted_cell_ids:
            update_highlight_mesh(state, plotter, pv_mesh)

    except Exception as e:
        print(f"Error updating mesh: {e}")
        print(f"Debug info: current_frame={state.current_frame}, shape of plotted_variable_array={plotted_variable_array.shape}")
        return

    if state.fig is not None and state.ax is not None:
        if state.time_line is None:
            state.time_line = state.ax.axvline(x=state.current_frame / state.timesteps_per_day, color='r', linestyle='--', label='Current Time')
        else:
            state.time_line.set_xdata([state.current_frame / state.timesteps_per_day])
        state.fig.canvas.draw()
        state.fig.canvas.flush_events()

    plotter.render()

def on_pick(state, picked_mesh, plotter, pv_mesh, plotted_variable_array, axis_label):
    if picked_mesh is not None:
        cell_id = picked_mesh['vtkOriginalCellIds'][0]
        plot_picked_cell_over_time(state, cell_id, plotter, pv_mesh, plotted_variable_array, axis_label)

def convert_to_local_time(global_time_data, facet_normal, sunlight_direction, rotation_axis, facet_id):
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
    
    conditional_print(False, f"Facet {facet_id} shifted by {shift} timesteps ({(shift/timesteps * 360):.1f} degrees)")
    
    return np.roll(global_time_data, shift)

def animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction, 
                  timesteps_per_day, solar_distance_au, rotation_period_hr, colour_map, plot_title, axis_label, animation_frames, 
                  save_animation, save_animation_name, background_colour, pre_selected_facets=[1220, 845]):

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
    
    state = AnimationState()
    state.timesteps_per_day = timesteps_per_day
    state.shape_model_name = path_to_shape_model_file
    state.facet_normals = facet_normals
    state.sunlight_direction = sunlight_direction
    state.rotation_axis = rotation_axis

    text_color = 'white' if background_colour == 'black' else 'black' 
    bar_color = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

    bounding_box = pv_mesh.bounds
    max_dimension = max(bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2], bounding_box[5] - bounding_box[4])
    state.camera_radius = max_dimension * 5

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing()
    plotter.add_key_event('space', lambda: on_press(state))
    plotter.add_key_event('Left', lambda: move_backward(state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label))
    plotter.add_key_event('Right', lambda: move_forward(state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label))
    plotter.add_key_event('c', lambda: clear_selections(state, plotter))
    plotter.add_key_event('r', lambda: reset_camera(state, plotter))  # Add the new key event for camera reset


    cylinder = pv.Cylinder(center=(0, 0, 0), direction=rotation_axis, height=max_dimension, radius=max_dimension/200)
    plotter.add_mesh(cylinder, color='green')

    sunlight_start = [sunlight_direction[i] * max_dimension for i in range(3)]
    sunlight_arrow = pv.Arrow(start=sunlight_start, direction=[-d for d in sunlight_direction], scale=max_dimension * 0.3)
    plotter.add_mesh(sunlight_arrow, color='yellow')

    plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False)

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
    plotter.add_text("'Spacebar' - Pause/play\n'Right click' - Select facet\n'C' - Clear selections\n'L/R Arrow keys' - Rotate\n'R' - Reset camera", position='upper_left', font_size=10, color=text_color)

    info_text = f"Solar Distance: {solar_distance_au} AU\nRotation Period: {rotation_period_hr} hours"
    plotter.add_text(info_text, position='upper_right', font_size=10, color=text_color)

    plotter.background_color = background_colour

    def update_callback(caller, event):
        if not state.is_paused:
            move_forward(state, plotter, pv_mesh, plotted_variable_array, vertices, rotation_axis, axis_label)
        plotter.render()

    if save_animation:
        plotter.open_gif(save_animation_name)
        for _ in range(animation_frames):
            update_callback(None, None)
            plotter.write_frame()
        plotter.close()
    else:
        end_time = time.time()
        print(f'Animation setup took {end_time - start_time:.2f} seconds.')
        
        # Set up the timer for live animation
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