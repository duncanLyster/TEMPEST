'''
Script to animate model quantities on the shape. The model is rotated about a specified axis and the quantity is plotted on the shape. You can move the camera around the shape and pause the animation with the arrow keys and spacebar. Right click on the shape to get the value of the quantity at that point.

TODO:
1) Leave animation up (or save it in an interactable mode) but allow the main code to continue running.
2) Fix BUG - plot window doesn't close when you hit x.
3) Fix BUG - camera movement is jumpy when using arrow keys.
4) Sort out colour scale bar values - should be more sensibly spaced.

NOTE: This is currently very slow when called from the main script for large shape models.
'''

import numpy as np
import pyvista as pv
import vtk
from stl import mesh
import time
import math
import matplotlib.pyplot as plt

class AnimationState:
    def __init__(self):
        self.is_paused = False
        self.current_frame = 0
        self.camera_phi = np.pi / 2
        self.camera_theta = np.pi / 2
        self.selected_cells = []
        self.fig, self.ax = None, None

# def on_press(state):
#     state.is_paused = not state.is_paused

   # Dealing with segmentation faults
def on_press(state):
    if state.is_paused:
        state.is_paused = False
    else:
        state.is_paused = True

    if state.fig is not None and state.ax is not None:
        state.fig.canvas.draw()
        state.fig.canvas.flush_events()

def update_camera_position(plotter, state, camera_radius):
    # Convert spherical coordinates to Cartesian coordinates
    x = camera_radius * np.sin(state.camera_phi) * np.cos(state.camera_theta)
    y = camera_radius * np.sin(state.camera_phi) * np.sin(state.camera_theta)
    z = camera_radius * np.cos(state.camera_phi)

    plotter.camera_position = [(x, y, z), (0, 0, 0), (0, 0, 1)]
    # plotter.reset_camera()
    plotter.camera.view_angle = 30
    plotter.render()

def move_up(plotter, state, camera_radius):
    state.camera_phi -= np.pi / 36  # Decrease polar angle
    update_camera_position(plotter, state, camera_radius)

def move_down(plotter, state, camera_radius):
    state.camera_phi += np.pi / 36  # Increase polar angle
    update_camera_position(plotter, state, camera_radius)

def move_left(plotter, state, camera_radius):
    state.camera_theta -= np.pi / 36  # Decrease azimuthal angle
    update_camera_position(plotter, state, camera_radius)

def move_right(plotter, state, camera_radius):
    state.camera_theta += np.pi / 36  # Increase azimuthal angle
    update_camera_position(plotter, state, camera_radius)

def round_up_to_nearest(x, base):
    return base * math.ceil(x / base)

def round_down_to_nearest(x, base):
    return base * math.floor(x / base)

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis /= np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction, 
                  timesteps_per_day, colour_map, plot_title, axis_label, animation_frames, 
                  save_animation, save_animation_name, background_colour):

    state = AnimationState()

    # Start timer
    start_time = time.time()

    # Load shape model
    shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file)
    vertices = shape_mesh.points.reshape(-1, 3)
    faces = np.hstack([np.full((shape_mesh.vectors.shape[0], 1), 3), 
                    np.arange(shape_mesh.vectors.shape[0] * 3).reshape(-1, 3)])

    # Create a PyVista mesh
    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh.cell_data[axis_label] = plotted_variable_array[:, 0]

    # Determine text color based on background color
    text_color = 'white' if background_colour=='black' else 'black' 
    bar_color = (1, 1, 1) if background_colour=='black' else (0, 0, 0)

    # Animation dimension calculations
    bounding_box = pv_mesh.bounds
    max_dimension = max(bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2], bounding_box[5] - bounding_box[4])
    camera_radius = max_dimension * 5

    # Create a Plotter object
    plotter = pv.Plotter()
    plotter.add_key_event('space', lambda: on_press(state))
    plotter.add_key_event('Up', lambda: move_up(plotter, state, camera_radius))
    plotter.add_key_event('Down', lambda: move_down(plotter, state, camera_radius))
    plotter.add_key_event('Left', lambda: move_left(plotter, state, camera_radius))
    plotter.add_key_event('Right', lambda: move_right(plotter, state, camera_radius))
    plotter.iren.initialize()

    update_camera_position(plotter, state, camera_radius)

    # Axis
    cylinder_start = np.array([0, 0, 0])
    cylinder_length = max_dimension  # Adjust the scale factor as needed
    cylinder = pv.Cylinder(center=cylinder_start, direction=rotation_axis, height=cylinder_length, radius=max_dimension/200)

    plotter.add_mesh(cylinder, color='green')

    # Sunlight vector
    sunlight_start = np.array([0, 0, 0]) + sunlight_direction * max_dimension
    sunlight_length = max_dimension * 0.3  # Adjust the scale factor as needed
    sunlight_arrow = pv.Arrow(start=sunlight_start, direction= -sunlight_direction, scale=sunlight_length)

    plotter.add_mesh(sunlight_arrow, color='yellow')

    # Add the text to the window
    plotter.add_text("Press spacebar to pause, right click to select a facet.", position='lower_edge', font_size=10, color=text_color)

    def update(caller, event, state):
        if not state.is_paused:
            state.current_frame = (state.current_frame + sampling_interval) % timesteps_per_day

            if state.current_frame >= num_frames:
                state.current_frame = 0

            theta = (2 * np.pi / timesteps_per_day) * state.current_frame
            rot_mat = rotation_matrix(rotation_axis, theta)
            rotated_vertices = np.dot(vertices, rot_mat.T)

            pv_mesh.points = rotated_vertices
            pv_mesh.cell_data[axis_label] = plotted_variable_array[:, state.current_frame % timesteps_per_day].copy()

            plotter.render()

    plotter.iren.add_observer('TimerEvent', lambda caller, event: update(caller, event, state))
    plotter.iren.create_timer(100)

    def plot_picked_cell_over_time(state, cell_id):
        if state.selected_cells is None:
            state.selected_cells = [cell_id]
        else:
            state.selected_cells.append(cell_id)

        if state.fig is None or state.ax is None:
            plt.ion()
            state.fig, state.ax = plt.subplots()

        state.ax.clear() 

        for cell in state.selected_cells:
            values_over_time = plotted_variable_array[cell, :]
            time_steps = np.arange(len(values_over_time)) / timesteps_per_day
            state.ax.plot(time_steps, values_over_time, label=f'Cell {cell}')

        state.ax.set_xlabel('Local time (days)')
        state.ax.set_ylabel(axis_label)
        state.ax.set_title(f'Diurnal variation of {axis_label} of selected Cells')
        state.ax.legend()
        state.fig.canvas.draw()
        state.fig.canvas.flush_events()


    def on_pick(state, picked_mesh):
        if picked_mesh is not None:
            cell_id = picked_mesh['vtkOriginalCellIds'][0]
            plot_picked_cell_over_time(state, cell_id)

    # Add the mesh to the plotter
    mesh_actor = plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False)
    plotter.enable_element_picking(callback=lambda picked_mesh: on_pick(state, picked_mesh), mode='cell', show_message=False)

    text_color_rgb = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

    # Scale bar properties
    plotter.scalar_bar.GetLabelTextProperty().SetColor(bar_color)
    plotter.scalar_bar.SetPosition(0.2, 0.05)  # Set the position of the bottom left corner of the scalar bar
    plotter.scalar_bar.GetLabelTextProperty().SetJustificationToCentered()  # Center the labels
    plotter.scalar_bar.SetTitle(axis_label)
    plotter.scalar_bar.GetTitleTextProperty().SetColor(text_color_rgb)

    # Get the range of the plotted variable
    min_val = np.min(plotted_variable_array)
    max_val = np.max(plotted_variable_array)
    range_val = max_val - min_val

    # Determine a suitable interval
    interval = 10 ** (math.floor(math.log10(range_val)) - 1)
    rounded_min_val = round_down_to_nearest(min_val, interval)
    rounded_max_val = round_up_to_nearest(max_val, interval)

    # Adjust interval if necessary to avoid too many/few labels
    while (rounded_max_val - rounded_min_val) / interval > 10:
        interval *= 2
    while (rounded_max_val - rounded_min_val) / interval < 4:
        interval /= 2

    # Create custom labels
    labels = np.arange(rounded_min_val, rounded_max_val + interval, interval)

    # Convert the custom labels to a vtkDoubleArray
    vtk_labels = vtk.vtkDoubleArray()
    vtk_labels.SetNumberOfValues(len(labels))
    for i, label in enumerate(labels):
        vtk_labels.SetValue(i, label)

    # Set the custom labels
    plotter.scalar_bar.SetCustomLabels(vtk_labels)

    # Enable the use of custom labels
    plotter.scalar_bar.SetUseCustomLabels(True)

    plotter.add_text(plot_title, position='upper_edge', font_size=12, color=text_color)
    plotter.background_color = background_colour

    # Calculate the sampling interval
    num_frames = len(plotted_variable_array[0])
    sampling_interval = max(1, num_frames // animation_frames)

    if save_animation:
        plotter.open_gif(save_animation_name)
        for _ in range(animation_frames):
            update(plotter, None, state)
            plotter.write_frame()
        plotter.close()
    else:
        end_time = time.time()
        print(f'Animation took {end_time - start_time:.2f} seconds to run.')
        plotter.show()#interactive_update=True)

    plotter.close()
    pv.close_all()
    plt.close('all')
