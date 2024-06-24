'''
Script to animate model quantities on the shape. The model is rotated about a specified axis and the quantity is plotted on the shape. You can move the camera around the shape and pause the animation with the arrow keys and spacebar. Right click on the shape to get the value of the quantity at that point.

TODO:
1) Leave animation up (or save it in an interactable mode) but allow the main code to continue running.
2) Fix BUG - plot window doesn't close when you hit x.
3) Fix BUG - camera movement is jumpy when using arrow keys.
4) Sort out colour scale bar values - should be more sensibly spaced.
5) Fix BUG - segmentation fault second time you run the animation. This could be due to the plotter not being closed properly.

NOTE: This is currently very slow when called from the main script for large shape models.
'''

import numpy as np
import pyvista as pv
import vtk
from stl import mesh
import time
import math
import matplotlib.pyplot as plt

# Global variables to control the animation state
is_paused = False
current_frame = 0

camera_phi = np.pi / 2  # Initial polar angle
camera_theta = np.pi / 2  # Initial azimuthal angle

selected_cells = None
fig, ax = None, None  # Global variables for the matplotlib figure and axis

def on_press():
    global is_paused
    is_paused = not is_paused

def update_camera_position(plotter, camera_radius):
    global camera_phi, camera_theta
    # Convert spherical coordinates to Cartesian coordinates
    x = camera_radius * np.sin(camera_phi) * np.cos(camera_theta)
    y = camera_radius * np.sin(camera_phi) * np.sin(camera_theta)
    z = camera_radius * np.cos(camera_phi)

    plotter.camera_position = [(x, y, z), (0, 0, 0), (0, 0, 1)]
    # plotter.reset_camera()
    plotter.camera.view_angle = 30
    plotter.render()

def move_up(plotter, camera_radius):
    global camera_phi
    camera_phi -= np.pi / 36  # Decrease polar angle
    update_camera_position(plotter, camera_radius)

def move_down(plotter, camera_radius):
    global camera_phi
    camera_phi += np.pi / 36  # Increase polar angle
    update_camera_position(plotter, camera_radius)

def move_left(plotter, camera_radius):
    global camera_theta
    camera_theta -= np.pi / 36  # Decrease azimuthal angle
    update_camera_position(plotter, camera_radius)

def move_right(plotter, camera_radius):
    global camera_theta
    camera_theta += np.pi / 36  # Increase azimuthal angle
    update_camera_position(plotter, camera_radius)

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
    global current_frame, selected_cells, fig, ax

    if fig is not None:
        plt.close(fig)

    # Reset the selected cells and matplotlib figure/axis (remove these lines if you want to keep the selected cells and plot open after closing the animation window)
    selected_cells = None
    fig, ax = None, None

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
    plotter.add_key_event('space', on_press)
    plotter.add_key_event('Up', lambda: move_up(plotter, camera_radius))
    plotter.add_key_event('Down', lambda: move_down(plotter, camera_radius))
    plotter.add_key_event('Left', lambda: move_left(plotter, camera_radius))
    plotter.add_key_event('Right', lambda: move_right(plotter, camera_radius))
    plotter.iren.initialize()

    update_camera_position(plotter, camera_radius)

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

    def update(caller, event):
        global current_frame
        if not is_paused:
            current_frame = (current_frame + sampling_interval) % timesteps_per_day

            if current_frame >= num_frames:
                current_frame = 0

            theta = (2 * np.pi / timesteps_per_day) * current_frame
            rot_mat = rotation_matrix(rotation_axis, theta)
            rotated_vertices = np.dot(vertices, rot_mat.T)

            pv_mesh.points = rotated_vertices
            pv_mesh.cell_data[axis_label] = plotted_variable_array[:, current_frame % timesteps_per_day].copy()

            plotter.render()

    plotter.iren.add_observer('TimerEvent', update)
    plotter.iren.create_timer(100)

    # # Plot the variable value of the picked cell over time NOTE: Would it be better to plot all selected cells on the same plot?
    # def plot_picked_cell_over_time(cell_id):
    #     values_over_time = plotted_variable_array[cell_id, :]
    #     time_steps = np.arange(len(values_over_time))/timesteps_per_day

    #     plt.ion()
    #     plt.figure()
    #     plt.plot(time_steps, values_over_time, label=f'Cell {cell_id} Value Over Time')
    #     plt.xlabel('Local time (days)')
    #     plt.ylabel(axis_label)
    #     plt.title(f'Diurnal variation of {axis_label} of Cell {cell_id}')
    #     plt.legend()
    #     plt.show(block=False)

    # Rewriting the above to put all selected cells on the same plot
    def plot_picked_cell_over_time(cell_id):
        global selected_cells, fig, ax
        if selected_cells is None:
            selected_cells = [cell_id]
        else:
            selected_cells.append(cell_id)

        if fig is None or ax is None:
            plt.ion()
            fig, ax = plt.subplots()

        ax.clear() 

        for cell in selected_cells:
            values_over_time = plotted_variable_array[cell, :]
            time_steps = np.arange(len(values_over_time)) / timesteps_per_day
            ax.plot(time_steps, values_over_time, label=f'Cell {cell}')

        ax.set_xlabel('Local time (days)')
        ax.set_ylabel(axis_label)
        ax.set_title(f'Diurnal variation of {axis_label} of selected Cells')
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()


    def on_pick(picked_mesh):
        if picked_mesh is not None:
            cell_id = picked_mesh['vtkOriginalCellIds'][0]
            print(f'Picked cell index: {cell_id}')
            print(f'Cell value: {pv_mesh.cell_data[axis_label][cell_id]}')

            # Plot the variable value of the picked cell over time
            plot_picked_cell_over_time(cell_id)

    # Add the mesh to the plotter
    mesh_actor = plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False)
    plotter.enable_element_picking(callback=on_pick, mode='cell', show_message=False)

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
            update(plotter, None)
            plotter.write_frame()
        plotter.close()
    else:
        end_time = time.time()
        print(f'Animation took {end_time - start_time:.2f} seconds to run.')
        plotter.show()#interactive_update=True)