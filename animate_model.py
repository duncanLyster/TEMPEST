'''
Script to animate model quantities on the shape.

TODO:
1) Generalise for all visualisations by passing data to be plotted on facets and colour scale to be used.
2) Add ability to click on a facet to get information about it.
3) Leave animation up (or save it in an interactable mode) but allow the main code to continue running.
4) Add option to save as an argument.
5) Add option for background colour. (Or Scene?)
6) Add option to change the colourmap.

NOTE: This is currently very slow when called from the main script for large shape models.
'''

import numpy as np
import pyvista as pv
from stl import mesh
import time
import vtk

# Global variables to control the animation state
is_paused = False
current_frame = 0

def on_press():
    global is_paused
    is_paused = not is_paused

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
    global current_frame

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

    # Create a Plotter object
    plotter = pv.Plotter()
    plotter.add_key_event('space', on_press)
    plotter.iren.initialize()

    # Add the text to the window
    plotter.add_text("Press spacebar to pause", position='lower_edge', font_size=10, color=text_color)

    # Add the mesh to the plotter
    mesh_actor = plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False)


    # Scale bar properties
    plotter.scalar_bar.GetLabelTextProperty().SetColor(bar_color)

    plotter.scalar_bar.SetPosition(0.2, 0.05)  # Set the position of the bottom left corner of the scalar bar
    plotter.scalar_bar.GetLabelTextProperty().SetJustificationToCentered()  # Center the labels

    plotter.add_text(plot_title, position='upper_edge', font_size=12, color=text_color)
    plotter.background_color = background_colour

    # Calculate the sampling interval
    num_frames = len(plotted_variable_array[0])
    sampling_interval = max(1, num_frames // animation_frames)

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

    def on_pick(mesh_actor, event):
        picked_index = event.point_id
        if picked_index >= 0:
            print(f'Picked facet index: {picked_index}')
            print(f'Facet insolation value: {pv_mesh.cell_data[axis_label][picked_index]}')

    plotter.enable_point_picking(callback=on_pick, show_point=True, show_message=False)

    if save_animation:
        plotter.open_gif(save_animation_name)
        for _ in range(animation_frames):
            update(plotter, None)
            plotter.write_frame()
        plotter.close()
    else:
        end_time = time.time()
        print(f'Animation took {end_time - start_time:.2f} seconds to run.')
        plotter.show(auto_close=False)