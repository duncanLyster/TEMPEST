'''
Script to animate model quantities on the shape. The model is rotated about a specified axis and the quantity is plotted on the shape. You can move the camera around the shape and pause the animation with the arrow keys and spacebar. Right click on the shape to get the value of the quantity at that point.

TODO:
1) Leave animation up (or save it in an interactable mode) but allow the main code to continue running.
2) Fix BUG - plot window doesn't close when you hit x.
3) Fix BUG - camera movement is jumpy when using arrow keys.
4) Sort out colour scale bar values - should be more sensibly spaced.
5) Fix BUG - segmentation fault second time you run the animation. This could be due to the plotter not being closed properly.

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
        self.picked_cells = []
        self.fig = None
        self.ax = None
        self.highlight_mesh = None

def create_highlight_mesh():
    # Create a tiny sphere at the origin
    sphere = pv.Sphere(radius=1e-6, center=(0, 0, 0))
    return sphere

def update_highlight_mesh(state, pv_mesh, cell_id, add=True):
    if state.highlight_mesh is None:
        state.highlight_mesh = create_highlight_mesh()

    cell = pv_mesh.extract_cells(cell_id)
    edges = cell.extract_all_edges()

    if add:
        state.highlight_mesh = state.highlight_mesh.merge(edges)
    else:
        # Remove the edges corresponding to this cell
        edge_points = edges.points
        existing_points = state.highlight_mesh.points
        mask = ~np.isin(existing_points, edge_points).all(axis=1)
        state.highlight_mesh = state.highlight_mesh.extract_points(mask)

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

def round_up_to_nearest(x, base):
    return base * math.ceil(x / base)

def round_down_to_nearest(x, base):
    return base * math.floor(x / base)

def update_camera_position(plotter, state, camera_radius):
    x = camera_radius * np.sin(state.camera_phi) * np.cos(state.camera_theta)
    y = camera_radius * np.sin(state.camera_phi) * np.sin(state.camera_theta)
    z = camera_radius * np.cos(state.camera_phi)

    plotter.camera_position = [(x, y, z), (0, 0, 0), (0, 0, 1)]
    plotter.camera.view_angle = 30
    plotter.render()

def move_camera(plotter, state, camera_radius, direction):
    if direction == 'up':
        state.camera_phi -= np.pi / 36
    elif direction == 'down':
        state.camera_phi += np.pi / 36
    elif direction == 'left':
        state.camera_theta -= np.pi / 36
    elif direction == 'right':
        state.camera_theta += np.pi / 36
    update_camera_position(plotter, state, camera_radius)

def plot_picked_cells_over_time(state, plotted_variable_array, timesteps_per_day, axis_label):
    if state.fig is None or state.ax is None:
        plt.ion()
        state.fig, state.ax = plt.subplots()

    state.ax.clear() 

    for cell in state.picked_cells:
        values_over_time = plotted_variable_array[cell, :]
        time_steps = np.arange(len(values_over_time)) / timesteps_per_day
        state.ax.plot(time_steps, values_over_time, label=f'Cell {cell}')

    state.ax.set_xlabel('Local time (days)')
    state.ax.set_ylabel(axis_label)
    state.ax.set_title(f'Diurnal variation of {axis_label} of selected Cells')
    state.ax.legend()
    state.fig.canvas.draw()
    state.fig.canvas.flush_events()

def setup_plotter(state, camera_radius, max_dimension, rotation_axis, sunlight_direction, text_color):
    plotter = pv.Plotter()
    plotter.add_key_event('space', lambda: setattr(state, 'is_paused', not state.is_paused))
    plotter.add_key_event('Up', lambda: move_camera(plotter, state, camera_radius, 'up'))
    plotter.add_key_event('Down', lambda: move_camera(plotter, state, camera_radius, 'down'))
    plotter.add_key_event('Left', lambda: move_camera(plotter, state, camera_radius, 'left'))
    plotter.add_key_event('Right', lambda: move_camera(plotter, state, camera_radius, 'right'))
    plotter.iren.initialize()

    update_camera_position(plotter, state, camera_radius)

    cylinder = pv.Cylinder(center=[0, 0, 0], direction=rotation_axis, height=max_dimension, radius=max_dimension/200)
    plotter.add_mesh(cylinder, color='green')

    sunlight_start = np.array([0, 0, 0]) + sunlight_direction * max_dimension
    sunlight_length = max_dimension * 0.3
    sunlight_arrow = pv.Arrow(start=sunlight_start, direction=-sunlight_direction, scale=sunlight_length)
    plotter.add_mesh(sunlight_arrow, color='yellow')

    plotter.add_text("Press spacebar to pause, right click to select a facet.", position='lower_edge', font_size=10, color=text_color)

    return plotter

def animate_model_old(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction, 
                  timesteps_per_day, colour_map, plot_title, axis_label, animation_frames, 
                  save_animation, save_animation_name, background_colour):
    state = AnimationState()

    if state.fig is not None:
        plt.close(state.fig)

    start_time = time.time()

    shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file)
    vertices = shape_mesh.points.reshape(-1, 3)
    faces = np.hstack([np.full((shape_mesh.vectors.shape[0], 1), 3), 
                       np.arange(shape_mesh.vectors.shape[0] * 3).reshape(-1, 3)])

    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh.cell_data[axis_label] = plotted_variable_array[:, 0]

    text_color = 'white' if background_colour=='black' else 'black' 
    bar_color = (1, 1, 1) if background_colour=='black' else (0, 0, 0)

    bounding_box = pv_mesh.bounds
    max_dimension = max(bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2], bounding_box[5] - bounding_box[4])
    camera_radius = max_dimension * 5

    plotter = setup_plotter(state, camera_radius, max_dimension, rotation_axis, sunlight_direction, text_color)

    state.highlight_mesh = create_highlight_mesh()
    highlight_actor = plotter.add_mesh(state.highlight_mesh, color='green', line_width=2)

    def update(caller, event):
        if not state.is_paused:
            state.current_frame = (state.current_frame + sampling_interval) % timesteps_per_day

            if state.current_frame >= num_frames:
                state.current_frame = 0

            theta = (2 * np.pi / timesteps_per_day) * state.current_frame
            rot_mat = rotation_matrix(rotation_axis, theta)
            rotated_vertices = np.dot(vertices, rot_mat.T)

            pv_mesh.points = rotated_vertices
            pv_mesh.cell_data[axis_label] = plotted_variable_array[:, state.current_frame % timesteps_per_day].copy()

            # Rotate the highlight mesh
            if state.highlight_mesh is not None:
                rotated_highlight_vertices = np.dot(state.highlight_mesh.points, rot_mat.T)
                state.highlight_mesh.points = rotated_highlight_vertices

            plotter.render()

    plotter.iren.add_observer('TimerEvent', update)
    plotter.iren.create_timer(100)

    def on_pick(picked_mesh):
        if picked_mesh is not None:
            cell_id = picked_mesh['vtkOriginalCellIds'][0]
            if cell_id in state.picked_cells:
                state.picked_cells.remove(cell_id)
                print(f'Removed cell {cell_id} from selection')
                update_highlight_mesh(state, pv_mesh, cell_id, add=False)
            else:
                state.picked_cells.append(cell_id)
                print(f'Added cell {cell_id} to selection')
                update_highlight_mesh(state, pv_mesh, cell_id, add=True)
            
            print(f'Cell value: {pv_mesh.cell_data[axis_label][cell_id]}')
            print(f'Currently selected cells: {state.picked_cells}')
            
            plot_picked_cells_over_time(state, plotted_variable_array, timesteps_per_day, axis_label)

            # Update the highlight mesh in the plotter
            plotter.remove_actor(highlight_actor)
            highlight_actor = plotter.add_mesh(state.highlight_mesh, color='green', line_width=2)


    mesh_actor = plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False)
    plotter.enable_element_picking(callback=on_pick, mode='cell', show_message=False)

    text_color_rgb = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

    plotter.scalar_bar.GetLabelTextProperty().SetColor(bar_color)
    plotter.scalar_bar.SetPosition(0.2, 0.05)
    plotter.scalar_bar.GetLabelTextProperty().SetJustificationToCentered()
    plotter.scalar_bar.SetTitle(axis_label)
    plotter.scalar_bar.GetTitleTextProperty().SetColor(text_color_rgb)

    min_val = np.min(plotted_variable_array)
    max_val = np.max(plotted_variable_array)
    range_val = max_val - min_val

    interval = 10 ** (math.floor(math.log10(range_val)) - 1)
    rounded_min_val = round_down_to_nearest(min_val, interval)
    rounded_max_val = round_up_to_nearest(max_val, interval)

    while (rounded_max_val - rounded_min_val) / interval > 10:
        interval *= 2
    while (rounded_max_val - rounded_min_val) / interval < 4:
        interval /= 2

    labels = np.arange(rounded_min_val, rounded_max_val + interval, interval)

    vtk_labels = vtk.vtkDoubleArray()
    vtk_labels.SetNumberOfValues(len(labels))
    for i, label in enumerate(labels):
        vtk_labels.SetValue(i, label)

    plotter.scalar_bar.SetCustomLabels(vtk_labels)
    plotter.scalar_bar.SetUseCustomLabels(True)

    plotter.add_text(plot_title, position='upper_edge', font_size=12, color=text_color)
    plotter.background_color = background_colour

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
        plotter.show()