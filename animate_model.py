'''
Script to animate model quantities on the shape. The model is rotated about a specified axis and the quantity is plotted on the shape. You can move the camera around the shape and pause the animation with the arrow keys and spacebar. Right click on the shape to get the value of the quantity at that point.

TODO:
1) Leave animation up (or save it in an interactable mode) but allow the main code to continue running.
2) BUG - plot window doesn't close when you hit x.
3) BUG - camera movement is jumpy when using arrow keys.
4) Sort out colour scale bar values - should be more sensibly spaced.
5) Fix BUG - segmentation fault the second time this script is run.
6) Press 'C' to clear the selected cells.
7) BUG - Highlighted cells not in the right place as animation progresses. 

NOTE: This is currently very slow when called from the main script for large shape models.
'''
import pyvista as pv
import vtk
from stl import mesh
import time
import math
import matplotlib.pyplot as plt
import gc
import numpy as np

class AnimationState:
    def __init__(self):
        self.is_paused = False
        self.current_frame = 0
        self.camera_phi = math.pi / 2
        self.camera_theta = math.pi / 2
        self.camera_radius = None
        self.selected_cells = []
        self.fig, self.ax = None, None
        self.pause_time = None
        self.time_line = None
        self.cell_colors = {}
        self.timesteps_per_day = None
        self.highlight_actors = {}

def on_press(state):
    state.is_paused = not state.is_paused
    if state.is_paused:
        state.pause_time = state.current_frame / state.timesteps_per_day
    else:
        state.pause_time = None

def update_camera_position(plotter, state):
    x = state.camera_radius * math.sin(state.camera_phi) * math.cos(state.camera_theta)
    y = state.camera_radius * math.sin(state.camera_phi) * math.sin(state.camera_theta)
    z = state.camera_radius * math.cos(state.camera_phi)
    plotter.camera_position = [(x, y, z), (0, 0, 0), (0, 0, 1)]
    plotter.camera.view_angle = 30
    plotter.render()

def move_up(plotter, state):
    state.camera_phi -= math.pi / 36
    update_camera_position(plotter, state)

def move_down(plotter, state):
    state.camera_phi += math.pi / 36
    update_camera_position(plotter, state)

def move_left(plotter, state):
    state.camera_theta -= math.pi / 36
    update_camera_position(plotter, state)

def move_right(plotter, state):
    state.camera_theta += math.pi / 36
    update_camera_position(plotter, state)

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

def plot_picked_cell_over_time(state, cell_id, plotter, pv_mesh, plotted_variable_array, axis_label):
    if cell_id in state.selected_cells:
        # Deselect the cell
        state.selected_cells.remove(cell_id)
        
        # Remove the highlight for this cell
        if cell_id in state.highlight_actors:
            plotter.remove_actor(state.highlight_actors[cell_id])
            del state.highlight_actors[cell_id]

        # Remove the line from the plot
        if state.fig is not None and cell_id in state.cell_colors:
            line = state.cell_colors.pop(cell_id)
            line.remove()
            if not state.cell_colors:
                plt.close(state.fig)
                state.fig = None
                state.ax = None
    else:
        # Select the cell
        state.selected_cells.append(cell_id)
        
        # Create a new mesh for the selected cell
        cell = pv_mesh.extract_cells([cell_id])
        edges = cell.extract_feature_edges(feature_angle=0, boundary_edges=True, non_manifold_edges=False, manifold_edges=False)
        
        # Generate a color for this cell
        color = plt.cm.tab10(len(state.selected_cells) % 10)
        
        # Add the highlight for this cell
        actor = plotter.add_mesh(edges, color=color[:3], line_width=5, render_lines_as_tubes=True, opacity=1)
        state.highlight_actors[cell_id] = actor

        # Plot the cell data
        if state.fig is None or state.ax is None:
            plt.ion()
            state.fig, state.ax = plt.subplots()

        values_over_time = plotted_variable_array[cell_id, :]
        time_steps = [i / state.timesteps_per_day for i in range(len(values_over_time))]
        line, = state.ax.plot(time_steps, values_over_time, label=f'Cell {cell_id}', color=color[:3])
        state.cell_colors[cell_id] = line

        state.ax.set_xlabel('Local time (days)')
        state.ax.set_ylabel(axis_label)
        state.ax.set_title(f'Diurnal variation of {axis_label} of selected Cells')

        if state.time_line is None:
            state.time_line = state.ax.axvline(x=state.current_frame / state.timesteps_per_day, color='r', linestyle='--', label='Current Time')

        state.ax.legend()
        state.fig.canvas.draw()
        state.fig.canvas.flush_events()

    plotter.render()

def update(caller, event, state, plotter, pv_mesh, highlight_mesh, plotted_variable_array, vertices, rotation_axis, axis_label):
    if not state.is_paused:
        state.current_frame = int((state.current_frame + 1) % state.timesteps_per_day)

        if state.current_frame >= plotted_variable_array.shape[1]:
            state.current_frame = 0

        theta = (2 * math.pi / state.timesteps_per_day) * state.current_frame
        rot_mat = rotation_matrix(rotation_axis, theta)
        try:
            rotated_vertices = np.dot(vertices, np.array(rot_mat).T)
            pv_mesh.points = rotated_vertices
            
            # Rotate highlight actors using the same rotation matrix
            transform = vtk.vtkTransform()
            transform.RotateWXYZ(theta * 180 / math.pi, rotation_axis[0], rotation_axis[1], rotation_axis[2])
            for actor in state.highlight_actors.values():
                actor.SetUserTransform(transform)

        except Exception as e:
            print(f"Error updating mesh vertices: {e}")
            print(f"Debug info: current_frame={state.current_frame}, shape of plotted_variable_array={plotted_variable_array.shape}")
            return

        if state.fig is not None and state.ax is not None:
            if state.time_line is None:
                state.time_line = state.ax.axvline(x=state.current_frame / state.timesteps_per_day, color='r', linestyle='--', label='Current Time')
            state.time_line.set_xdata(state.current_frame / state.timesteps_per_day)
            state.fig.canvas.draw()
            state.fig.canvas.flush_events()

    plotter.render()


def on_pick(state, picked_mesh, plotter, pv_mesh, plotted_variable_array, axis_label):
    if picked_mesh is not None:
        cell_id = picked_mesh['vtkOriginalCellIds'][0]
        plot_picked_cell_over_time(state, cell_id, plotter, pv_mesh, plotted_variable_array, axis_label)

def animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction, 
                  timesteps_per_day, colour_map, plot_title, axis_label, animation_frames, 
                  save_animation, save_animation_name, background_colour):

    state = AnimationState()
    state.timesteps_per_day = timesteps_per_day
    state.highlight_actors = {}

    start_time = time.time()

    try:
        shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file)
    except Exception as e:
        print(f"Failed to load shape model: {e}")
        return
    
    if shape_mesh.vectors.shape[0] != plotted_variable_array.shape[0]:
        print("The plotted variable array must have the same number of rows as the number of cells in the shape model.")
        return
    
    if np.all(plotted_variable_array == plotted_variable_array[0]):
        print("All points in the plotted variable array are identical. Please check the data.")
        return
     
    vertices = shape_mesh.points.reshape(-1, 3)
    faces = [[3, 3*i, 3*i+1, 3*i+2] for i in range(shape_mesh.vectors.shape[0])]

    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh.cell_data[axis_label] = plotted_variable_array[:, 0]

    highlight_mesh = pv_mesh.extract_all_edges()
    highlight_mesh.cell_data['selected'] = np.zeros(highlight_mesh.n_cells, dtype=bool)
    highlight_mesh.cell_data['color'] = np.zeros((highlight_mesh.n_cells, 4))  # RGBA values

    text_color = 'white' if background_colour == 'black' else 'black' 
    bar_color = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

    bounding_box = pv_mesh.bounds
    max_dimension = max(bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2], bounding_box[5] - bounding_box[4])
    state.camera_radius = max_dimension * 5

    plotter = pv.Plotter()
    plotter.enable_anti_aliasing() 
    plotter.add_key_event('space', lambda: on_press(state))
    plotter.add_key_event('Up', lambda: move_up(plotter, state))
    plotter.add_key_event('Down', lambda: move_down(plotter, state))
    plotter.add_key_event('Left', lambda: move_left(plotter, state))
    plotter.add_key_event('Right', lambda: move_right(plotter, state))
    plotter.iren.initialize()

    cylinder = pv.Cylinder(center=(0, 0, 0), direction=rotation_axis, height=max_dimension, radius=max_dimension/200)
    plotter.add_mesh(cylinder, color='green')

    sunlight_start = [sunlight_direction[i] * max_dimension for i in range(3)]
    sunlight_arrow = pv.Arrow(start=sunlight_start, direction=[-d for d in sunlight_direction], scale=max_dimension * 0.3)
    plotter.add_mesh(sunlight_arrow, color='yellow')

    plotter.iren.add_observer('TimerEvent', lambda caller, event: update(caller, event, state, plotter, pv_mesh, highlight_mesh, plotted_variable_array, vertices, rotation_axis, axis_label))
    plotter.iren.create_timer(100)

    mesh_actor = plotter.add_mesh(pv_mesh, scalars=axis_label, cmap=colour_map, show_edges=False)
    state.highlight_actor = plotter.add_mesh(highlight_mesh, 
                                             scalars=highlight_mesh.cell_data['color'],
                                             rgb=True, 
                                             render_lines_as_tubes=True, 
                                             line_width=1)  

    plotter.enable_element_picking(callback=lambda picked_mesh: on_pick(state, picked_mesh, plotter, pv_mesh, plotted_variable_array, axis_label), mode='cell', show=False, show_message=False)

    text_color_rgb = (1, 1, 1) if background_colour == 'black' else (0, 0, 0)

    plotter.scalar_bar.GetLabelTextProperty().SetColor(bar_color)
    plotter.scalar_bar.SetPosition(0.2, 0.05)
    plotter.scalar_bar.GetLabelTextProperty().SetJustificationToCentered()
    plotter.scalar_bar.SetTitle(axis_label)
    plotter.scalar_bar.GetTitleTextProperty().SetColor(text_color_rgb)

    min_val = min(min(row) for row in plotted_variable_array)
    max_val = max(max(row) for row in plotted_variable_array)
    range_val = max_val - min_val

    interval = 10 ** (math.floor(math.log10(range_val)) - 1)
    rounded_min_val = round_down_to_nearest(min_val, interval)
    rounded_max_val = round_up_to_nearest(max_val, interval)

    while (rounded_max_val - rounded_min_val) / interval > 10:
        interval *= 2
    while (rounded_max_val - rounded_min_val) / interval < 4:
        interval /= 2

    labels = []
    label = rounded_min_val
    while label <= rounded_max_val:
        labels.append(label)
        label += interval

    vtk_labels = vtk.vtkDoubleArray()
    vtk_labels.SetNumberOfValues(len(labels))
    for i, label in enumerate(labels):
        vtk_labels.SetValue(i, label)

    plotter.scalar_bar.SetCustomLabels(vtk_labels)
    plotter.scalar_bar.SetUseCustomLabels(True)

    plotter.add_text(plot_title, position='upper_edge', font_size=12, color=text_color)
    plotter.add_text("Press spacebar to pause, right click to select a facet.", position='lower_edge', font_size=10, color=text_color)
    plotter.background_color = background_colour

    num_frames = len(plotted_variable_array[0])
    sampling_interval = max(1, num_frames // animation_frames)
    

    if save_animation:
        plotter.open_gif(save_animation_name)
        for _ in range(animation_frames):
            update(None, None, state, plotter, pv_mesh, highlight_mesh, plotted_variable_array, vertices, rotation_axis)
            plotter.write_frame()
        plotter.close()
    else:
        end_time = time.time()
        print(f'Animation took {end_time - start_time:.2f} seconds to run.')
        plotter.show()
        plotter.close()

    plotter.close()
    plotter.deep_clean()

    del pv_mesh
    del shape_mesh
    del plotter

    if state.fig:
        plt.close(state.fig)
    
    pv.close_all()
    plt.close('all')

    if 'vtk_labels' in locals():
        vtk_labels.RemoveAllObservers()
        del vtk_labels

    state.fig = None
    state.ax = None
    state.selected_cells = []

    gc.collect()
