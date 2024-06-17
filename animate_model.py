'''
Script to animate model quantities on the shape. 

TODO: 
1) Generalise so this can be used for all visualisations by passign data to be plotted on facets and colour scale to be used. 
2) Add ability to click on a facet to get information about it. 
3) Leave animation up (or save it in an interactable mode) but allow main code to continue running.
4) Add option to save as argument.
5) Add option for background colour. (Or Scene?)
6) Add option to change colourmap.

NOTE: This is currently very slow when called from the main script for large shape models.

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import matplotlib.animation as animation
import time
from stl import mesh

# Global variables to control the animation state
is_paused = False
current_frame = 0
# picked_indices = []

def onPress(event):
    global is_paused
    if event.key == ' ':
        is_paused ^= True

def animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction, timesteps_per_day, colour_map, plot_title, axis_label, animation_frames, save_animation, save_animation_name, background_colour):
    ''' 
    This function animates the evolution of the body. It uses the same rotation_matrix function as the visualise_shape_model function, and the same update function as the visualise_shape_model function but it updates the temperature of each facet at each frame using the temperature array from the data cube.
    '''
    global current_frame

    # Start timer
    start_time = time.time()

    # Load the shape model from the STL file
    shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file)

    # # Create an array of facet indices with their corresponding centroids
    # facet_centroids = np.array([np.mean(facet, axis=0) for facet in shape_mesh.vectors])

    # Create a figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Connect the key press event to toggle animation pause
    fig.canvas.mpl_connect('key_press_event', onPress)

    # Auto scale to the mesh size
    scale = shape_mesh.points.flatten('C')
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_aspect('equal')

    # Fix the view
    ax.view_init(elev=0, azim=0)

    # Get the current limits after autoscaling
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Find the maximum range
    max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
 
    # Calculate the middle points of each axis
    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    mid_z = np.mean(zlim)

    # Set new limits based on the maximum range to ensure equal scaling
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax.set_axis_off()
  
    # Initialize colour map and normalization
    norm = plt.Normalize(plotted_variable_array.min(), plotted_variable_array.max())
    colormap = plt.cm.get_cmap(colour_map)

    # Create a ScalarMappable object with the normalization and colormap
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])

    line_length = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) * 0.5

    # Add the colour scale bar to the figure
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(axis_label, rotation=270, labelpad=20)

    plt.figtext(0.05, 0.01, 'Rotate with mouse, pause/resume with spacebar.', fontsize=10, ha='left')

    # Calculate the sampling interval
    num_frames = len(plotted_variable_array[0])
    sampling_interval = num_frames // animation_frames

    def update(frame, shape_mesh, ax):
        global current_frame
        if not is_paused:
            current_frame = (current_frame + sampling_interval) % timesteps_per_day

        # Rotate the mesh
        theta = (2 * np.pi / timesteps_per_day) * current_frame
        rot_mat = rotation_matrix(rotation_axis, theta)
        rotated_vertices = np.dot(shape_mesh.vectors.reshape((-1, 3)), rot_mat.T).reshape((-1, 3, 3))

        # Get temperatures for the current frame and apply colour map
        temp_for_frame = plotted_variable_array[:, current_frame % timesteps_per_day]
        face_colours = colormap(norm(temp_for_frame))

        # for ind in picked_indices:
        #     face_colours[ind] = [1, 0, 0, 1]  # Ensure picked indices remain red

        for art in reversed(ax.collections):
            art.remove()

        poly_collection = art3d.Poly3DCollection(rotated_vertices, facecolors=face_colours, linewidths=0.5, edgecolors=face_colours, alpha=1.0) # picker=0.01)

        # Print poly_collection indices to check if the issue is with the picker NOTE: Could be an issue with z  sorting?
        # print(f'Poly collection indices: {poly_collection.get_array()}')

        ax.add_collection3d(poly_collection)

        # Plot the reversed sunlight direction arrow pointing towards the center
        shift_factor = line_length * 2
        arrow_start = shift_factor * sunlight_direction
        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                -sunlight_direction[0], -sunlight_direction[1], -sunlight_direction[2],
                length=line_length, color='orange', linewidth=2)

    # def on_pick(event):
    #     # NOTE: I worked out from the insolation values that the issue is with the picker, not the display. Ie facets that turn red are the ones that are picked. The issue is between the mouse click and returning the facet index. 
        
    #     global picked_indices
    #     if isinstance(event.artist, art3d.Poly3DCollection):
    #         picked_indices = event.ind # NOTE: The issue is here or upstream in the picker.
            
    #         print(f'Picked indices: {picked_indices}')

    #         # Return the centroid and vertex indices of the picked facet
    #         for ind in picked_indices:
    #             vertices = shape_mesh.vectors[ind]
    #             print(f'Vertices: {vertices}')
    #             picked_centroid = np.mean(vertices, axis=0)
    #             print(f'Centroid: {picked_centroid}')

    #         # Print the facet index in facet_centroids that corresponds to the picked facet's centroid coordinates
    #         for i, centroid in enumerate(facet_centroids):
    #             if np.allclose(centroid, picked_centroid):
    #                 print(f'Facet index: {i}')

    #         # Print the insolation value of the picked facets
    #         for ind in picked_indices:
    #             print(f'Insolation value: {insolation_array[ind, current_frame % timesteps_per_day]}')
                
    #         plt.draw()  # Redraw to reflect changes 
        
    # fig.canvas.mpl_connect('pick_event', on_pick)

    # Animate
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, timesteps_per_day), fargs=(shape_mesh, ax), blit=False)

    if save_animation:
        ani.save(save_animation_name, writer='pillow', fps=10) # Save the animation as a .gif file in an outputs folder

    # End timer
    end_time = time.time()

    print(f'Animation took {end_time - start_time:.2f} seconds to run.')

    plt.show()

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])