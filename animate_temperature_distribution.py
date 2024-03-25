import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from matplotlib import colormaps
from stl import mesh


def animate_temperature_distribution(filename, temperature_array, rotation_axis, rotation_period, solar_distance_au, sunlight_direction, timesteps_per_day, delta_t):
    ''' 
    This function animates the temperature evolution of the body for the final day of the model run. It rotates the body and updates the temperature of each facet. 

    It uses the same rotation_matrix function as the visualise_shape_model function, and the same update function as the visualise_shape_model function but it updates the temperature of each facet at each frame using the temperature array from the data cube.
    '''

    # Load the shape model from the STL file
    shape_mesh = mesh.Mesh.from_file(filename)
    
    # Create a figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Auto scale to the mesh size
    scale = shape_mesh.points.flatten()
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

    # Use these limits to determine the length of the axis line
    # Here we choose the maximum range among x, y, z dimensions to define the line length
    line_length = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) * 0.5
    
    # Calculate start and end points for the axis line based on the rotation axis and line length
    axis_start = rotation_axis * -line_length*1.5
    axis_end = rotation_axis * line_length*1.5
    
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
    
    # Initialize colour map
    norm = plt.Normalize(temperature_array.min(), temperature_array.max())
    colormap = plt.cm.coolwarm  # Use plt.cm.coolwarm to ensure compatibility

    # Create a ScalarMappable object with the normalization and colormap
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])  # This line is necessary for ScalarMappable

    # Add the colour scale bar to the figure
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Temperature (K)', rotation=270, labelpad=15)

    # Animation function with sunlight arrow updated at each frame
    def update(num, shape_mesh, ax):
        # Clear the plot
        ax.clear()

        # Rotate the mesh
        theta = (2 * np.pi / timesteps_per_day) * num  # Convert frame number to radians
        rot_mat = rotation_matrix(rotation_axis, theta)
        
        # Apply rotation to mesh vertices
        rotated_vertices = np.dot(shape_mesh.vectors.reshape((-1, 3)), rot_mat.T).reshape((-1, 3, 3))

        # Get temperatures for the current frame and apply colour map
        temp_for_frame = temperature_array[:, int(num)%timesteps_per_day]
        face_colours = colormap(norm(temp_for_frame))
        
        # Re-plot the rotated mesh with updated face colours
        ax.add_collection3d(mplot3d.art3d.Poly3DCollection(rotated_vertices, facecolors=face_colours, linewidths=0, edgecolors='k', alpha=1.0))

        # Set new limits based on the maximum range to ensure equal scaling
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        # Plot the rotation axis
        ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], [axis_start[2], axis_end[2]], 'r-', linewidth=2)

        # Calculate the arrow's starting position to point towards the center of the body
        # Adjust the 'shift_factor' as necessary to position the arrow outside the body model
        shift_factor = line_length * 2
        arrow_start = shift_factor * sunlight_direction
        
        # Plot the reversed sunlight direction arrow pointing towards the center
        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2], -sunlight_direction[0], -sunlight_direction[1], -sunlight_direction[2], length=line_length, color='orange', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return
    
    # Animate
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, timesteps_per_day), fargs=(shape_mesh, ax), blit=False)

    # Display rotation period and solar distance as text
    plt.figtext(0.05, 0.95, f'Diurnal temperature evolution of body for one rotation', fontsize=14, ha='left')
    plt.figtext(0.05, 0.90, f'Period: {rotation_period}s, ({rotation_period/3600:.3g} hours)', fontsize=12)
    plt.figtext(0.05, 0.85, f'Solar Distance: {solar_distance_au} AU', fontsize=12)

    plt.show()

    # Save the animation as a .gif file in an outputs folder
    ani.save('outputs/temperature_distribution.gif', writer='pillow', fps=10)

if __name__ == "__main__":
    filename = "shape_models/67P_not_to_scale_low_res.stl"
    rotation_axis = np.array([0.3, -0.5, 1])
    rotation_period = 100000
    solar_distance_au = 1.0
    sunlight_direction = np.array([0, -1, 0])
    timesteps_per_day = 40
    delta_t = 86400/timesteps_per_day
    
    # Initilise the temperature array
    temperature_array = np.zeros((62, timesteps_per_day))

    # Fill the temperature array from saved file ('outputs/final_day_temperatures.csv'
    temperature_array = np.loadtxt('test_data/final_day_temperatures.csv', delimiter=',')

    # Call your visualization function here
    animate_temperature_distribution(filename, temperature_array, rotation_axis, rotation_period, solar_distance_au, sunlight_direction, timesteps_per_day, delta_t)