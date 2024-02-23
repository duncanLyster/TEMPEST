import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from stl import mesh


def visualise_shape_model(filename, rotation_axis, rotation_period, solar_distance, sunlight_direction):
    ''' 
    This function visualises the shape model of the planetary body to allow the user to intuiutively check the setup is as intended. It shows an animation of the body rotating with a vector arrow that indicated incident sunlight from an external observers position. The rotation axis is shown as a line through the body and period is shown in the viewing window as text. 
    '''
    # Load the shape mesh from the STL file
    shape_mesh = mesh.Mesh.from_file(filename)
    
    # Create a figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the model mesh 
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(shape_mesh.vectors, facecolors='grey', linewidths=1, edgecolors='black', alpha=.25))
    
    # Auto scale to the mesh size
    scale = shape_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_aspect('equal')
    
    # Fix the view
    ax.view_init(elev=30, azim=30)

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

    # Animation function with sunlight arrow updated at each frame
    def update(num, shape_mesh, ax):
        # Clear the plot
        ax.clear()

        # Rotate the mesh
        theta = (2 * np.pi / rotation_period) * num  # Convert frame number to radians
        rot_mat = rotation_matrix(rotation_axis, theta)
        
        # Apply rotation to mesh vertices
        rotated_vertices = np.dot(shape_mesh.vectors.reshape((-1, 3)), rot_mat.T).reshape((-1, 3, 3))
        
        # Re-plot the rotated mesh, sunlight arrow, and rotation axis
        ax.add_collection3d(mplot3d.art3d.Poly3DCollection(rotated_vertices, facecolors='grey', linewidths=1, edgecolors='black', alpha=.9))
        
        # Set new limits based on the maximum range to ensure equal scaling
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        # Plot the rotation axis
        ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], [axis_start[2], axis_end[2]], 'r-', linewidth=2)

        # Calculate the arrow's starting position to point towards the center of the mesh
        # Adjust the 'shift_factor' as necessary to position the arrow outside the shape model
        shift_factor = line_length * 2
        arrow_start = shift_factor * sunlight_direction
        
        # Plot the reversed sunlight direction arrow pointing towards the center
        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2], -sunlight_direction[0], -sunlight_direction[1], -sunlight_direction[2], length=line_length, color='orange', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return
    
    # Animate
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, rotation_period, rotation_period/100), fargs=(shape_mesh, ax), blit=False)

    # Display rotation period and solar distance as text
    plt.figtext(0.05, 0.95, f'Rotation Period: {rotation_period}s, ({rotation_period/3600:.3g} hours)', fontsize=12)
    plt.figtext(0.05, 0.90, f'Solar Distance: {solar_distance} AU', fontsize=12)

    plt.show()

if __name__ == "__main__":
    filename = "67P_low_res.stl"
    
    # Call your visualization function here
    visualise_shape_model(filename, rotation_axis=np.array([0, 0, 1]), rotation_period=12*3600, solar_distance="TEST_SOLAR_DISTANCE", sunlight_direction = np.array([0, -1, 0]))
