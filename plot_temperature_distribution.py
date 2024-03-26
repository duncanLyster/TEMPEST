import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import colormaps
from stl import mesh
import os

def plot_temperature_distribution(shape_mesh, temperature_array, path_to_save):
    ''' 
    This function plots the temperature evolution of the body for a single timestep of the model run.
    '''
    # Check if the temperature array has the same number of elements as the number of facets in the shape model
    if len(temperature_array) != len(shape_mesh.vectors):
        print("The number of temperature values does not match the number of facets in the shape model.")
        print("The number in the temperature array is", len(temperature_array), "and the number of facets in the shape model is", len(shape_mesh.vectors))
        return
    
    # Ensure the outputs directory exists
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
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
    
    # Initialize colour map
    norm = plt.Normalize(temperature_array.min(), temperature_array.max())
    colormap = plt.cm.coolwarm  # Use plt.cm.coolwarm to ensure compatibility

    # Create a ScalarMappable object with the normalization and colormap
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])  # This line is necessary for ScalarMappable

    # Add the colour scale bar to the figure
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Temperature (K)', rotation=270, labelpad=15)

    face_colours = colormap(norm(temperature_array))
        
    # Re-plot the rotated mesh with updated face colours
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(shape_mesh.vectors, facecolors=face_colours, edgecolors='k', linewidths=0))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    # Save the plot as a PNG file
    plt.savefig(path_to_save + 'temperature_distribution.png', dpi=300)

    return


if __name__ == "__main__":
    filename = "shape_models/67P_not_to_scale_1666_facets.stl"

    # Load the shape model from the STL file
    shape_mesh = mesh.Mesh.from_file(filename)
    
    # Initilise the temperature array
    temperature_array = np.zeros(62)

    # Fill the temperature array from saved file ('outputs/final_day_temperatures.csv'
    temperature_array = np.loadtxt('test_data/temperatures.csv', delimiter=',')

    # Call your visualization function here
    plot_temperature_distribution(shape_mesh, temperature_array, 'outputs/test_outputs/')