''' 
This Python script simulates diurnal temperature variations of a solar system body based on
a given shape model. It reads in the shape model, sets material and model properties, calculates 
insolation and temperature arrays, and iterates until the model converges. The results are saved and 
visualized.

It was built as a tool for planning the comet interceptor mission, but is intended to be 
general enough to be used for asteroids, and other planetary bodies e.g. fractures on 
Enceladus' surface.

All calculation figures are in SI units, except where clearly stated otherwise.

Full documentation to be found (one day) at: **INSERT LINK TO GITHUB REPOSITORY**.

OPEN QUESTIONS: 
Do we consider partial shadow? 
Do we treat facets as points or full 2D polygons?
Should insolation be calculated once for each facet as a 2D map where the sun later passes over? Or just as a curve each time the model runs? 

EXTENSIONS: 
Binaries: Complex shading from non-rigid geometry (Could be a paper) 
Add temporary heat sources. 

IMPORTANT CONSIDERATIONS: 
Generalising the model so it can be used e.g for asteroids, Enceladus fractures, adjacent emitting bodies (e.g. binaries, Saturn) 

Started: 15 Feb 2024
Last updated: 15 Feb 2024

Author: Duncan Lyster
'''

import numpy as np
import os
from body_visualisation import visualise_shape_model

import sys 
print(sys.path)

# Define global variables
# Comet-wide material properties (currently placeholders)
emmisivity = 0.5                            # Dimensionless
albedo = 0.5                                # Dimensionless
thermal_conductivity = 1.0                  # W/mK 
density = 1.0                               # kg/m^3
specific_heat_capacity = 1.0                # J/kgK
beaming_factor = 0.5                        # Dimensionless

# Model setup parameters
layer_thickness = 0.1                       # m (this may be calculated properly from insolation curve later, but just a value for now)
n_layers = 10                               # Number of layers in the conduction model
solar_distance = 1.0                        # AU
solar_constant = 1361                       # W/m^2 Solar constant is the solar radiation received per unit area at 1 AU (could use luminosity of the sun and distance to the sun to calculate this)
time_step = 1000                            # s
rotation_period = 100000                    # s (1 day on the comet)
max_days = 5                                # Maximum number of days to run the model for
rotation_axis = np.array([0.3, -0.5, 1])         # Unit vector pointing along the rotation axis
body_orientation = np.array([0, 0, 1])      # Unit vector pointing along the body's orientation


# Define any necessary functions
def read_shape_model(filename):
    ''' 
    This function reads in the shape model of the comet from a .stl file and return an array of facets, each with its own area, position, and normal vector.

    Ensure that the .stl file is saved in ASCII format, and that the file is in the same directory as this script. Additionally, ensure that the model dimensions are in meters and that the normal vectors are pointing outwards from the comet. An easy way to convert the file is to open it in Blender and export it as an ASCII .stl file.

    This function will give an error if the file is not in the correct format, or if the file is not found.
    '''
    
    # Check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} was not found.")
    
    # Attempt to read the file and check for ASCII STL format
    try:
        with open(filename, 'r') as file:
            first_line = file.readline().strip()
            if not first_line.startswith('solid'):
                raise ValueError("The file is not in ASCII STL format.")
    except UnicodeDecodeError:
        raise ValueError("The file is not in ASCII STL format or is binary.")

    # Reopen the file for parsing after format check
    with open(filename, 'r') as file:
        lines = file.readlines()

    facets = []
    for i in range(len(lines)):
        if lines[i].strip().startswith('facet normal'):
            normal = np.array([float(n) for n in lines[i].strip().split()[2:]])
            vertex1 = np.array([float(v) for v in lines[i+2].strip().split()[1:]])
            vertex2 = np.array([float(v) for v in lines[i+3].strip().split()[1:]])
            vertex3 = np.array([float(v) for v in lines[i+4].strip().split()[1:]])
            facets.append({'normal': normal, 'vertices': [vertex1, vertex2, vertex3]})
    
    # Process facets to calculate area and centroid
    for facet in facets:
        v1, v2, v3 = facet['vertices']
        area = calculate_area(v1, v2, v3)
        centroid = (v1 + v2 + v3) / 3
        facet['area'] = area
        facet['position'] = centroid

    print(f"Read {len(facets)} facets from the shape model.")
    print(f"Facet 1: {facets[0]}")
    
    return facets

def calculate_area(v1, v2, v3):
    '''Calculate the area of the triangle formed by vertices v1, v2, and v3.'''
    u = v2 - v1
    v = v3 - v1
    return np.linalg.norm(np.cross(u, v)) / 2

def initialise_data_cube():
    ''' 
    This function sets up the data cube, which will store the geometry, temperatures, insolation arrays (not sure if 1D or 2D) and secondary radiation arrays for each layer of the model. It is currently a placeholder, and will be replaced with a proper data cube initialiser.


    Facet#	Area	Position	Normal vector 	T0_t0	T0_t1	T1_t0 	T1_t1 etc	Insolation curve	Secondary radiation 
    1	    Float	Coordinates	Vector (3D)	    Float	Float	Float	Float 	    1D array (or 2D?) 	2D array [index, geometric coefficient]

    QUESTIONS:
    1) Should the insolation curve be a 1D array or a 2D array?
    2) Do we save every temperature for every time step, or just one per day plus the most recent?

    '''
    data_cube = "placeholder"
    							
    return data_cube

def calculate_insolation():
    ''' 
    This function calculates the insolation for each facet of the comet. It calculates the position of the sun relative to the comet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.
    '''
    pass

def main():
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.
    '''

    # Get the shape model
    filename = "67P_low_res.stl"
    shape_model = read_shape_model(filename)

    # Visualise the shape model
    visualise_shape_model(filename, rotation_axis, rotation_period, solar_distance)

    # Setup the data cube
    data_cube = initialise_data_cube()

    # Calculate insolation array for each facet
        # Calculate the position of the sun relative to the comet
        # Calculate the insolation for each facet factoring in shadows
        # Write the insolation to the data cube

    # Calulate initial temperature array

    # Calculate secondary radiation array
        # Ray tracing to work out which facets are visible from each facet
        # Calculate the geometric coefficient of secondary radiation from each facet
        # Write the index and coefficient to the data cube
    
    # while (not converged) and (days < max_days):
        #for time in range(0, rotation_period, time_step):
            # Calculate new temperatures everywhere in the model
            #for facet in shape_model:
                # Calculate insolation
                # Calculate secondary radiation
                # Calculate conducted heat
                # Calculate re-emitted radiation
                # Calculate sublimation energy loss
                # Put the above into equation for new surface temperature
                # Calculate new temperatures for all sub-surface layers
                # Save the new temperatures to the data cube
                # Set T = T_new
            
            # Calculate convergence factor

    # Save the data cube

    # Visualise the results - animation of 1 day 

    # Output the results to a file that can be used with ephemeris to produce instrument simulations

# Call the main program to start execution
if __name__ == "__main__":
    main()
