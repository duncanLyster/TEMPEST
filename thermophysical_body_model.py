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
import matplotlib.pyplot as plt
from body_visualisation import visualise_shape_model

# Define global variables
# Comet-wide material properties (currently placeholders)
emmisivity = 0.5                                    # Dimensionless
albedo = 0.5                                        # Dimensionless
thermal_conductivity = 1.0                          # W/mK 
density = 1.0                                       # kg/m^3
specific_heat_capacity = 1.0                        # J/kgK
beaming_factor = 0.5                                # Dimensionless

# Model setup parameters
layer_thickness = 0.1                               # m (this may be calculated properly from insolation curve later, but just a value for now)
n_layers = 10                                       # Number of layers in the conduction model
solar_distance = 1.0                                # AU
solar_luminosity = 3.828e26                         # W
sunlight_direction = np.array([0, -1, 0])           # Unit vector pointing from the sun to the comet
n_timesteps = 40                                    # Number of time steps per day
time_step = 86400 / n_timesteps                     # s (1 day in seconds)
rotation_period = 100000                            # s (1 day on the comet)
max_days = 5                                        # Maximum number of days to run the model for NOTE - this is not intended to be the final model run time as this will be determined by convergence. Just a safety limit.
rotation_axis = np.array([0.3, -0.5, 1])            # Unit vector pointing along the rotation axis
body_orientation = np.array([0, 0, 1])              # Unit vector pointing along the body's orientation
convergence_target = 5                              # K

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
        facet['normal'] = normal
        facet['area'] = area
        facet['position'] = centroid
        #initialise insolation and secondary radiation arrays
        facet['insolation'] = np.zeros(n_timesteps) # Insolation curve doesn't change day to day
        facet['secondary_radiation'] = np.zeros(len(facets))
        #initialise temperature arrays
        facet['temperature'] = np.zeros((n_timesteps * max_days, n_layers))

    print(f"Read {len(facets)} facets from the shape model.\n")
    
    return facets

def calculate_area(v1, v2, v3):
    '''Calculate the area of the triangle formed by vertices v1, v2, and v3.'''
    u = v2 - v1
    v = v3 - v1
    return np.linalg.norm(np.cross(u, v)) / 2

def calculate_insolation(shape_model):
    ''' 
    This function calculates the insolation for each facet of the comet. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.
    '''

    # Calculate rotation matrix for the comet's rotation
    def rotation_matrix(axis, theta):
        '''Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.'''
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    # Calculate the zenith angle (angle between the sun and the normal vector of the facet) for each facet at every timestep for one full rotation of the body 
    for facet in shape_model:
        for t in range(n_timesteps):
            # Normal vector of the facet at time t=0
            normal = facet['normal']

            new_normal = np.dot(rotation_matrix(rotation_axis, (2 * np.pi * time_step / rotation_period) * t), normal)

            # Calculate zenith angle
            zenith_angle = np.arccos(np.dot(sunlight_direction, new_normal) / (np.linalg.norm(sunlight_direction) * np.linalg.norm(new_normal)))

            #  Elimate angles where the sun is below the horizon
            if zenith_angle > np.pi / 2:
                insolation = 0

            else:
                # Calculate illumination factor
                # NOTE PLACEHOLDER: This will be a bit tricky to calculate - need to consider whether other facets fall on the vector between the sun and the facet
                illumination_factor = 1

                # Calculate insolation converting AU to m
                insolation = solar_luminosity * (1 - albedo) * illumination_factor * np.cos(zenith_angle) / (4 * np.pi * (solar_distance * 1.496e+11)**2) 
                
            # Write the insolation value to the insolation array for this facet at time t
            facet['insolation'][t] = insolation

    print(f"Calculated insolation for each facet.\n")
    print(f"Facet 1: {shape_model[0]}\n")

    # Plot the insolation curve for a single facet with number of days on the x-axis
    plt.plot(shape_model[0]['insolation'])
    plt.xlabel('Number of timesteps')
    plt.ylabel('Insolation (W/m^2)')
    plt.title('Insolation curve for a single facet for one full rotation of the comet')
    plt.show()

    return shape_model

def calculate_initial_temperatures(shape_model):
    ''' 
    This function calculates the initial temperature of each facet and sub-surface layer of the comet based on the insolation curve for that facet. It writes the initial temperatures to the data cube.

    Additionally, it plots a histogram of the initial temperatures for all facets.
    '''

    # Calculate initial temperature for each facet
    for facet in shape_model:
        # Calculate the initial temperature based on the integrated insolation curve
        # Integrate the insolation curve to get the total energy received by the facet over one full rotation
        total_energy = np.trapz(facet['insolation'], dx=time_step)
        # Calculate the temperature of the facet using the Stefan-Boltzmann law and set the initial temperature of all layers to the same value
        for layer in range(n_layers):
            facet['temperature'][0][layer] = (total_energy / (emmisivity * facet['area'] * 5.67e-8))**(1/4)

    print(f"Calculated initial temperatures for each facet.\n")
    print(f"Facet 1 temperatures at t=0: {shape_model[0]['temperature'][0]}\n")
    print(f"Facet 2 temperatures at t=0: {shape_model[1]['temperature'][0]}\n")

    # Plot a histogram of the initial temperatures for all facets
    initial_temperatures = [facet['temperature'][0][0] for facet in shape_model]
    plt.hist(initial_temperatures, bins=20)
    plt.xlabel('Initial temperature (K)')
    plt.ylabel('Number of facets')
    plt.title('Initial temperature distribution of all facets')
    plt.show()

    return shape_model

def main():
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.
    '''

    # Get the shape model and setup data storage arrays
    filename = "67P_low_res.stl"
    shape_model = read_shape_model(filename)

    # Visualise the shape model
    visualise_shape_model(filename, rotation_axis, rotation_period, solar_distance, sunlight_direction)

    # Calculate insolation array for each facet
    shape_model = calculate_insolation(shape_model)

    # Calulate initial temperature array
    shape_model = calculate_initial_temperatures(shape_model)

    # Calculate secondary radiation array
        # Ray tracing to work out which facets are visible from each facet
        # Calculate the geometric coefficient of secondary radiation from each facet
        # Write the index and coefficient to the data cube

    # Proceed to iterate the model until it converges
    while (convergence_factor > 1) and (days < max_days):
        for time in range(0, rotation_period, time_step):
            for facet in shape_model:
                # Calculate insolation term

                # Calculate secondary radiation term
                # Calculate conducted heat term
                # Calculate re-emitted radiation term
                # Calculate sublimation energy loss term
                # Calculate new surface temperature
                # Calculate new temperatures for all sub-surface layers
                # Save the new temperatures to the data cube

                test = 1 # Placeholder for the above calculations
            test = 1    # Placeholder for the above calculations

        # Calculate convergence factor (average temperature error at surface across all facets divided by convergence target)
        temperature_error = 0
        for facet in shape_model:
                temperature_error += abs(facet['temperature'][0][0] - facet['temperature'][n_timesteps][0])

        convergence_factor = (temperature_error / (len(shape_model))) / convergence_target

        days += 1

    # Visualise the results - animation of final day's temperature distribution

    # Save the final day temperatures to a file that can be used with ephemeris to produce instrument simulations

# Call the main program to start execution
if __name__ == "__main__":
    main()
