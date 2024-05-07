''' 
This Python script simulates diurnal temperature variations of a solar system body based on
a given shape model. It reads in the shape model, sets material and model properties, calculates 
insolation and temperature arrays, and iterates until the model converges. The results are saved and 
visualized.

It was built as a tool for planning the comet interceptor mission, but is intended to be 
generalised for use with asteroids, and other planetary bodies e.g. fractures on 
Enceladus' surface.

All calculation figures are in SI units, except where clearly stated otherwise.

Full documentation to be found (one day) at: https://github.com/duncanLyster/comet_nucleus_model

CURRENT TASK:
Modify so model can be run by an external script. 
Setup so it can be passed input parameters and run as a function. (if run locally these should be taken in from the JSON file still)

NEXT STEPS (scientifically important):
1) Modify animations so they have a fixed number of frames and are not dependent on the number of timesteps.
2) Speed up the model by using numba or other optimisation techniques throughout.
3) Parameter sensitivity analysis
4) Find ways to make the model more robust
    a) Calculate n_layers and layer thickness based on thermal inertia (?) - these shouldn't be input by the user
    b) Look into ML/optimisation of finite difference method to avoid instability
    c) Look into gradient descent optimisation technique
5) Write a performance report for the model
6) Remove all NOTE and TODO comments from the code

OPTIONAL NEXT STEPS (fun):
- Implement secondary radiation/self-heating
- Implement sublimation energy loss
- Ensure colour scale is consistent across frames
- Build in mesh converstion for binary .STL and .OBJ files
- Create web interface for ease of use?
- Integrate with JPL Horizons ephemeris to get real-time insolation data
- Come up with a way of representing output data for many rotation axes and periods for mission planning | Do this and provide recommendations to MIRMIS team

KNOWN BUGS:
1) Skin depths and layer thicknesses are surprisingly high. Check these calculations.

EXTENSIONS: 
Binaries: Complex shading from non-rigid geometry (Could be a paper) 
Add temporary local heat sources e.g. jets
Horizontal conduction at high resolution

Started: 15 Feb 2024

Author: Duncan Lyster
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys
import json
from visualise_shape_model import visualise_shape_model
from animate_temperature_distribution import animate_temperature_distribution
from plot_temperature_distribution import plot_temperature_distribution
from nice_gif import nice_gif
from animate_shadowing import animate_shadowing
from numba import jit
from stl import mesh
from tqdm import tqdm

class Simulation:
    def __init__(self, config_path):
        self.load_configuration(config_path)
    
    def load_configuration(self, config_path):
        with open(config_path, "r") as file:
            config = json.load(file)
        
        # Assign configuration to attributes, converting lists to numpy arrays as needed
        for key, value in config.items():
            if isinstance(value, list):  # Convert lists to numpy arrays
                value = np.array(value)
            setattr(self, key, value)
        
        # Initialization calculations based on the loaded parameters
        self.solar_distance = self.solar_distance_au * 1.496e11  # Convert AU to meters
        self.rotation_period_s = self.rotation_period_hours * 3600  # Convert hours to seconds
        self.angular_velocity = (2 * np.pi) / self.rotation_period_s  # Calculate angular velocity
        self.skin_depth = (self.thermal_conductivity / (self.density * self.specific_heat_capacity * self.angular_velocity))**0.5
        self.thermal_inertia = (self.density * self.specific_heat_capacity * self.thermal_conductivity)**0.5
        self.layer_thickness = 8 * self.skin_depth / self.n_layers
        self.timesteps_per_day = int(round(self.layer_thickness**2 * self.density * self.specific_heat_capacity / (2 * self.thermal_conductivity))) 
        self.delta_t = self.rotation_period_s / self.timesteps_per_day
        self.X = self.layer_thickness / self.skin_depth
        
        # Compute unit vector from ra and dec
        ra_radians = np.radians(self.ra_degrees)
        dec_radians = np.radians(self.dec_degrees)
        self.rotation_axis = np.array([np.cos(ra_radians) * np.cos(dec_radians), np.sin(ra_radians) * np.cos(dec_radians), np.sin(dec_radians)])

class Facet:
    def __init__(self, normal, vertices, timesteps_per_day, max_days, n_layers):
        self.normal = normal
        self.vertices = vertices
        self.area = self.calculate_area(vertices)
        self.position = np.mean(vertices, axis=0)
        self.insolation = np.zeros(timesteps_per_day)
        # Initialize without knowing the shape model length
        self.visible_facets = None
        self.secondary_radiation_coefficients = None
        self.temperature = np.zeros((timesteps_per_day * (max_days + 1), n_layers))

    def set_dynamic_arrays(self, length):
        self.visible_facets = np.zeros(length)
        self.secondary_radiation_coefficients = np.zeros(length)    

    @staticmethod
    def calculate_area(vertices):
        # Implement area calculation based on vertices
        v0, v1, v2 = vertices
        return np.linalg.norm(np.cross(v1-v0, v2-v0)) / 2

# Define any necessary functions
def read_shape_model(filename, timesteps_per_day, n_layers, max_days):
    ''' 
    This function reads in the shape model of the body from a .stl file and return an array of facets, each with its own area, position, and normal vector.

    Ensure that the .stl file is saved in ASCII format, and that the file is in the same directory as this script. Additionally, ensure that the model dimensions are in meters and that the normal vectors are pointing outwards from the body. An easy way to convert the file is to open it in Blender and export it as an ASCII .stl file.

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

    shape_model = []
    for i in range(len(lines)):
        if lines[i].strip().startswith('facet normal'):
            normal = np.array([float(n) for n in lines[i].strip().split()[2:]])
            vertex1 = np.array([float(v) for v in lines[i+2].strip().split()[1:]])
            vertex2 = np.array([float(v) for v in lines[i+3].strip().split()[1:]])
            vertex3 = np.array([float(v) for v in lines[i+4].strip().split()[1:]])
            facet = Facet(normal, [vertex1, vertex2, vertex3], timesteps_per_day, max_days, n_layers)
            shape_model.append(facet)

    for facet in shape_model:
        facet.set_dynamic_arrays(len(shape_model))
    
    return shape_model

def calculate_area(v1, v2, v3):
    '''Calculate the area of the triangle formed by vertices v1, v2, and v3.'''
    u = v2 - v1
    v = v3 - v1
    return np.linalg.norm(np.cross(u, v)) / 2

def calculate_visible_facets(shape_model):
    ''' 
    This function calculates the visible (test) facets from each subject facet. It calculates the angle between the normal vector of each facet and the line of sight to every other facet. It writes the indices of the visible facets to the data cube.
    
    Limitations:
        1) Doesn't account for partial shadowing (e.g. a facet may be only partially covered by the shadow cast by another facet) - more of an issue for low facet count models. 
        2) Shadowing is not calculated for secondary radiation ie if three or more facets are in a line, the third facet will not be shadowed by the second from radiation emitted by the first.
    '''
    for i, subject_facet in enumerate(shape_model):
        visible_facets = []  # Initialize an empty list for storing indices of visible facets
        for j, test_facet in enumerate(shape_model):
            if i == j:
                continue  # Skip the subject facet itself
            # Calculate whether the center of the test facet is above the plane of the subject facet
            if np.dot(subject_facet.normal, test_facet.position - subject_facet.position) > 0:
                # Calculate whether the test facet faces towards the subject facet
                if np.dot(subject_facet.normal, test_facet.normal) > 0:
                    # Add the index of the visible facet to the list
                    visible_facets.append(j)
        # Store the list of visible facet indices in the subject facet
        subject_facet.visible_facets = visible_facets

    # NOTE TO DO: Check if there are visible facets shadowed by other visible facets and remove them from the list

    return shape_model

@jit(nopython=True)
def does_triangle_intersect_line(line_start, line_direction, triangle_vertices):
    '''
    This function implements the Möller–Trumbore intersection algorithm to determine whether a triangle intersects a line. It returns True if the triangle intersects the line, and False if it does not.
    THESIS: Reference https://dl.acm.org/doi/abs/10.1145/1198555.1198746 
    '''

    # Check inputs
    if len(line_start) != 3 or len(line_direction) != 3:
        raise ValueError("The line start and direction must be 3D vectors.")
    if len(triangle_vertices) != 3:
        raise ValueError("The triangle must have three vertices.")
    # Check if line_direction is a unit vector
    if not np.isclose(np.linalg.norm(line_direction), 1, atol=1e-8):
        raise ValueError("The line direction must be a unit vector.")

    epsilon = 1e-6 # A small number to avoid division by zero
    vertex0, vertex1, vertex2 = triangle_vertices
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(line_direction, edge2)
    a = np.dot(edge1, h)
    if a > -epsilon and a < epsilon:
        return False # No intersection
    f = 1 / a
    s = line_start - vertex0
    u = f * np.dot(s, h)
    if u < 0 or u > 1:
        return False # No intersection
    q = np.cross(s, edge1)
    v = f * np.dot(line_direction, q)
    if v < 0 or u + v > 1:
        return False # No intersection
    t = f * np.dot(edge2, q)
    if t > epsilon:
        return True # Intersection
    else:
        return False # No intersection

def calculate_shadowing(facet_position, sunlight_direction, shape_model, facet_indices):
    '''
    This function calculates whether a facet is in shadow at a given time step. It cycles through all visible facets and passes their vertices to does_triangle_intersect_line which determines whether they fall on the sunlight direction vector (starting at the facet position). If they do, the facet is in shadow. 
    
    It returns the illumination factor for the facet at that time step. 0 if the facet is in shadow, 1 if it is not. 
    '''

    for facet_index in facet_indices:
        facet = shape_model[facet_index] # Get the visible facet

        vertices_np = np.array(facet.vertices)

        if does_triangle_intersect_line(facet_position, sunlight_direction, vertices_np):
            return 0 # The facet is in shadow
        
    return 1 # The facet is not in shadow

def calculate_insolation(shape_model, simulation):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.
    '''

    # Calculate rotation matrix for the body's rotation
    @jit(nopython=True)
    def calculate_rotation_matrix(axis, theta):
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
    
    # Initialize insolation array with zeros for all facets and timesteps
    insolation_array = np.zeros((len(shape_model), simulation.timesteps_per_day))

    # Calculate the zenith angle (angle between the sun and the normal vector of the facet) for each facet at every timestep for one full rotation of the body 
    for facet in tqdm(shape_model, desc="Calculating insolation"):
        for t in range(simulation.timesteps_per_day):
            # Normal vector of the facet at time t=0
            normal = facet.normal
            rotation_matrix = calculate_rotation_matrix(simulation.rotation_axis, (2 * np.pi / simulation.timesteps_per_day) * t)

            new_normal = np.dot(rotation_matrix, normal)

            # Calculate the rotated sunlight direction from the body's fixed reference frame (NB rotation matrix is transposed as this is for rotation of the sunlight direction not the body)
            rotated_sunlight_direction = np.dot(rotation_matrix.T, simulation.sunlight_direction)

            # Normalize the rotated sunlight direction to ensure it is a unit vector
            rotated_sunlight_direction /= np.linalg.norm(rotated_sunlight_direction)

            # Calculate zenith angle
            zenith_angle = np.arccos(np.dot(simulation.sunlight_direction, new_normal) / (np.linalg.norm(simulation.sunlight_direction) * np.linalg.norm(new_normal)))

            #  Elimate angles where the sun is below the horizon
            if zenith_angle > np.pi / 2:
                insolation = 0

            else:
                # Calculate illumination factor, pass facet position, sunlight direction, shape model and rotation information (rotation axis, rotation period, timesteps per day, delta t, t)
                illumination_factor = calculate_shadowing(facet.position, rotated_sunlight_direction, shape_model, facet.visible_facets)

                # Calculate insolation converting AU to m
                insolation = simulation.solar_luminosity * (1 - simulation.albedo) * illumination_factor * np.cos(zenith_angle) / (4 * np.pi * simulation.solar_distance**2) 
                
            # Write the insolation value to the insolation array for this facet at time t
            facet.insolation[t] = insolation

            insolation_array[shape_model.index(facet)][t] = insolation

    return shape_model, insolation_array

def calculate_initial_temperatures(shape_model, n_layers, emissivity, deep_temperature):
    ''' 
    This function calculates the initial temperature of each facet and sub-surface layer of the body based on the insolation curve for that facet. It writes the initial temperatures to the data cube.
    '''

    # Calculate initial temperature for each facet
    for facet in shape_model:
        # Calculate the initial temperature based on average power in
        power_in = np.mean(facet.insolation)
        # Calculate the temperature of the facet using the Stefan-Boltzmann law
        calculated_temp = (power_in / (emissivity * 5.67e-8))**(1/4)

        # Check if calculated temperature is below the deep_temperature
        if calculated_temp < deep_temperature:
            # Create a linear gradient from calculated_temp at the surface to deep_temperature at the deepest layer
            temperatures = np.linspace(calculated_temp, deep_temperature, n_layers)
        else:
            # Set all layers to the calculated temperature if it's not below deep_temperature
            temperatures = np.full(n_layers, calculated_temp)

        # Assign the calculated temperatures to the facet's temperature array
        facet.temperature[0] = temperatures

    return shape_model

def calculate_secondary_radiation_coefficients(shape_model):
    ''' 
    This function calculates the secondary radiation coefficients for each facet. It only considers the visible facets (as calculated in calculate_visible_facets) from each subject facet. It calculates the angle between the normal vector of the test facet and the line of sight to the subject facet. It then calculates the geometric coefficient of secondary radiation and writes the index and coefficient to the data cube.
    '''

    return shape_model

def calculate_secondary_radiation_term(shape_model, facet, delta_t):
    ''' 
    This function calculates the secondary radiation received by the facet in question from all visible facets. It calculates the geometric coefficient of secondary radiation from each facet and writes the index and coefficient to the data cube.

    Issues:
    - Doesn't account for shadowing of secondary raditation by a third facet (e.g. if three facets are in a line, the third will not be shadowed by the second from radiation emitted by the first).
    
    NOTE: I will need to calculate the secondary radiation coefficients for each facet before this function can be completed.
    '''

    # Calculate the secondary radiation term
    secondary_radiation_term = 0

    return secondary_radiation_term

def export_results(shape_model_name, path_to_setup_file, path_to_shape_model_file, temperature_array):
    ''' 
    This function exports the final results of the model to be used in an instrument simulator. It creates a folder within /outputs with the shape model, model parameters, a plot of the temperature distribution, and final timestep temperatures.
    '''

    folder_name = f"{shape_model_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}" # Create new folder name
    os.makedirs(f"outputs/{folder_name}") # Create folder for results
    shape_mesh = mesh.Mesh.from_file(path_to_shape_model_file) # Load shape model
    os.system(f"cp {path_to_shape_model_file} outputs/{folder_name}") # Copy shape model .stl file to folder
    os.system(f"cp {path_to_setup_file} outputs/{folder_name}") # Copy model parameters .json file to folder
    np.savetxt(f"outputs/{folder_name}/temperatures.csv", temperature_array, delimiter=',') # Save the final timestep temperatures to .csv file

    # Plot the temperature distribution for the final timestep and save it to the folder
    temp_output_file_path = f"outputs/{folder_name}/"
    plot_temperature_distribution(shape_mesh, temperature_array, temp_output_file_path)

def thermophysical_body_model(shape_model, simulation):
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.

    '''

    mean_temperature_error = simulation.convergence_target + 1 # Set to convergence target to start the loop
    day = 0 
    temperature_error = 0

    # Proceed to iterate the model until it converges
    while day < simulation.max_days and mean_temperature_error > simulation.convergence_target:
        for time_step in range(simulation.timesteps_per_day):
            for facet in shape_model:
                current_step = int(time_step + (day * simulation.timesteps_per_day))

                # Check for nan, inf or negative temperature
                if np.isnan(facet.temperature[current_step][0]) or np.isinf(facet.temperature[current_step][0]) or facet.temperature[current_step][0] < 0:
                    print(f"Ending run at timestep {current_step} due to facet {shape_model.index(facet)} having a temperature of {facet.temperature[current_step][0]} K.\n Try increasing the number of time steps per day")
                    sys.exit()  # Terminate the script immediately

                # Calculate insolation term, bearing in mind that the insolation curve is constant for each facet and repeats every rotation period
                insolation_term = facet.insolation[time_step] * simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)

                # Calculate re-emitted radiation term
                re_emitted_radiation_term = simulation.emissivity * simulation.beaming_factor * 5.67e-8 * (facet.temperature[current_step][0]**4) * simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)

                # Calculate secondary radiation term TODO: Implement this
                # secondary_radiation_term = calculate_secondary_radiation_term(shape_model, facet, delta_t) #NOTE calculate secondary radiation coefficients first

                # Calculate sublimation energy loss term TODO: Implement this

                # Calculate conducted heat term TODO: Update as per Muller (2007)
                conducted_heat_term = 2 * (simulation.delta_t / (simulation.X**2 * simulation.rotation_period_s)) * (facet.temperature[current_step][1] - facet.temperature[current_step][0])

                # Calculate the new temperature of the surface layer (currently very simplified)
                facet.temperature[current_step + 1][0] = facet.temperature[current_step][0] + insolation_term - re_emitted_radiation_term + conducted_heat_term

                # Calculate the new temperatures of the subsurface layers, ensuring that the temperature of the deepest layer is fixed at its current value
                # THESIS: Courant-Friedrichs-Lewy condition. 
                # REFERENCE: Useful info in Muller (2007) sect 3.2.2 and 3.3.2

                # TODO: Implement piecewise fix if temperature adjustment is greater than temperature difference - throw a warning. 
                # TODO: Implement a check for the Courant-Friedrichs-Lewy condition
                for layer in range(1, simulation.n_layers - 1):
                    facet.temperature[current_step + 1][layer] = facet.temperature[current_step][layer] + (simulation.delta_t / (simulation.X**2 * simulation.rotation_period_s)) * (facet.temperature[current_step][layer + 1] - 2 * facet.temperature[current_step][layer] + facet.temperature[current_step][layer - 1])
                    
                # Calculate convergence factor (average temperature error at surface across all facets divided by convergence target)
            temperature_error = 0
            
            for facet in shape_model:
                    temperature_error += abs(facet.temperature[day * simulation.timesteps_per_day][0] - facet.temperature[(day - 1) * simulation.timesteps_per_day][0])

        mean_temperature_error = temperature_error / (len(shape_model))

        print(f"Day: {day} | Temperature error: {mean_temperature_error} K | Convergence target: {simulation.convergence_target} K")
        
        day += 1

    # Decrement the day counter
    day -= 1    
    
    if mean_temperature_error < simulation.convergence_target:

        # Create an array of temperatures at each timestep in final day for each facet
        # Initialise the array
        final_day_temperatures = np.zeros((len(shape_model), simulation.timesteps_per_day))

        # Fill the array
        for i, facet in enumerate(shape_model):
            for t in range(simulation.timesteps_per_day):
                final_day_temperatures[i][t] = facet.temperature[day * simulation.timesteps_per_day + t][0]

        # Create an array of final timestep temperatures
        final_timestep_temperatures = np.zeros(len(shape_model))
        for i, facet in enumerate(shape_model):
            final_timestep_temperatures[i] = facet.temperature[(day + 1) * simulation.timesteps_per_day][0]

    else:
        final_timestep_temperatures = None
        print(f"Maximum days reached without achieving convergence. \n\nFinal temperature error: {temperature_error / (len(shape_model))} K\n Try increasing max_days or decreasing convergence_target.")

        # Break the loop and return None
        sys.exit()

    return final_day_temperatures, final_timestep_temperatures, day, temperature_error

def main():
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.

    TODO: Remove all redundant code that is covered by thermophysical_body_model function.
    '''

    # Shape model name
    shape_model_name = "67P_not_to_scale_16670_facets.stl"

    # Get setup file and shape model
    path_to_shape_model_file = f"shape_models/{shape_model_name}"
    path_to_setup_file = "model_setups/generic_model_parameters.json"

    # Load setup parameters from JSON file
    simulation = Simulation(path_to_setup_file)

    print(f"Number of timesteps per day: {simulation.timesteps_per_day}\n")
    print(f"X: {simulation.X}\n")
    print(f"Layer thickness: {simulation.layer_thickness} m\n")
    print(f"Thermal inertia: {simulation.thermal_inertia} W m^-2 K^-1 s^0.5\n")
    print(f"Skin depth: {simulation.skin_depth} m\n")

    shape_model = read_shape_model(path_to_shape_model_file, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days)

    # Visualise the shape model
    print(f"Preparing shape model visualisation.\n")
    visualise_shape_model(path_to_shape_model_file, simulation.rotation_axis, simulation.rotation_period_s, simulation.solar_distance_au, simulation.sunlight_direction, simulation.timesteps_per_day)

    # Setup the model
    print(f"Calculating visible facets.\n")
    shape_model = calculate_visible_facets(shape_model)

    print(f"Calculating insolation.\n")
    shape_model, insolation_array = calculate_insolation(shape_model, simulation)

    # Visualise the shadowing across the shape model
    print(f"Preparing shadowing visualisation.\n")
    animate_shadowing(path_to_shape_model_file, insolation_array, simulation.rotation_axis, simulation.sunlight_direction, simulation.timesteps_per_day)

    # Plot the insolation curve for a single facet with number of days on the x-axis
    plt.plot(shape_model[4].insolation)
    plt.xlabel('Number of timesteps')
    plt.ylabel('Insolation (W/m^2)')
    plt.title('Insolation curve for a single facet for one full rotation of the body')
    plt.show()

    print(f"Calculating initial temperatures.\n")
    shape_model = calculate_initial_temperatures(shape_model, simulation.n_layers, simulation.emissivity, simulation.deep_temperature)

    # Plot a histogram of the initial temperatures for all facets
    initial_temperatures = [facet.temperature[0][0] for facet in shape_model]
    plt.hist(initial_temperatures, bins=20)
    plt.xlabel('Initial temperature (K)')
    plt.ylabel('Number of facets')
    plt.title('Initial temperature distribution of all facets')
    plt.show()

    print(f"Calculating secondary radiation coefficients.\n")
    shape_model = calculate_secondary_radiation_coefficients(shape_model)
    
    print(f"Running main simulation loop.\n")
    start_time = time.time()
    final_day_temperatures, final_timestep_temperatures, day, temperature_error = thermophysical_body_model(shape_model, simulation)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Convergence target achieved after {day} days.\n\nFinal temperature error: {temperature_error / (len(shape_model))} K\n")
    print(f"Execution time: {execution_time} seconds")

    # temperature_min = np.min(final_timestep_temperatures)
    # temperature_max = np.max(final_timestep_temperatures)

    # Plot the final day's temperature distribution for all facets
    plt.plot(final_day_temperatures.T)
    plt.xlabel('Timestep')
    plt.ylabel('Temperature (K)')
    plt.title('Final day temperature distribution for all facets')
    plt.show()

    # Visualise the results - animation of final day's temperature distribution
    print(f"Preparing temperature animation.\n")
    animate_temperature_distribution(path_to_shape_model_file, final_day_temperatures, simulation.rotation_axis, simulation.rotation_period_s, simulation.solar_distance_au, simulation.sunlight_direction, simulation.timesteps_per_day, simulation.delta_t)

    # # Visualise the results - animation of final day's temperature distribution
    # print(f"Preparing beautiful temperature animation.\n")
    # nice_gif(path_to_shape_model_file, final_day_temperatures, simulation.rotation_axis, simulation.sunlight_direction, simulation.timesteps_per_day)

    # Save the final day temperatures for facet 4 to a file that can be used with ephemeris to produce instrument simulation NOTE: This may be a large file that takes a long time to run for large shape models
    print(f"Saving final day temps to file.\n")
    np.savetxt('outputs/final_day_temperatures.csv', final_day_temperatures[4], delimiter=',')

    # Save to a new folder the shape model, model parameters, and final timestep temperatures (1 temp per facet)
    print(f"Saving folder with final timestep temperatures.\n")
    export_results(shape_model_name, path_to_setup_file, path_to_shape_model_file, final_day_temperatures[:, -1])

    print(f"Model run complete.\n")

# Call the main program to start execution
if __name__ == "__main__":
    main()