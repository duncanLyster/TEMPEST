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

NEXT STEPS:
- Import body parameters from a separate text file so it doesn't need to be done manually with every update
- Include Bea's code for converting rotation axis coordinates to a unit vector
- Implement secondary radiation/self-heating
- Implement sublimation energy loss
- Print thermal intertia to screen for user to check
- Ensure colour scale is consistent across frames
- Build in mesh converstion for binary .STL and .OBJ files
- Come up with a way of representing output data for many rotation axes and periods for mission planning | Do this and provide recommendations to MIRMIS team
- Create web interface for ease of use?
- Integrate with JPL Horizons ephemeris to get real-time insolation data

KNOWN BUGS:
1) Calculation of initial temperature is not correct - sets model up for non-realistic stable state later. 

OPEN QUESTIONS: 
Do we consider partial shadow? | Currently no - just use smaller facets 
Why are initial temperatures not normally distributed? 

EXTENSIONS: 
Binaries: Complex shading from non-rigid geometry (Could be a paper) 
Add temporary local heat sources e.g. jets
Horizontal conduction at high resolution

IMPORTANT CONSIDERATIONS: 
Generalising the model so it can be used e.g for asteroids, Enceladus fractures, adjacent emitting bodies (e.g. binaries, Saturn) 

Started: 15 Feb 2024

Author: Duncan Lyster
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import time
from visualise_shape_model import visualise_shape_model
from animate_temperature_distribution import animate_temperature_distribution
from nice_gif import nice_gif
from matplotlib import colormaps
from tqdm import tqdm
from numba import jit

# Define global variables
# Material properties (currently placeholders)
emmisivity = 0.5                                    # Dimensionless
albedo = 0.5                                        # Dimensionless
thermal_conductivity = 1.0                          # W/mK 
density = 500.0                                     # kg/m^3
specific_heat_capacity = 1000.0                     # J/kgK BUG model crashes with very low numbers 
beaming_factor = 1.0                                # Dimensionless

# NOTE: Print thermal inertia to screen for user to check

# Model setup parameters
layer_thickness = 0.1                               # m (this may be calculated properly from insolation curve later, but just a value for now)
n_layers = 10                                       # Number of layers in the conduction model
solar_distance_au = 1.0                             # AU
solar_distance = solar_distance_au * 1.496e11       # m
solar_luminosity = 3.828e26                         # W
sunlight_direction = np.array([1, 0, 0])            # Unit vector pointing from the sun to the 
timesteps_per_day = 100                             # Number of time steps per day
delta_t = 86400 / timesteps_per_day                 # s (1 day in seconds)
rotation_period = 100000                            # s
max_days = 10                                       # Maximum number of days to run the model for NOTE - this is not intended to be the final model run time as this will be determined by convergence. Just a safety limit.
rotation_axis = np.array([0, 0.1, 0.9])             # Unit vector pointing along the rotation axis
body_orientation = np.array([1, 0, 1])              # Unit vector pointing along the body's orientation
convergence_target = 1                              # K

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
def read_shape_model(filename):
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
    
    print(f"Read {len(shape_model)} facets from the shape model.\n")
    
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
    for i, subject_facet in tqdm(enumerate(shape_model), total=len(shape_model), desc='Processing visible facets'):
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
    This function implements the Möller–Trumbore intersection algorithm to determine whether a triangle intersects a line. It returns True if the triangle intersects the line, and False if it does not.'''

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

def calculate_insolation(shape_model):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.
    '''

    # Calculate rotation matrix for the body's rotation
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

    # Calculate the zenith angle (angle between the sun and the normal vector of the facet) for each facet at every timestep for one full rotation of the body 
    for facet in tqdm(shape_model, desc='Calculating insolation for each facet'):
        for t in range(timesteps_per_day):
            # Normal vector of the facet at time t=0
            normal = facet.normal
            rotation_matrix = calculate_rotation_matrix(rotation_axis, (2 * np.pi / timesteps_per_day) * t)

            new_normal = np.dot(rotation_matrix, normal)

            # Calculate the rotated sunlight direction from the body's fixed reference frame (NB rotation matrix is transposed as this is for rotation of the sunlight direction not the body)
            rotated_sunlight_direction = np.dot(rotation_matrix.T, sunlight_direction)

            # Normalize the rotated sunlight direction to ensure it is a unit vector
            rotated_sunlight_direction /= np.linalg.norm(rotated_sunlight_direction)

            # Calculate zenith angle
            zenith_angle = np.arccos(np.dot(sunlight_direction, new_normal) / (np.linalg.norm(sunlight_direction) * np.linalg.norm(new_normal)))

            #  Elimate angles where the sun is below the horizon
            if zenith_angle > np.pi / 2:
                insolation = 0

            else:
                # Calculate illumination factor, pass facet position, sunlight direction, shape model and rotation information (rotation axis, rotation period, timesteps per day, delta t, t)
                illumination_factor = calculate_shadowing(facet.position, rotated_sunlight_direction, shape_model, facet.visible_facets)

                # Calculate insolation converting AU to m
                insolation = solar_luminosity * (1 - albedo) * illumination_factor * np.cos(zenith_angle) / (4 * np.pi * solar_distance**2) 
                
            # Write the insolation value to the insolation array for this facet at time t
            facet.insolation[t] = insolation

    print(f"Calculated insolation for each facet.\n")

    # Plot the insolation curve for a single facet with number of days on the x-axis
    #plt.plot(shape_model[0]['insolation'])
    #plt.xlabel('Number of timesteps')
    #plt.ylabel('Insolation (W/m^2)')
    #plt.title('Insolation curve for a single facet for one full rotation of the body')
    #plt.show()

    return shape_model

def calculate_initial_temperatures(shape_model):
    ''' 
    This function calculates the initial temperature of each facet and sub-surface layer of the body based on the insolation curve for that facet. It writes the initial temperatures to the data cube.

    Additionally, it plots a histogram of the initial temperatures for all facets.
    '''

    # Calculate initial temperature for each facet
    for facet in shape_model:
        # Calculate the initial temperature based on the integrated insolation curve
        # Integrate the insolation curve to get the total energy received by the facet over one full rotation
        power_in = np.trapz(facet.insolation, dx=delta_t)/rotation_period
        # Calculate the temperature of the facet using the Stefan-Boltzmann law and set the initial temperature of all layers to the same value
        for layer in range(n_layers):
            facet.temperature[0][layer] = (power_in / (emmisivity * 5.67e-8))**(1/4)

    print(f"Calculated initial temperatures for each facet.\n")

    # Plot a histogram of the initial temperatures for all facets
    initial_temperatures = [facet.temperature[0][0] for facet in shape_model]
    #print initial temperatures for debugging
    print(f"")
    plt.hist(initial_temperatures, bins=20)
    plt.xlabel('Initial temperature (K)')
    plt.ylabel('Number of facets')
    plt.title('Initial temperature distribution of all facets')
    plt.show()

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

def main():
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.
    '''

    # Get the shape model and setup data storage arrays
    path_to_filename = "shape_models/500m_ico_sphere_1280_facets.stl"
    shape_model = read_shape_model(path_to_filename)

    # Visualise the shape model
    visualise_shape_model(path_to_filename, rotation_axis, rotation_period, solar_distance_au, sunlight_direction, timesteps_per_day)

    start_time = time.time()  # Start timing

    # Setup the model
    shape_model = calculate_visible_facets(shape_model)
    shape_model = calculate_insolation(shape_model)
    shape_model = calculate_initial_temperatures(shape_model)
    shape_model = calculate_secondary_radiation_coefficients(shape_model)
    
    convergence_factor = 10 # Set to a value greater than 1 to start the iteration
    day = 0 

    # Proceed to iterate the model until it converges
    while day < max_days and convergence_factor > 1:
        for time_step in range(timesteps_per_day):
            for facet in shape_model:
                current_step = int(time_step + (day * timesteps_per_day))
                # Calculate insolation term, bearing in mind that the insolation curve is constant for each facet and repeats every rotation period
                insolation_term = facet.insolation[time_step] * delta_t / (layer_thickness * density * specific_heat_capacity)

                # Calculate re-emitted radiation term
                re_emitted_radiation_term = emmisivity * beaming_factor * 5.67e-8 * (facet.temperature[current_step][0]**4) * delta_t / (layer_thickness * density * specific_heat_capacity)

                # Calculate secondary radiation term (identify facets above horizon first, then check if they face, same process for shadows but maybe segment facet into shadow/light with a calculated line?)

                secondary_radiation_term = calculate_secondary_radiation_term(shape_model, facet, delta_t) #NOTE calculate secondary radiation coefficients first

                # Calculate conducted heat term
                conducted_heat_term = thermal_conductivity * (facet.temperature[current_step][1] - facet.temperature[current_step][0]) * delta_t / (layer_thickness * density * specific_heat_capacity)

                # Calculate sublimation energy loss term

                # Calculate the new temperature of the surface layer (currently very simplified)
                facet.temperature[current_step + 1][0] = facet.temperature[current_step][0] + insolation_term - re_emitted_radiation_term + conducted_heat_term

                # Calculate the new temperatures of the subsurface layers, ensuring that the temperature of the deepest layer is fixed at its current value
                for layer in range(1, n_layers - 1):
                    facet.temperature[current_step + 1][layer] = facet.temperature[current_step][layer] + thermal_conductivity * (facet.temperature[current_step][layer + 1] - 2 * facet.temperature[current_step][layer] + facet.temperature[current_step][layer - 1]) * delta_t / (layer_thickness**2 * density * specific_heat_capacity)

        # Calculate convergence factor (average temperature error at surface across all facets divided by convergence target)
        day += 1

        temperature_error = 0
        for facet in shape_model:
                temperature_error += abs(facet.temperature[day * timesteps_per_day][0] - facet.temperature[(day - 1) * timesteps_per_day][0])

        convergence_factor = (temperature_error / (len(shape_model))) / convergence_target

        print(f"Day {day} temperature error: {temperature_error / (len(shape_model))} K\n")

    # Decrement the day counter
    day -= 1    
    
    # Post-loop check to display appropriate message
    if convergence_factor <= 1:
        print(f"Convergence target achieved after {day} days.\n\nFinal temperature error: {temperature_error / (len(shape_model))} K\n")

        # Create an array of temperatures at each timestep in final day for each facet
        # Initialise the array
        final_day_temperatures = np.zeros((len(shape_model), timesteps_per_day))

        # Fill the array
        for i, facet in enumerate(shape_model):
            for t in range(timesteps_per_day):
                final_day_temperatures[i][t] = facet.temperature[day * timesteps_per_day + t][0]

        end_time = time.time()  # End timing
        execution_time = end_time - start_time  # Calculate the execution time
        print(f"Execution time: {execution_time} seconds")

        # Plot the final day's temperature distribution for all facets
        plt.plot(final_day_temperatures.T)
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for all facets')
        plt.show()
        
        # Visualise the results - animation of final day's temperature distribution
        animate_temperature_distribution(path_to_filename, final_day_temperatures, rotation_axis, rotation_period, solar_distance_au, sunlight_direction, timesteps_per_day, delta_t)

        # Visualise the results - animation of final day's temperature distribution
        nice_gif(path_to_filename, final_day_temperatures, rotation_axis, sunlight_direction, timesteps_per_day)

        # Save a sample of the final day's temperature distribution to a file
        np.savetxt('test_data/final_day_temperatures.csv', final_day_temperatures, delimiter=',')
    
    else:
        print(f"Maximum days reached without achieving convergence. \n\nFinal temperature error: {temperature_error / (len(shape_model))} K\n")

    # Save the final day temperatures to a file that can be used with ephemeris to produce instrument simulations NOTE: This may be a large file that takes a long time to run for large shape models
    #np.savetxt('outputs/final_day_temperatures.csv', final_day_temperatures, delimiter=',')

# Call the main program to start execution
if __name__ == "__main__":
    main()