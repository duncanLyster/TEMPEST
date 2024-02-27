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
- Implement vector intersection calculation for secondary radiation and shadowing
- Implement secondary radiation
- Implement sublimation energy loss
- Implement shadowing
- Come up with a way of representing output data for many rotation axes and periods for mission planning

BUGS:
- The animation of the temperature distribution jumps at the end of the day
- Initial temperatures are higher for smaller facets/higher resolution shape models. 
- Possibly same as above - initial temperatures are much too high for certain models. 

OPEN QUESTIONS: 
Do we consider partial shadow? 
Do we treat facets as points or full 2D polygons?

EXTENSIONS: 
Binaries: Complex shading from non-rigid geometry (Could be a paper) 
Add temporary local heat sources e.g. jets
Horizontal conduction at high resolution

IMPORTANT CONSIDERATIONS: 
Generalising the model so it can be used e.g for asteroids, Enceladus fractures, icy moons like Europa, adjacent emitting bodies (e.g. binaries, Saturn) 

Started: 15 Feb 2024

Author: Duncan Lyster
'''

'''
Below is the thermal model representation for the asteroid Bennu. The parameters are actual values obtained from literature unless stated. 

This is a test to see how the asteroid is different to the comet thermal model as intended therefore mechanical properties have not been added.

Author: Bea Chikani
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from visualise_shape_model import visualise_shape_model
from animate_temperature_distribution import animate_temperature_distribution
from matplotlib import colormaps
import math

# Define global variables
# Material properties for Bennu
emmisivity = 0.9                                    # Dimensionless
albedo = 0.044  #average taken                     # Dimensionless
thermal_conductivity = 0.5 #estimate but from literature      # W/mK 
density = 1190                                     # kg/m^3
specific_heat_capacity = 755 #At 300 K             # J/kgK
beaming_factor = 0 #For comets only                               # Dimensionless

# Model setup parameters
layer_thickness = 0.1                               # m (this may be calculated properly from insolation curve later, but just a value for now)
n_layers = 10                                       # Number of layers in the conduction model
solar_distance_au = 1.13           #average taken   # AU
solar_distance = solar_distance_au * 1.496e11       # m
solar_luminosity = 3.828e26                         # W
sunlight_direction = np.array([1, 0, 0])            # Unit vector pointing from the sun to the 
timesteps_per_day = 100                             # Number of time steps per day
delta_t = 86400 / timesteps_per_day                 # s (1 day in seconds)
rotation_period = 15480                            # s
max_days = 20                                       # Maximum number of days to run the model for NOTE - this is not intended to be the final model run time as this will be determined by convergence. Just a safety limit.

#Rotation axis
#Convert degrees to radians
ra_degrees = 85.65
dec_degrees = -60.17
ra_radians = math.radians(ra_degrees)
dec_radians = math.radians(dec_degrees)

#Compute unit vector components
x = math.cos(dec_radians) * math.cos(ra_radians)
y = math.cos(dec_radians) * math.sin(ra_radians)
z = math.sin(dec_radians)

#Normalise the vector
magnitude = math.sqrt(x**2 + y**2 + z**2)
x_normalized = x / magnitude
y_normalized = y / magnitude
z_normalized = z / magnitude

print("Unit Vector: ({:.6f}, {:.6f}, {:.6f})".format(x_normalized, y_normalized, z_normalized))
rotation_axis = np.array([0.037729, 0.495995, -0.867505])             # Unit vector pointing along the rotation axis
body_orientation = np.array([1, 0, 1])              # Unit vector pointing along the body's orientation
convergence_target = 0.01                              # K

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

    facets = []
    for i in range(len(lines)):
        if lines[i].strip().startswith('facet normal'):
            normal = np.array([float(n) for n in lines[i].strip().split()[2:]])
            vertex1 = np.array([float(v) for v in lines[i+2].strip().split()[1:]])
            vertex2 = np.array([float(v) for v in lines[i+3].strip().split()[1:]])
            vertex3 = np.array([float(v) for v in lines[i+4].strip().split()[1:]])
            facets.append({'normal': normal, 'vertices': [vertex1, vertex2, vertex3]})
    
    # Process facets to calculate area and centroid
    for i,  facet in enumerate(facets):
        v1, v2, v3 = facet['vertices']
        area = calculate_area(v1, v2, v3)
        centroid = (v1 + v2 + v3) / 3
        # facet['normal'] = normal[i]
        facet['area'] = area
        facet['position'] = centroid
        #initialise insolation
        facet['insolation'] = np.zeros(timesteps_per_day) # Insolation curve doesn't change day to day
        # initialise visible facets array NOTE: Need to add secondary radiation coefficients
        facet['visible_facets'] = np.zeros(len(facets))
        #initialise temperature arrays
        facet['temperature'] = np.zeros((timesteps_per_day * (max_days + 1), n_layers))

    print(f"Read {len(facets)} facets from the shape model.\n")
    
    return facets

def calculate_area(v1, v2, v3):
    '''Calculate the area of the triangle formed by vertices v1, v2, and v3.'''
    u = v2 - v1
    v = v3 - v1
    return np.linalg.norm(np.cross(u, v)) / 2

def calculate_visible_facets(facets):
    ''' 
    PLACEHOLDER WITH ONE POSSIBLE METHOD. This function calculates the visible (test) facets from each subject facet. It calculates the angle between the normal vector of each facet and the line of sight to every other facet. It writes the indices of the visible facets to the data cube.
    
    Issues:
        1) Doesn't account for partial shadowing (e.g. a facet may be only partially covered by the shadow cast by another facet) - more of an issue for high facet count models. 
        2) Shadowing is not calculated for secondary radiation ie if three or more facets are in a line, the third facet will not be shadowed by the second from radiation emitted by the first.
    '''
    # Set up two nested loops that go through each subject facet and tests each other facet
    for i, subject_facet in enumerate(facets):
        for j, test_facet in enumerate(facets):
            if i == j:
                continue
            # Calculate whether the center of the test facet is above the plane of the subject facet
            if np.dot(subject_facet['normal'], test_facet['position'] - subject_facet['position']) > 0:
                # Calculate whether the test facet faces towards the subject facet, if both tests are passed, the test facet is visible
                if np.dot(subject_facet['normal'], test_facet['normal']) > 0:
                    # Write the index of the visible facet to the data cube
                    subject_facet['visible_facets'][j] = 1

    

    return facets

def calculate_shadowing(facet_position, sunlight_direction, visible_facets, rotation_axis, rotation_period, timesteps_per_day, delta_t, t):
    '''
    PLACEHOLDER: This function calculates whether a facet is in shadow at a given time step. It cycles through all visible facets and checks whether they fall on the sunlight direction vector. If they do, the facet is in shadow. 
    
    It returns the illumination factor for the facet at that time step.
    '''
    return 1

def calculate_insolation(shape_model):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.
    '''

    # Calculate rotation matrix for the body's rotation
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
        for t in range(timesteps_per_day):
            # Normal vector of the facet at time t=0
            normal = facet['normal']

            new_normal = np.dot(rotation_matrix(rotation_axis, (2 * np.pi * delta_t / rotation_period) * t), normal)

            # Calculate zenith angle
            zenith_angle = np.arccos(np.dot(sunlight_direction, new_normal) / (np.linalg.norm(sunlight_direction) * np.linalg.norm(new_normal)))

            #  Elimate angles where the sun is below the horizon
            if zenith_angle > np.pi / 2:
                insolation = 0

            else:
                # Calculate illumination factor, pass facet position, sunlight direction, shape model and rotation information (rotation axis, rotation period, timesteps per day, delta t, t)
                illumination_factor = calculate_shadowing(facet['position'], sunlight_direction, facet['visible_facets'], rotation_axis, rotation_period, timesteps_per_day, delta_t, t)

                # Calculate insolation converting AU to m
                insolation = solar_luminosity * (1 - albedo) * illumination_factor * np.cos(zenith_angle) / (4 * np.pi * solar_distance**2) 
                
            # Write the insolation value to the insolation array for this facet at time t
            facet['insolation'][t] = insolation

    print(f"Calculated insolation for each facet.\n")

    # Plot the insolation curve for a single facet with number of days on the x-axis
    plt.plot(shape_model[0]['insolation'])
    plt.xlabel('Number of timesteps')
    plt.ylabel('Insolation (W/m^2)')
    plt.title('Insolation curve for a single facet for one full rotation of the body')
    plt.show()

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
        total_energy = np.trapz(facet['insolation'], dx=delta_t)
        # Calculate the temperature of the facet using the Stefan-Boltzmann law and set the initial temperature of all layers to the same value
        for layer in range(n_layers):
            facet['temperature'][0][layer] = (total_energy / (emmisivity * facet['area'] * 5.67e-8))**(1/4)

    print(f"Calculated initial temperatures for each facet.\n")

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
    #path_to_filename = "shape_models/Bennu_not_to_scale_98_facets.stl"
    #path_to_filename = "Bennu_Testing\shape_models\Bennu_not_to_scale_98_facets.stl"
    #path_to_filename = r"C:\Users\chikani.OXAOPPWELAP5\comet_nucleus_model\Bennu_Testing\shape_models\Bennu_not_to_scale_98_facets.stl"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "shape_models/Bennu_not_to_scale_98_facets.stl"
    filename = os.path.join(script_dir, relative_path)


    shape_model = read_shape_model(filename)
 
    # Visualise the shape model
    visualise_shape_model(filename, rotation_axis, rotation_period, solar_distance_au, sunlight_direction)

    # Calculate insolation array for each facet
    shape_model = calculate_insolation(shape_model)

    # Calulate initial temperature array
    shape_model = calculate_initial_temperatures(shape_model)

    # Calculate secondary radiation array
        # Ray tracing to work out which facets are visible from each facet
        # Calculate the geometric coefficient of secondary radiation from each facet
        # Write the index and coefficient to the data cube
    
    convergence_factor = 10 # Set to a value greater than 1 to start the iteration
    day = 0 

    # Proceed to iterate the model until it converges
    while day < max_days and convergence_factor > 1:
        for time_step in range(timesteps_per_day):
            for facet in shape_model:
                current_step = int(time_step + (day * timesteps_per_day))
                # Calculate insolation term, bearing in mind that the insolation curve is constant for each facet and repeats every rotation period
                insolation_term = facet['insolation'][time_step] * delta_t / (layer_thickness * density * specific_heat_capacity)

                # Calculate re-emitted radiation term
                re_emitted_radiation_term = emmisivity * beaming_factor * 5.67e-8 * (facet['temperature'][current_step][0]**4) * delta_t / (layer_thickness * density * specific_heat_capacity)

                # Calculate secondary radiation term (identify facets above horizon first, then check if they face, same process for shadows but maybe segment facet into shadow/light with a calculated line?)
                # Calculate conducted heat term
                conducted_heat_term = thermal_conductivity * (facet['temperature'][current_step][1] - facet['temperature'][current_step][0]) * delta_t / (layer_thickness * density * specific_heat_capacity)

                # Calculate sublimation energy loss term

                # Calculate the new temperature of the surface layer (currently very simplified)
                facet['temperature'][current_step + 1][0] = facet['temperature'][current_step][0] + insolation_term - re_emitted_radiation_term + conducted_heat_term

                # Calculate the new temperatures of the subsurface layers, ensuring that the temperature of the deepest layer is fixed at its current value
                for layer in range(1, n_layers - 1):
                    facet['temperature'][current_step + 1][layer] = facet['temperature'][current_step][layer] + thermal_conductivity * (facet['temperature'][current_step][layer + 1] - 2 * facet['temperature'][current_step][layer] + facet['temperature'][current_step][layer - 1]) * delta_t / (layer_thickness**2 * density * specific_heat_capacity)

        # Calculate convergence factor (average temperature error at surface across all facets divided by convergence target)
        day += 1

        temperature_error = 0
        for facet in shape_model:
                temperature_error += abs(facet['temperature'][day * timesteps_per_day][0] - facet['temperature'][(day - 1) * timesteps_per_day][0])

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
                final_day_temperatures[i][t] = facet['temperature'][day * timesteps_per_day + t][0]

        # Plot the final day's temperature distribution for all facets
        plt.plot(final_day_temperatures.T)
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for all facets')
        plt.show()
        
        # Visualise the results - animation of final day's temperature distribution
        animate_temperature_distribution(filename, final_day_temperatures, rotation_axis, rotation_period, solar_distance_au, sunlight_direction, timesteps_per_day, delta_t)

        # Save a sample of the final day's temperature distribution to a file
        np.savetxt('test_data/final_day_temperatures.csv', final_day_temperatures, delimiter=',')
    
    else:
        print(f"Maximum days reached without achieving convergence. \n\nFinal temperature error: {temperature_error / (len(shape_model))} K\n")

    # Save the final day temperatures to a file that can be used with ephemeris to produce instrument simulations NOTE: This may be a large file that takes a long time to run for large shape models
    #np.savetxt('outputs/final_day_temperatures.csv', final_day_temperatures, delimiter=',')

# Call the main program to start execution
if __name__ == "__main__":
    main()
