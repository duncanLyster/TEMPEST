''' 
This Python script simulates diurnal temperature variations of a solar system body based on
a given shape model. It reads in the shape model, sets material and model properties, calculates 
insolation and temperature arrays, and iterates until the model converges. The results are saved and 
visualized.

It was built as a tool for planning the comet interceptor mission, but is intended to be 
generalised for use with asteroids, and other planetary bodies e.g. fractures on 
Enceladus' surface.

All calculation figures are in SI units, except where clearly stated otherwise.

Full documentation can be found at: https://github.com/duncanLyster/comet_nucleus_model

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
7) Work out why model not converging since adding deep temp
8) Consider scattering of light from facets (as opposed to just re-radiation)
9) Add parallelisation to the model

OPTIONAL NEXT STEPS (fun):
- Implement secondary radiation/self-heating
- Implement sublimation energy loss
- Ensure colour scale is consistent across frames
- Run very high resolution models
- Build in mesh converstion for binary .STL and .OBJ files
- Create web interface for ease of use?
- Integrate with JPL Horizons ephemeris to get real-time insolation data
- Come up with a way of representing output data for many rotation axes and periods for mission planning | Do this and provide recommendations to MIRMIS team
- Add filter visualisations to thermal model
    - Simulate retrievals for temperature based on instrument
- Ongoing verification against J. Spencer's thermprojrs - currrently good agreement but there are small systematic differences. 

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
from animate_temperature_distribution import animate_temperature_distribution
from plot_temperature_distribution import plot_temperature_distribution
from nice_gif import nice_gif
from animate_shadowing import animate_shadowing
from animate_model import animate_model
from numba import jit, types
from joblib import Parallel, delayed
from stl import mesh
from tqdm import tqdm
from typing import Tuple
from scipy.interpolate import interp1d
import ipywidgets as widgets
from IPython.display import display
from numba.typed import List
from numba.core import types
from numba.extending import overload

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
        self.solar_distance_m = self.solar_distance_au * 1.496e11  # Convert AU to meters
        self.rotation_period_s = self.rotation_period_hours * 3600  # Convert hours to seconds
        self.angular_velocity = (2 * np.pi) / self.rotation_period_s  # Calculate angular velocity in rad/s
        self.skin_depth = (self.thermal_conductivity / (self.density * self.specific_heat_capacity * self.angular_velocity))**0.5
        self.thermal_inertia = (self.density * self.specific_heat_capacity * self.thermal_conductivity)**0.5
        self.layer_thickness = 8 * self.skin_depth / self.n_layers
        self.thermal_diffusivity = self.thermal_conductivity / (self.density * self.specific_heat_capacity)
        self.timesteps_per_day = int(round(self.rotation_period_s / (self.layer_thickness**2 / (2 * self.thermal_diffusivity)))) # Courant-Friedrichs-Lewy condition for conduction stability
        self.delta_t = self.rotation_period_s / self.timesteps_per_day

        # Print out the configuration
        print(f"Configuration loaded from {config_path}")
        for key, value in config.items():
            print(f"{key}: {value}")
        
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
        # self.unphysical_energy_loss = np.zeros(timesteps_per_day * (max_days + 1))
        # self.insolation_energy = np.zeros(timesteps_per_day * (max_days + 1))
        # self.re_emitted_energy = np.zeros(timesteps_per_day * (max_days + 1))
        # self.surface_energy_change = np.zeros(timesteps_per_day * (max_days + 1))
        # self.conducted_energy = np.zeros(timesteps_per_day * (max_days + 1))

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

def calculate_visible_facets(positions, normals):
    ''' 
    This function calculates the visible (test) facets from each subject facet. It calculates the angle between the normal vector of each facet and the line of sight to every other facet. It writes the indices of the visible facets to the data cube.
    
    NB: This doesn't account for partial shadowing (e.g. a facet may be only partially covered by the shadow cast by another facet) - more of an issue for low facet count models. Additionally, for very complicated shapes, facets may be identified as visible when there are structures in the way.
    '''
    visible_indices =[[] for _ in range(len(positions))]
    
    for i in range(len((positions))):
        # Compute the relative positions of all facets from the current subject facet, this results in a vector from the subject facet to every other facet
        relative_positions =  positions[i] - positions

        # The dot product between the relative positions and the normals of the subject facet tells us if the facet is above the horizon
        above_horizon = relative_positions @ normals[i] < 0
        
        # Thr dot product between the relative positions and the normals of the subject facet tells us if the facet is facing towards the subject facet
        facing_towards = np.einsum('ij,ij->i', -relative_positions, normals) < 0 
    
        # Combine the two conditions to determine if the facet is visible
        visible = above_horizon & facing_towards
    
        # Ensure that the facet does not consider itself as visible
        visible[i] = False
        
        # Write the indices of the visible facets to the subject facet
        visible_indices[i] = np.where(visible)[0]

    return visible_indices

@jit(nopython=True)
def calculate_shadowing(ray_origin, ray_direction, shape_model_vertices, facet_indices):
    '''
    This function calculates whether a facet is in shadow at a given time step. It cycles through all visible facets and passes their vertices to rays_triangles_intersections which determines whether they fall on the sunlight direction vector (starting at the facet position). If they do, the facet is in shadow. 
    
    It returns the illumination factor for the facet at that time step. 0 if the facet is in shadow, 1 if it is not.
    '''

    # Ensure triangles_vertices is an array of shape (m, 3, 3)
    triangles_vertices = shape_model_vertices[facet_indices]

    # Call the intersection function
    intersections, t_values = rays_triangles_intersection(
        ray_origin,
        ray_direction,
        triangles_vertices
    )

    # Check for any intersection
    if intersections.any():
        return 0  # The facet is in shadow
        
    return 1 # The facet is not in shadow

@jit(nopython=True)
def rays_triangles_intersection(
    ray_origin: np.ndarray, ray_directions: np.ndarray, triangles_vertices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sourced from: https://gist.github.com/V0XNIHILI/87c986441d8debc9cd0e9396580e85f4

    Möller–Trumbore intersection algorithm for calculating whether the ray intersects the triangle
    and for which t-value. Based on: https://github.com/kliment/Printrun/blob/master/printrun/stltool.py,
    which is based on:
    http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    Parameters
    ----------
    ray_origin : np.ndarray(3)
        Origin coordinate (x, y, z) from which the ray is fired
    ray_directions : np.ndarray(n, 3)
        Directions (dx, dy, dz) in which the rays are going
    triangle_vertices : np.ndarray(m, 3, 3)
        3D vertices of multiple triangles
    Returns
    -------
    tuple[np.ndarray<bool>(n, m), np.ndarray(n, m)]
        The first array indicates whether or not there was an intersection, the second array
        contains the t-values of the intersections
    """

    output_shape = (len(ray_directions), len(triangles_vertices))

    all_rays_t = np.full(output_shape, np.nan)
    all_rays_intersected = np.zeros(output_shape, dtype=np.bool_)

    v1 = triangles_vertices[:, 0]
    v2 = triangles_vertices[:, 1]
    v3 = triangles_vertices[:, 2]

    eps = 0.000001

    edge1 = v2 - v1
    edge2 = v3 - v1

    for i, ray in enumerate(ray_directions):
        all_t = np.zeros((len(triangles_vertices)))
        intersected = np.full((len(triangles_vertices)), True)

        pvec = np.cross(ray, edge2)

        det = np.sum(edge1 * pvec, axis=1)

        non_intersecting_original_indices = np.absolute(det) < eps

        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        inv_det = 1.0 / det

        tvec = ray_origin - v1

        u = np.sum(tvec * pvec, axis=1) * inv_det

        non_intersecting_original_indices = (u < 0.0) + (u > 1.0)
        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        qvec = np.cross(tvec, edge1)

        v = np.sum(ray * qvec, axis=1) * inv_det

        non_intersecting_original_indices = (v < 0.0) + (u + v > 1.0)

        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        t = (
            np.sum(
                edge2 * qvec,
                axis=1,
            )
            * inv_det
        )

        non_intersecting_original_indices = t < eps
        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        intersecting_original_indices = np.invert(non_intersecting_original_indices)
        all_t[intersecting_original_indices] = t[intersecting_original_indices]

        all_rays_t[i] = all_t
        all_rays_intersected[i] = intersected

    return all_rays_intersected, all_rays_t

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

def calculate_insolation(shape_model, simulation):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.

    NOTE: This could be done as the model runs rather than saving at the start to reduce storage in RAM. Would applying rotation matrix to the entire shape model be faster than applying it to each facet? Or don't use the rotation matrix at all and just work out the geometry of the insolation at each timestep?
    '''
    # Initialize insolation array with zeros for all facets and timesteps
    insolation_array = np.zeros((len(shape_model), simulation.timesteps_per_day))

    # Precompute rotation matrices and rotated sunlight directions
    rotation_matrices = np.zeros((simulation.timesteps_per_day, 3, 3))
    rotated_sunlight_directions = np.zeros((simulation.timesteps_per_day, 3))

    for t in range(simulation.timesteps_per_day):
        rotation_matrix = calculate_rotation_matrix(simulation.rotation_axis, (2 * np.pi / simulation.timesteps_per_day) * t)
        rotation_matrices[t] = rotation_matrix
        
        rotated_sunlight_direction = np.dot(rotation_matrix.T, simulation.sunlight_direction)
        rotated_sunlight_direction /= np.linalg.norm(rotated_sunlight_direction)
        rotated_sunlight_directions[t] = rotated_sunlight_direction

    sunlight_direction_norm = np.linalg.norm(simulation.sunlight_direction)

    shape_model_vertices = np.array([facet.vertices for facet in shape_model])
    
    for i, facet in enumerate(tqdm(shape_model, desc="Calculating insolation")):
        normal = facet.normal
        
        for t in range(simulation.timesteps_per_day):
            new_normal = np.dot(rotation_matrices[t], normal)
            new_normal_norm = np.linalg.norm(new_normal)  # Precompute new normal vector norm
            sun_dot_normal = np.dot(simulation.sunlight_direction, new_normal)
            
            # Precompute cosine of zenith angle
            cos_zenith_angle = sun_dot_normal / (sunlight_direction_norm * new_normal_norm)
            
            # Zenith angle calculation
            if cos_zenith_angle > 0:
                illumination_factor = 1  # Default to no shadowing

                if len(facet.visible_facets) != 0:
                    illumination_factor = calculate_shadowing(np.array(facet.position), np.array([rotated_sunlight_directions[t]]), shape_model_vertices, facet.visible_facets)
                
                # Calculate insolation
                insolation = simulation.solar_luminosity * (1 - simulation.albedo) * illumination_factor * cos_zenith_angle / (4 * np.pi * simulation.solar_distance_m**2)
            else:
                insolation = 0
            
            insolation_array[i][t] = insolation
            facet.insolation[t] = insolation

    return shape_model, insolation_array

def calculate_initial_temperatures(shape_model, n_layers, emissivity, n_jobs=-1):
    ''' 
    This function calculates the initial temperature of each facet and sub-surface layer of the body based on the insolation curve for that facet. It writes the initial temperatures to the data cube.
    '''
    # Stefan-Boltzmann constant
    sigma = 5.67e-8

    # Define the facet processing function inside the main function
    def process_facet(insolation, emissivity, sigma):
        # Calculate the initial temperature based on average power in
        power_in = np.mean(insolation)
        # Calculate the temperature of the facet using the Stefan-Boltzmann law
        calculated_temp = (power_in / (emissivity * sigma))**(1/4)

        # Return the calculated temperature for all layers
        return calculated_temp

    # Parallel processing of facets
    results = Parallel(n_jobs=n_jobs)(delayed(process_facet)(facet.insolation, emissivity, sigma) for facet in shape_model)

    print(f"Initial temperatures calculated for {len(shape_model)} facets.")

    # Update the original shape_model with the results NOTE: This step is causing the crash for large shape models. 
    for facet, temperature in zip(shape_model, results):
        facet.temperature[:] = temperature

    print("Initial temperatures saved for all facets.")

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
    This is the main calculation function for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges.
    '''

    mean_temperature_error = simulation.convergence_target + 1 # Set to convergence target to start the loop
    day = 0 
    temperature_error = 0

    const1 = simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
    const2 = simulation.emissivity * simulation.beaming_factor * 5.67e-8 * simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
    const3 = simulation.thermal_diffusivity * simulation.delta_t / simulation.layer_thickness**2

    # Proceed to iterate the model until it converges as long as the maximum number of days has not been reached, ensuring the minimum number of days is observed
    while day < simulation.max_days and (day < simulation.min_days or mean_temperature_error > simulation.convergence_target):
        # Include tqdm for progress bar
        for time_step in tqdm(range(simulation.timesteps_per_day), desc=f"Day {day + 1}"):
            for facet in shape_model:
                current_step = int(time_step + (day * simulation.timesteps_per_day))
                

                ############################# Calculate temperature of each layer for each facet #############################

                # Calculate insolation term, bearing in mind that the insolation curve is constant for each facet and repeats every rotation period
                insolation_term = facet.insolation[time_step] * const1
                #simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)

                # Calculate re-emitted radiation term
                re_emitted_radiation_term = - const2 * (facet.temperature[current_step][0]**4)

                # Calculate secondary radiation term TODO: Implement this
                # secondary_radiation_term = calculate_secondary_radiation_term(shape_model, facet, delta_t) #NOTE calculate secondary radiation coefficients first

                # Calculate sublimation energy loss term TODO: Implement this

                # Calculate conducted heat term
                conducted_heat_term = const3 * (facet.temperature[current_step][1] - facet.temperature[current_step][0])

                # Calculate the new temperatures of the subsurface layers, ensuring that the temperature of the deepest layer is fixed at its current value
                for layer in range(1, simulation.n_layers - 1):
                    facet.temperature[current_step + 1][layer] = facet.temperature[current_step][layer] + const3 * (facet.temperature[current_step][layer + 1] - 2 * facet.temperature[current_step][layer] + facet.temperature[current_step][layer - 1])

                # Calculate the new temperature of the surface layer
                facet.temperature[current_step + 1][0] = facet.temperature[current_step][0] + insolation_term + re_emitted_radiation_term + conducted_heat_term

                # Print the heat terms for the facet
                # if shape_model.index(facet) == 0:
                #     print(f"Facet: {shape_model.index(facet)}, Time step: {time_step}, "
                #         f"Current step temp: {facet.temperature[current_step][0]} K, "
                #         f"Insolation term: {insolation_term}, "
                #         f"Re-emitted radiation term: {re_emitted_radiation_term}, "
                #         f"Conducted heat term: {conducted_heat_term}, "
                #         f"New temp: {facet.temperature[current_step + 1][0]} K, "
                #         f"Delta t: {simulation.delta_t}")

                # ############################# Calculate unphysical energy for facet (assuming unit area) #############################
                # facet.insolation_energy[current_step] = facet.insolation[time_step] * simulation.delta_t
                # facet.re_emitted_energy[current_step] = -simulation.emissivity * simulation.beaming_factor * 5.670374419e-8 * (facet.temperature[current_step][0]**4) * simulation.delta_t
                # facet.surface_energy_change[current_step] = simulation.density * simulation.specific_heat_capacity * simulation.layer_thickness * (facet.temperature[current_step + 1][0] - facet.temperature[current_step][0])

                # sub_surface_energy_conducted = 0

                # for layer in range(1, simulation.n_layers - 1):
                #     sub_surface_energy_conducted += simulation.density * simulation.specific_heat_capacity * simulation.layer_thickness * (facet.temperature[current_step + 1][layer] - facet.temperature[current_step][layer])

                # facet.conducted_energy[current_step] = -sub_surface_energy_conducted

                # facet.unphysical_energy_loss[current_step] = facet.insolation_energy[current_step] + facet.re_emitted_energy[current_step] - facet.surface_energy_change[current_step] + facet.conducted_energy[current_step]

                # ############################# Calculate unphysical energy for facet surface only #############################
                # facet.insolation_energy[current_step] = facet.insolation[time_step] * simulation.delta_t
                # facet.re_emitted_energy[current_step] = - simulation.emissivity * simulation.beaming_factor * 5.670374419e-8 * (facet.temperature[current_step][0]**4) * simulation.delta_t
                # facet.surface_energy_change[current_step] = - simulation.density * simulation.specific_heat_capacity * simulation.layer_thickness * (facet.temperature[current_step + 1][0] - facet.temperature[current_step][0])
                # facet.conducted_energy[current_step] = simulation.thermal_conductivity * simulation.delta_t * (facet.temperature[current_step][1] - facet.temperature[current_step][0]) / simulation.layer_thickness

                # facet.unphysical_energy_loss[current_step] = facet.insolation_energy[current_step] + facet.re_emitted_energy[current_step] + facet.surface_energy_change[current_step] + facet.conducted_energy[current_step]

                if np.isnan(facet.temperature[current_step][0]) or np.isinf(facet.temperature[current_step][0]) or facet.temperature[current_step][0] < 0:
                    print(f"Ending run at timestep {current_step} due to facet {shape_model.index(facet)} having a temperature of {facet.temperature[current_step][0]} K.\n Try increasing the number of time steps per day")
                    
                    # # Plot the energy terms for the facet
                    # plt.plot(facet.insolation_energy[:current_step], label="Insolation energy")
                    # plt.plot(facet.re_emitted_energy[:current_step], label="Re-emitted energy")
                    # plt.plot(facet.surface_energy_change[:current_step], label="Surface energy change")
                    # plt.plot(facet.conducted_energy[:current_step], label="Conducted energy")
                    # plt.plot(facet.unphysical_energy_loss[:current_step], label="Unphysical energy loss")
                    # plt.legend()
                    # plt.show()

                    # Plot the insolation curve for the facet
                    plt.plot(facet.insolation)
                    plt.xlabel('Number of timesteps')
                    plt.ylabel('Insolation (W/m^2)')
                    plt.title('Insolation curve for a single facet for one full rotation')
                    plt.show()

                    # Plot sub-surface temperatures for the facet
                    for layer in range(1, simulation.n_layers):
                        plt.plot([facet.temperature[t][layer] for t in range(current_step)])
                    plt.xlabel('Number of timesteps')
                    plt.ylabel('Temperature (K)')
                    plt.title('Sub-surface temperature curve for a single facet for one full rotation')
                    plt.legend([f"Layer {layer}" for layer in range(1, simulation.n_layers)])
                    plt.show()


                    sys.exit()

            # Calculate convergence factor (average temperature error at surface across all facets divided by convergence target)
            temperature_error = 0
            
            for facet in shape_model:
                    temperature_error += abs(facet.temperature[day * simulation.timesteps_per_day][0] - facet.temperature[(day - 1) * simulation.timesteps_per_day][0])

        # Calculate the mean temperature of the surface layer for each facet for the last full day
        mean_surface_temperatures = np.zeros(len(shape_model))
        for i, facet in enumerate(shape_model):
            mean_surface_temperatures[i] = np.mean([facet.temperature[day * simulation.timesteps_per_day + t][0] for t in range(simulation.timesteps_per_day)])

        # Set deep temperature for each facet to the average of surface layer temperature of that facet for the last full day for all future timesteps 
        for i, facet in enumerate(shape_model):
            for t in range(simulation.timesteps_per_day):
                facet.temperature[(day + 1) * simulation.timesteps_per_day + t][simulation.n_layers - 1] = mean_surface_temperatures[i]

        mean_temperature_error = temperature_error / (len(shape_model))

        print(f"Day: {day} | Temperature error: {mean_temperature_error} K | Convergence target: {simulation.convergence_target} K")
        
        day += 1

    # Decrement the day counter
    day -= 1    
    
    if mean_temperature_error < simulation.convergence_target:

        # Create an array of temperatures at each timestep in final day for the surface layer of each facet
        final_day_temperatures = np.zeros((len(shape_model), simulation.timesteps_per_day))

        # Fill the array
        for i, facet in enumerate(shape_model):
            for t in range(simulation.timesteps_per_day):
                final_day_temperatures[i][t] = facet.temperature[day * simulation.timesteps_per_day + t][0]

        # Create an array of final timestep temperatures
        final_timestep_temperatures = np.zeros(len(shape_model))
        for i, facet in enumerate(shape_model):
            final_timestep_temperatures[i] = facet.temperature[(day + 1) * simulation.timesteps_per_day][0]

        # Create an array of temperatures at each timestep in final day for all layers of each facet
        final_day_temperatures_all_layers = np.zeros((len(shape_model), simulation.timesteps_per_day, simulation.n_layers))

        # Fill the array
        for i, facet in enumerate(shape_model):
            for t in range(simulation.timesteps_per_day):
                for layer in range(simulation.n_layers):
                    final_day_temperatures_all_layers[i][t][layer] = facet.temperature[day * simulation.timesteps_per_day + t][layer]

    else:
        final_timestep_temperatures = None
        print(f"Maximum days reached without achieving convergence. \n\nFinal temperature error: {temperature_error / (len(shape_model))} K\n Try increasing max_days or decreasing convergence_target.")

        # Break the loop and return None
        sys.exit()

    return final_day_temperatures, final_day_temperatures_all_layers, final_timestep_temperatures, day+1, temperature_error

def main():
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.
    '''

    # Shape model name
    shape_model_name = "67P_not_to_scale_low_res.stl"

    # Get setup file and shape model
    path_to_shape_model_file = f"shape_models/{shape_model_name}"
    path_to_setup_file = "model_setups/John_Spencer_default_model_parameters.json"

    # Load setup parameters from JSON file
    simulation = Simulation(path_to_setup_file)

    print(f"Number of timesteps per day: {simulation.timesteps_per_day}\n")
    print(f"Layer thickness: {simulation.layer_thickness} m\n")
    print(f"Thermal inertia: {simulation.thermal_inertia} W m^-2 K^-1 s^0.5\n")
    print(f"Skin depth: {simulation.skin_depth} m\n")

    shape_model = read_shape_model(path_to_shape_model_file, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days)

    # Setup the model
    print(f"Calculating visible facets.\n")
    positions = np.array([facet.position for facet in shape_model])
    normals = np.array([facet.normal for facet in shape_model])
    visible_indices = calculate_visible_facets(positions,normals)

    for i in range(len(shape_model)):
        shape_model[i].visible_facets = visible_indices[i]

    shape_model, insolation_array = calculate_insolation(shape_model, simulation)

    ################ PLOTTING ################
    plot_shadowing = True
    plot_insolation_curve = False
    plot_initial_temp_histogram = False
    plot_final_day_temp_distribution = False
    plot_final_day_all_layers_temp_distribution = False
    plot_all_days_all_layers_temp_distribution = False
    plot_energy_terms = False
    plot_temp_distribution_for_final_day = False
    animate_final_day_temp_distribution = True
    animate_final_day_temp_distribution_display = False
    plot_final_day_comparison = False

    facet_index = 0 # Index of facet to plot

    if plot_shadowing:
        print(f"Preparing shadowing visualisation.\n")
        # animate_shadowing(path_to_shape_model_file, insolation_array, simulation.rotation_axis, simulation.sunlight_direction, simulation.timesteps_per_day)

        animate_model(path_to_shape_model_file, insolation_array, simulation.rotation_axis, simulation.sunlight_direction, simulation.timesteps_per_day, colour_map='binary_r', plot_title='Shadowing on the body', axis_label='Insolation (W/m^2)', animation_frames=200, save_animation=False, save_animation_name='shadowing_animation.gif', background_colour = 'white')

    if plot_insolation_curve:
        fig_insolation = plt.figure()
        plt.plot(shape_model[facet_index].insolation)
        plt.xlabel('Number of timesteps')
        plt.ylabel('Insolation (W/m^2)')
        plt.title('Insolation curve for a single facet for one full rotation of the body')
        fig_insolation.show()

    print(f"Calculating initial temperatures.\n")
    shape_model = calculate_initial_temperatures(shape_model, simulation.n_layers, simulation.emissivity)

    if plot_initial_temp_histogram:
        fig_histogram = plt.figure()
        initial_temperatures = [facet.temperature[0][0] for facet in shape_model]
        plt.hist(initial_temperatures, bins=20)
        plt.xlabel('Initial temperature (K)')
        plt.ylabel('Number of facets')
        plt.title('Initial temperature distribution of all facets')
        fig_histogram.show()

    print(f"Calculating secondary radiation coefficients.\n")
    shape_model = calculate_secondary_radiation_coefficients(shape_model)
    
    print(f"Running main simulation loop.\n")
    start_time = time.time()
    final_day_temperatures, final_day_temperatures_all_layers, final_timestep_temperatures, day, temperature_error = thermophysical_body_model(shape_model, simulation)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Convergence target achieved after {day} days.\n\nFinal temperature error: {temperature_error / (len(shape_model))} K\n")
    print(f"Execution time: {execution_time} seconds")

    if plot_final_day_temp_distribution:
        fig_final_temp_dist = plt.figure()
        plt.plot(final_day_temperatures[facet_index])
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for all facets')
        fig_final_temp_dist.show()

    if plot_final_day_all_layers_temp_distribution:
        fig_final_all_layers_temp_dist = plt.figure()
        plt.plot(final_day_temperatures_all_layers[facet_index])
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for all layers in facet')
        fig_final_all_layers_temp_dist.show()

    if plot_all_days_all_layers_temp_distribution:
        fig_all_days_all_layers_temp_dist = plt.figure()
        plt.plot(shape_model[facet_index].temperature[:(day) * simulation.timesteps_per_day, :])
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature distribution for all layers in facet for the full run')
        fig_all_days_all_layers_temp_dist.show()

    if plot_energy_terms:
        fig_energy_terms = plt.figure()
        plt.plot(shape_model[facet_index].unphysical_energy_loss[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Unphysical energy loss')
        plt.plot(shape_model[facet_index].insolation_energy[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Insolation energy')
        plt.plot(shape_model[facet_index].re_emitted_energy[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Re-emitted energy')
        plt.plot(-shape_model[facet_index].surface_energy_change[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Surface energy change')
        plt.plot(shape_model[facet_index].conducted_energy[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Conducted energy')
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('Energy (J)')
        plt.title('Energy terms for facet for the final day')
        fig_energy_terms.show()

    if plot_temp_distribution_for_final_day:
        fig_final_day_temps = plt.figure()
        plt.plot(shape_model[facet_index].temperature[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day])
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature distribution for all layers in facet for the full run')
        fig_final_day_temps.show()

    if animate_final_day_temp_distribution:
        print(f"Preparing temperature animation.\n")
        #animate_temperature_distribution(path_to_shape_model_file, final_day_temperatures, simulation.rotation_axis, simulation.rotation_period_s, simulation.solar_distance_au, simulation.sunlight_direction, simulation.timesteps_per_day, simulation.delta_t)

        animate_model(path_to_shape_model_file, final_day_temperatures, simulation.rotation_axis, simulation.sunlight_direction, simulation.timesteps_per_day, colour_map='coolwarm', plot_title='Temperature distribution on the body', axis_label='Temperature (K)', animation_frames=200, save_animation=False, save_animation_name='temperature_animation.gif', background_colour = 'black')


    if animate_final_day_temp_distribution_display:
        print(f"Preparing temperature animation for display.\n")
        nice_gif(path_to_shape_model_file, final_day_temperatures, simulation.rotation_axis, simulation.sunlight_direction, simulation.timesteps_per_day)

    if plot_final_day_comparison:
        print(f"Saving final day temperatures for facet to CSV file.\n")
        np.savetxt("final_day_temperatures.csv", np.column_stack((np.linspace(0, 2 * np.pi, simulation.timesteps_per_day), final_day_temperatures[facet_index])), delimiter=',', header='Rotation angle (rad), Temperature (K)', comments='')

        print(f"Saving folder with final timestep temperatures.\n")
        export_results(shape_model_name, path_to_setup_file, path_to_shape_model_file, final_day_temperatures[:, -1])

        thermprojrs_data = np.loadtxt("thermprojrs_data.csv", delimiter=',', skiprows=1)

        fig_model_comparison = plt.figure()
        plt.plot(thermprojrs_data[:, 0], thermprojrs_data[:, 1], label='Thermprojrs')
        plt.plot(np.linspace(0, 2 * np.pi, simulation.timesteps_per_day), final_day_temperatures[facet_index], label='This model')
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for facet')
        plt.legend()
        fig_model_comparison.show()

        x_original = np.linspace(0, 2 * np.pi, simulation.timesteps_per_day)
        x_new = np.linspace(0, 2 * np.pi, thermprojrs_data.shape[facet_index])

        interp_func = interp1d(x_new, thermprojrs_data[:, 1], kind='linear')
        thermprojrs_interpolated = interp_func(x_original)

        fig_model_difference = plt.figure()
        plt.plot(x_original, final_day_temperatures[facet_index] - thermprojrs_interpolated, label='This model')
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Temperature difference (K)')
        plt.title('Temperature difference between this model and Thermprojrs for facet')
        plt.legend()
        plt.show()

        np.savetxt("final_day.csv", np.column_stack((x_original, final_day_temperatures[facet_index])), delimiter=',', header='Rotation angle (rad), Temperature (K)', comments='')


    print(f"Model run complete.\n")

# Call the main program to start execution
if __name__ == "__main__":
    main()