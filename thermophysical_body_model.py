''' 
This Python script simulates diurnal temperature variations of a solar system body based on
a given shape model. It reads in the shape model, sets material and model properties, calculates 
insolation and temperature arrays, and iterates until the model converges. The results are saved and 
visualized.

It was built as a tool for planning the comet interceptor mission, but is intended to be 
generalised for use with asteroids, and other planetary bodies e.g. fractures on 
Enceladus' surface.

All calculation figures are in SI units, except where clearly stated otherwise.

Full documentation can be found at: https://github.com/duncanLyster/TEMPEST

Started: 15 Feb 2024

Author: Duncan Lyster
'''

import hashlib
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys
import json
import pandas as pd
import datetime
import warnings
from animate_model import animate_model
from numba import jit, njit, float64, int64, boolean
from numba.typed import List
from joblib import Parallel, delayed, cpu_count
from stl import mesh
from tqdm import tqdm
from typing import Tuple
from scipy.interpolate import interp1d
from collections import defaultdict

class Config:
    def __init__(self):
        ################ START OF USER INPUTS ################
        
        # Get setup file and shape model
        # self.path_to_shape_model_file = "shape_models/1D_square.stl"
        # self.path_to_setup_file = "model_setups/John_Spencer_default_model_parameters.json"

        # self.path_to_shape_model_file = "private/Lucy/Dinkinesh/Dinkinesh.stl"
        # self.path_to_setup_file = "private/Lucy/Dinkinesh/Dinkinesh_parameters.json"

        self.path_to_shape_model_file = "shape_models/67P_not_to_scale_16670_facets.stl"
        self.path_to_setup_file = "private/Lucy/Dinkinesh/Dinkinesh_parameters.json"

        ################ GENERAL ################
        self.silent_mode = False
        self.remote = True
        n_jobs = 16 # Default 

        ################ PERFORMANCE ################
        # Number of parallel jobs to run
        # Use -1 to use all available cores (USE WITH CAUTION on shared computing facilities!)
        # Default to n_jobs or max available cores, whichever is smaller
        self.n_jobs = min(n_jobs, cpu_count())  
        self.chunk_size = 100  # Number of facets to process per parallel task

        ################ MODELLING ################
        self.convergence_method = 'mean' # 'mean' or 'max'. Mean is recommended for most cases, max is best for investigating permanently shadowed regions.
        self.include_shadowing = True # Recommended to keep this as True for most cases
        self.n_scatters = 2 # Set to 0 to disable scattering. 1 or 2 is recommended for most cases. 3 is almost always unncecessary.
        self.include_self_heating = False
        self.apply_roughness = True

        ################ PLOTTING ################
        self.plotted_facet_index = 1220 # Index of facet to plot
        self.plot_insolation_curve = False
        self.plot_insolation_curve = False 
        self.plot_initial_temp_histogram = False
        self.plot_final_day_all_layers_temp_distribution = False
        self.plot_final_day_all_layers_temp_distribution = False
        self.plot_energy_terms = False # NOTE: You must set config.calculate_energy_terms to True to plot energy terms
        self.plot_final_day_comparison = False

        ################ ANIMATIONS ################        BUG: If using 2 animations, the second one doesn't work (pyvista segmenation fault)
        self.animate_roughness_model = False
        self.animate_shadowing = False
        self.animate_secondary_radiation_view_factors = False
        self.animate_secondary_contributions = False
        self.animate_final_day_temp_distribution = True

        ################ DEBUGGING ################
        self.calculate_energy_terms = False         # NOTE: Numba must be disabled if calculating energy terms - model will run much slower

        ################ END OF USER INPUTS ################

    def validate_jobs(self):
        """
        Validate the number of jobs requested and issue warnings if necessary.
        Returns the actual number of jobs to use.
        """
        available_cores = cpu_count()
        
        if self.n_jobs == -1:
            warnings.warn(
                "Using all available cores (-1). This should be used with caution on shared "
                "computing facilities as it may impact other users. Consider setting a specific "
                f"number of cores instead. Will use {available_cores} cores.",
                UserWarning
            )
            return available_cores
            
        if self.n_jobs > available_cores:
            warnings.warn(
                f"Requested {self.n_jobs} cores but only {available_cores} are available. "
                f"Reducing to {available_cores} cores.",
                UserWarning
            )
            return available_cores
            
        if self.n_jobs < 1:
            warnings.warn(
                f"Invalid number of jobs ({self.n_jobs}). Must be >= 1. Setting to 1.",
                UserWarning
            )
            return 1
            
        return self.n_jobs

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
        
        # Compute unit vector from ra and dec
        ra_radians = np.radians(self.ra_degrees)
        dec_radians = np.radians(self.dec_degrees)
        self.rotation_axis = np.array([np.cos(ra_radians) * np.cos(dec_radians), np.sin(ra_radians) * np.cos(dec_radians), np.sin(dec_radians)])

class Facet:
    def __init__(self, normal, vertices, timesteps_per_day, max_days, n_layers, calculate_energy_terms):
        self.normal = np.array(normal)
        self.vertices = np.array(vertices)
        self.area = self.calculate_area(vertices)
        self.position = np.mean(vertices, axis=0)
        self.visible_facets = []

    def set_dynamic_arrays(self, length):
        self.secondary_radiation_view_factors = np.zeros(length)    

    @staticmethod
    def calculate_area(vertices):
        # Implement area calculation based on vertices
        v0, v1, v2 = vertices
        return np.linalg.norm(np.cross(v1-v0, v2-v0)) / 2

class ThermalData:
    def __init__(self, n_facets, timesteps_per_day, n_layers, max_days, calculate_energy_terms):
        self.temperatures = np.zeros((n_facets, timesteps_per_day * max_days, n_layers), dtype=np.float64) # Possibly change to float32 to save memory
        self.insolation = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        self.visible_facets = [np.array([], dtype=np.int64) for _ in range(n_facets)]
        self.secondary_radiation_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]

        self.calculate_energy_terms = calculate_energy_terms

        if calculate_energy_terms:
            self.insolation_energy = np.zeros((n_facets, timesteps_per_day * max_days))
            self.re_emitted_energy = np.zeros((n_facets, timesteps_per_day * max_days))
            self.surface_energy_change = np.zeros((n_facets, timesteps_per_day * max_days))
            self.conducted_energy = np.zeros((n_facets, timesteps_per_day * max_days))
            self.unphysical_energy_loss = np.zeros((n_facets, timesteps_per_day * max_days))
            
    def set_visible_facets(self, visible_facets):
        self.visible_facets = [np.array(facets, dtype=np.int64) for facets in visible_facets]

    def set_secondary_radiation_view_factors(self, view_factors):
        self.secondary_radiation_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]

def conditional_print(silent_mode, message):
    if not silent_mode:
        print(message)

def conditional_tqdm(iterable, silent_mode, **kwargs):
    if silent_mode:
        return iterable
    else:
        return tqdm(iterable, **kwargs)

def read_shape_model(filename, timesteps_per_day, n_layers, max_days, calculate_energy_terms):
    ''' 
    This function reads in the shape model of the body from a .stl file and return an array of facets, each with its own area, position, and normal vector.

    Ensure that the .stl file is saved in ASCII format, and that the file is in the same directory as this script. Additionally, ensure that the model dimensions are in meters and that the normal vectors are pointing outwards from the body. An easy way to convert the file is to open it in Blender and export it as an ASCII .stl file.

    This function will give an error if the file is not in the correct format, or if the file is not found.
    '''
    
    # Check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} was not found.")
    
    try:
        # Try reading as ASCII first
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        shape_model = []
        for i in range(len(lines)):
            if lines[i].strip().startswith('facet normal'):
                normal = np.array([float(n) for n in lines[i].strip().split()[2:]])
                vertex1 = np.array([float(v) for v in lines[i+2].strip().split()[1:]])
                vertex2 = np.array([float(v) for v in lines[i+3].strip().split()[1:]])
                vertex3 = np.array([float(v) for v in lines[i+4].strip().split()[1:]])
                facet = Facet(normal, [vertex1, vertex2, vertex3], timesteps_per_day, max_days, n_layers, calculate_energy_terms)
                shape_model.append(facet)
    except UnicodeDecodeError:
        # If ASCII reading fails, try binary
        stl_mesh = mesh.Mesh.from_file(filename)
        shape_model = []
        for i in range(len(stl_mesh.vectors)):
            normal = stl_mesh.normals[i]
            vertices = stl_mesh.vectors[i]
            facet = Facet(normal, vertices, timesteps_per_day, max_days, n_layers, calculate_energy_terms)
            shape_model.append(facet)

    return shape_model

def save_shape_model(shape_model, filename, config):
    """
    Save the shape model to an ASCII STL file.
    """
    with open(filename, 'w') as f:
        f.write("solid model\n")
        for facet in shape_model:
            f.write(f"facet normal {' '.join(map(str, facet.normal))}\n")
            f.write("  outer loop\n")
            for vertex in facet.vertices:
                f.write(f"    vertex {' '.join(map(str, vertex))}\n")
            f.write("  endloop\n")
            f.write("endfacet\n")
        f.write("endsolid model\n")
    
    conditional_print(config.silent_mode, f"Shape model saved to {filename}")

def check_remote_and_animate(remote, path_to_shape_model_file, plotted_variable_array, rotation_axis, 
                           sunlight_direction, timesteps_per_day, solar_distance_au, rotation_period_hr, 
                           **kwargs):
    """
    Handles animation logic based on whether remote mode is enabled.
    If remote is False, calls animate_model function directly.
    If remote is True, saves the arguments into .npz and .json files.
    """
    if not remote:
        # Local mode: call animate_model as usual
        animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction,
                     timesteps_per_day, solar_distance_au, rotation_period_hr, **kwargs)
    else:
        # Remote mode: create a directory to save the animation parameters
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"outputs/remote_outputs/animation_outputs_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Save numpy arrays to NPZ file
        numpy_arrays = {
            'plotted_variable_array': plotted_variable_array,
            'rotation_axis': rotation_axis,
            'sunlight_direction': sunlight_direction
        }
        npz_file = os.path.join(output_dir, 'animation_params.npz')
        np.savez_compressed(npz_file, **numpy_arrays)

        # Save other parameters to JSON file
        json_params = {
            'args': [
                path_to_shape_model_file,  # First argument is always the shape model file
                None,  # Placeholder for plotted_variable_array (stored in NPZ)
            ],
            'kwargs': {
                # Required positional arguments stored as kwargs for easier handling
                'timesteps_per_day': timesteps_per_day,
                'solar_distance_au': solar_distance_au,
                'rotation_period_hr': rotation_period_hr,
                'rotation_axis': None,  # Placeholder for rotation_axis (stored in NPZ)
                'sunlight_direction': None,  # Placeholder for sunlight_direction (stored in NPZ)
                # Additional keyword arguments
                **kwargs
            }
        }

        json_file = os.path.join(output_dir, 'animation_params.json')
        with open(json_file, 'w') as f:
            json.dump(json_params, f, indent=2)

        print(f"Animation parameters saved in:\nJSON: {json_file}\nNPZ: {npz_file}")


def apply_roughness(shape_model, simulation, config, subdivision_levels=5, displacement_factors=None):
    """
    Apply roughness to the shape model using iterative sub-facet division and displacement.
    
    Parameters:
    - shape_model: List of Facet objects
    - simulation: Simulation object
    - subdivision_levels: Number of times to perform the subdivision and adjustment process
    - displacement_factors: List of displacement factors for each subdivision level
    
    Returns:
    - new_shape_model: List of new Facet objects with applied roughness
    """
    
    if displacement_factors is None:
        displacement_factors = [0.2] * subdivision_levels
    elif len(displacement_factors) != subdivision_levels:
        raise ValueError(f"The number of displacement factors ({len(displacement_factors)}) must match the number of subdivision levels ({subdivision_levels})")
    
    def get_vertex_id(vertex):
        return tuple(np.round(vertex, decimals=6))  # Round to avoid float precision issues

    def midpoint_displacement(v1, v2, max_displacement):
        mid = (v1 + v2) / 2
        displacement = np.random.uniform(-max_displacement, max_displacement)
        normal = np.cross(v2 - v1, np.random.randn(3))
        normal /= np.linalg.norm(normal)
        return mid + displacement * normal

    def subdivide_triangle(vertices, max_displacement, vertex_dict):
        v1, v2, v3 = vertices
        
        m1_id = get_vertex_id((v1 + v2) / 2)
        m2_id = get_vertex_id((v2 + v3) / 2)
        m3_id = get_vertex_id((v3 + v1) / 2)
        
        if m1_id in vertex_dict:
            m1 = vertex_dict[m1_id]
        else:
            m1 = midpoint_displacement(v1, v2, max_displacement)
            vertex_dict[m1_id] = m1

        if m2_id in vertex_dict:
            m2 = vertex_dict[m2_id]
        else:
            m2 = midpoint_displacement(v2, v3, max_displacement)
            vertex_dict[m2_id] = m2

        if m3_id in vertex_dict:
            m3 = vertex_dict[m3_id]
        else:
            m3 = midpoint_displacement(v3, v1, max_displacement)
            vertex_dict[m3_id] = m3
        
        return [
            [v1, m1, m3],
            [m1, v2, m2],
            [m3, m2, v3],
            [m1, m2, m3]
        ]
    
    for level in range(subdivision_levels):
        new_shape_model = []
        vertex_dict = {}  # Reset vertex_dict for each level
        
        # First, add all current vertices to the vertex_dict
        for facet in shape_model:
            for vertex in facet.vertices:
                vertex_id = get_vertex_id(vertex)
                if vertex_id not in vertex_dict:
                    vertex_dict[vertex_id] = vertex

        for facet in shape_model:
            # Calculate max edge length
            edges = [np.linalg.norm(facet.vertices[i] - facet.vertices[(i+1)%3]) for i in range(3)]
            max_edge_length = max(edges)
            max_displacement = max_edge_length * displacement_factors[level]
            
            # Use the vertex_dict to get the potentially updated vertices
            current_vertices = [vertex_dict[get_vertex_id(v)] for v in facet.vertices]
            
            subdivided = subdivide_triangle(current_vertices, max_displacement, vertex_dict)
            
            for sub_vertices in subdivided:
                # Calculate sub-facet properties
                sub_normal = np.cross(sub_vertices[1] - sub_vertices[0], sub_vertices[2] - sub_vertices[0])
                sub_normal /= np.linalg.norm(sub_normal)
                sub_position = np.mean(sub_vertices, axis=0)
                
                # Create new Facet object
                new_facet = Facet(sub_normal, sub_vertices, simulation.timesteps_per_day, simulation.max_days, simulation.n_layers, config.calculate_energy_terms)
                new_shape_model.append(new_facet)
        
        # Update shape_model for the next iteration
        shape_model = new_shape_model
    
    return new_shape_model

@jit(nopython=True)
def calculate_visible_facets(positions, normals):
    ''' 
    This function calculates the visible (test) facets from each subject facet. It calculates the angle between the normal vector of each facet and the line of sight to every other facet. It returns the indices of the visible facets.
    
    NB: This doesn't account for partial shadowing (e.g. a facet may be only partially covered by the shadow cast by another facet) - more of an issue for low facet count models. Could add subdivision option to the model for better partial shadowing, but probably best to just use higher facet count models.

    NOTE: Idea for improvement - instaed of geometrically comparing every facet - do this only for nearby facets, then use tracing of evenly distributed rays to determine shadowing 'globe/sphere' for the rest of the facets.
    '''
    n_facets = len(positions)
    potentially_visible_indices = [np.empty(0, dtype=np.int64) for _ in range(n_facets)]

    epsilon = 1e-10
    
    for i in range(n_facets):
        # Compute the relative positions of all facets from the current subject facet
        relative_positions = positions[i] - positions

        # Check if the facet is above the horizon
        above_horizon = np.sum(relative_positions * normals[i], axis=1) < epsilon
        
        # Check if the facet is facing towards the subject facet
        facing_towards = np.sum(-relative_positions * normals, axis=1) < epsilon
    
        # Combine the two conditions to determine if the facet is visible
        potentially_visible = above_horizon & facing_towards
    
        potentially_visible[i] = False  # Exclude self
        
        # Get the indices of the visible facets
        visible_indices = np.where(potentially_visible)[0]
        potentially_visible_indices[i] = visible_indices

    return potentially_visible_indices

@jit(nopython=True)
def eliminate_obstructed_facets(positions, shape_model_vertices, potentially_visible_facet_indices):
    n_facets = len(positions)
    unobstructed_facets = [np.empty(0, dtype=np.int64) for _ in range(n_facets)]

    for i in range(n_facets):
        potential_indices = potentially_visible_facet_indices[i]
        if len(potential_indices) == 0:
            continue

        subject_position = positions[i]
        test_positions = positions[potential_indices]
        ray_directions = test_positions - subject_position
        ray_directions = normalize_vectors(ray_directions)

        unobstructed = []
        for j, test_facet_index in enumerate(potential_indices):
            other_indices = potential_indices[potential_indices != test_facet_index]
            test_vertices = shape_model_vertices[other_indices]

            if len(test_vertices) == 0:
                unobstructed.append(test_facet_index)
                continue

            # Perform ray-triangle intersection test
            intersections, _ = rays_triangles_intersection(
                subject_position,
                ray_directions[j:j+1],  # Single ray direction
                test_vertices
            )

            # If no intersections, the facet is unobstructed
            if not np.any(intersections):
                unobstructed.append(test_facet_index)

        unobstructed_facets[i] = np.array(unobstructed, dtype=np.int64)

    return unobstructed_facets


# @jit(nopython=True)
# def calculate_visible_facets_chunk(positions, normals, start_idx, end_idx):
#     """Calculate visible facets for a chunk of the shape model."""
#     n_facets = len(positions)
#     chunk_size = end_idx - start_idx
#     potentially_visible_indices = [np.empty(0, dtype=np.int64) for _ in range(chunk_size)]
#     epsilon = 1e-10
    
#     for i in range(chunk_size):
#         actual_idx = i + start_idx
#         relative_positions = positions[actual_idx] - positions
#         above_horizon = np.sum(relative_positions * normals[actual_idx], axis=1) < epsilon
#         facing_towards = np.sum(-relative_positions * normals, axis=1) < epsilon
#         potentially_visible = above_horizon & facing_towards
#         potentially_visible[actual_idx] = False
#         visible_indices = np.where(potentially_visible)[0]
#         potentially_visible_indices[i] = visible_indices
        
#     return potentially_visible_indices

# def calculate_visible_facets_parallel(positions, normals, n_jobs, chunk_size):
#     """Parallel wrapper for calculate_visible_facets using joblib."""
#     n_facets = len(positions)
#     n_chunks = (n_facets + chunk_size - 1) // chunk_size
    
#     # Create chunks of work
#     chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_facets)) 
#              for i in range(n_chunks)]
    
#     # Process chunks in parallel using joblib
#     results = Parallel(n_jobs=n_jobs, verbose=1)(
#         delayed(calculate_visible_facets_chunk)(positions, normals, start_idx, end_idx)
#         for start_idx, end_idx in chunks
#     )
    
#     # Combine results
#     all_visible_indices = []
#     for chunk_result in results:
#         all_visible_indices.extend(chunk_result)
    
#     return all_visible_indices

# @jit(nopython=True)
# def eliminate_obstructed_facets_chunk(positions, shape_model_vertices, potentially_visible_facet_indices, 
#                                     start_idx, end_idx):
#     """Process a chunk of facets for obstruction elimination."""
#     chunk_size = end_idx - start_idx
#     unobstructed_facets = [np.empty(0, dtype=np.int64) for _ in range(chunk_size)]

#     for i in range(chunk_size):
#         actual_idx = i + start_idx
#         potential_indices = potentially_visible_facet_indices[actual_idx]
        
#         if len(potential_indices) == 0:
#             continue

#         subject_position = positions[actual_idx]
#         test_positions = positions[potential_indices]
#         ray_directions = test_positions - subject_position
#         ray_directions = normalize_vectors(ray_directions)

#         unobstructed = []
#         for j, test_facet_index in enumerate(potential_indices):
#             other_indices = potential_indices[potential_indices != test_facet_index]
#             test_vertices = shape_model_vertices[other_indices]

#             if len(test_vertices) == 0:
#                 unobstructed.append(test_facet_index)
#                 continue

#             intersections, _ = rays_triangles_intersection(
#                 subject_position,
#                 ray_directions[j:j+1],
#                 test_vertices
#             )

#             if not np.any(intersections):
#                 unobstructed.append(test_facet_index)

#         unobstructed_facets[i] = np.array(unobstructed, dtype=np.int64)

#     return unobstructed_facets

# def eliminate_obstructed_facets_parallel(positions, shape_model_vertices, potentially_visible_facet_indices,
#                                        n_jobs, chunk_size):
#     """Parallel wrapper for eliminate_obstructed_facets using joblib."""
#     n_facets = len(positions)
#     n_chunks = (n_facets + chunk_size - 1) // chunk_size
    
#     # Create chunks of work
#     chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_facets)) 
#              for i in range(n_chunks)]
    
#     # Process chunks in parallel using joblib
#     results = Parallel(n_jobs=n_jobs, verbose=1)(
#         delayed(eliminate_obstructed_facets_chunk)(
#             positions, shape_model_vertices, potentially_visible_facet_indices,
#             start_idx, end_idx
#         )
#         for start_idx, end_idx in chunks
#     )
    
#     # Combine results
#     all_unobstructed_facets = []
#     for chunk_result in results:
#         all_unobstructed_facets.extend(chunk_result)
    
#     return all_unobstructed_facets

def calculate_and_cache_visible_facets(silent_mode, shape_model, positions, normals, vertices, config):
    shape_model_hash = get_shape_model_hash(shape_model)
    visible_facets_filename = get_visible_facets_filename(shape_model_hash)

    if os.path.exists(visible_facets_filename):
        conditional_print(silent_mode, "Loading existing visible facets...")
        with np.load(visible_facets_filename, allow_pickle=True) as data:
            visible_indices = data['visible_indices']
        visible_indices = [np.array(indices) for indices in visible_indices]
    else:
        # Validate and get actual number of jobs to use
        actual_n_jobs = config.validate_jobs()
        conditional_print(silent_mode, f"Calculating visible facets using {actual_n_jobs} parallel jobs...")
        
        potentially_visible_indices = calculate_visible_facets(
            positions, normals, actual_n_jobs, config.chunk_size
        )
        
        conditional_print(silent_mode, "Eliminating obstructed facets...")
        visible_indices = eliminate_obstructed_facets(
            positions, vertices, potentially_visible_indices,
            actual_n_jobs, config.chunk_size
        )
        
        os.makedirs(os.path.dirname(visible_facets_filename), exist_ok=True)
        np.savez_compressed(visible_facets_filename, 
                          visible_indices=np.array(visible_indices, dtype=object))

    return visible_indices

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

@jit(nopython=True)
def normalize_vector(v):
    """Normalize a vector."""
    norm = np.sqrt(np.sum(v**2))
    return v / norm if norm != 0 else v

@jit(nopython=True)
def normalize_vectors(vectors):
    """Normalize an array of vectors."""
    norms = np.sqrt(np.sum(vectors**2, axis=1))
    return vectors / norms[:, np.newaxis]

@jit(nopython=True)
def random_points_in_triangle(v0, v1, v2, n):
    r1 = np.sqrt(np.random.rand(n))
    r2 = np.random.rand(n)
    return (1 - r1)[:, np.newaxis] * v0 + (r1 * (1 - r2))[:, np.newaxis] * v1 + (r1 * r2)[:, np.newaxis] * v2

@jit(nopython=True)
def calculate_view_factors(subject_vertices, subject_normal, subject_area, test_vertices, test_areas, n_rays):
    '''
    Calculate view factors between subject and test vertices. The view factors are calculated by firing rays from the subject vertices to the test vertices and checking if the rays intersect with the test vertices. The view factor is calculated as the number of rays that intersect with the test vertices divided by the total number of rays fired from the subject vertices.
    '''

    ray_sources = random_points_in_triangle(subject_vertices[0], subject_vertices[1], subject_vertices[2], n_rays)
    
    ray_directions = np.random.randn(n_rays, 3)
    for i in range(n_rays):
        ray_directions[i] = normalize_vector(ray_directions[i])
    
    valid_directions = np.zeros(n_rays, dtype=np.bool_)
    for i in range(n_rays):
        valid_directions[i] = np.dot(ray_directions[i], subject_normal) > 0
    
    valid_count = np.sum(valid_directions)
    while valid_count < n_rays:
        additional_rays_needed = n_rays - valid_count
        additional_ray_directions = np.random.randn(additional_rays_needed, 3)
        for i in range(additional_rays_needed):
            additional_ray_directions[i] = normalize_vector(additional_ray_directions[i])
        
        for i in range(additional_rays_needed):
            if np.dot(additional_ray_directions[i], subject_normal) > 0:
                ray_directions[valid_count] = additional_ray_directions[i]
                valid_count += 1
                if valid_count == n_rays:
                    break
    
    intersections = np.zeros((n_rays, len(test_vertices)), dtype=np.bool_)
    for i in range(n_rays):
        ray_origin = ray_sources[i]
        ray_dir = ray_directions[i]
        intersect, _ = rays_triangles_intersection(ray_origin, ray_dir.reshape(1, 3), test_vertices)
        intersections[i] = intersect[0]
    
    view_factors = np.sum(intersections, axis=0).astype(np.float64) / n_rays * subject_area / test_areas
    view_factors = view_factors * (test_areas / subject_area)
    
    return view_factors

# def calculate_shape_model_view_factors(shape_model, thermal_data, simulation, config, n_rays=10000):
#     all_view_factors = []
    
#     shape_model_hash = get_shape_model_hash(shape_model)
#     view_factors_filename = get_view_factors_filename(shape_model_hash)

#     if os.path.exists(view_factors_filename):
#         with np.load(view_factors_filename, allow_pickle=True) as data:
#             conditional_print(config.silent_mode,  "Loading existing view factors...")
#             all_view_factors = list(data['view_factors'])
#     else:
#         conditional_print(config.silent_mode,  "No existing view factors found.")
#         all_view_factors = []
    
#     if not all_view_factors:
#         conditional_print(simulation, "Calculating new view factors...")
#         for i in conditional_tqdm(range(len(shape_model)), config.silent_mode, desc="Calculating secondary radiation view factors"):
#             visible_indices = thermal_data.visible_facets[i]

#             subject_vertices = shape_model[i].vertices
#             subject_area = shape_model[i].area
#             subject_normal = shape_model[i].normal
#             test_vertices = np.array([shape_model[j].vertices for j in visible_indices]).reshape(-1, 3, 3)
#             test_areas = np.array([shape_model[j].area for j in visible_indices])

#             view_factors = calculate_view_factors(subject_vertices, subject_normal, subject_area, test_vertices, test_areas, n_rays)

#             if np.any(np.isnan(view_factors)) or np.any(np.isinf(view_factors)):
#                 conditional_print(config.silent_mode,  f"Warning: Invalid view factor for facet {i}")
#                 conditional_print(config.silent_mode,  f"View factors: {view_factors}")
#                 conditional_print(config.silent_mode,  f"Visible facets: {visible_indices}")
#             all_view_factors.append(view_factors)

#         # Save the calculated view factors
#         os.makedirs("view_factors", exist_ok=True)
#         np.savez_compressed(view_factors_filename, view_factors=np.array(all_view_factors, dtype=object))

#     return all_view_factors

def process_view_factors_chunk(shape_model, thermal_data, start_idx, end_idx, n_rays):
    """
    Process a chunk of facets for view factor calculations.
    Returns a list of view factors and any warnings generated.
    """
    chunk_view_factors = []
    warnings = []
    
    for i in range(start_idx, end_idx):
        visible_indices = thermal_data.visible_facets[i]
        
        subject_vertices = shape_model[i].vertices
        subject_area = shape_model[i].area
        subject_normal = shape_model[i].normal
        test_vertices = np.array([shape_model[j].vertices for j in visible_indices]).reshape(-1, 3, 3)
        test_areas = np.array([shape_model[j].area for j in visible_indices])

        view_factors = calculate_view_factors(
            subject_vertices, subject_normal, subject_area, 
            test_vertices, test_areas, n_rays
        )

        if np.any(np.isnan(view_factors)) or np.any(np.isinf(view_factors)):
            warnings.append({
                'facet': i,
                'view_factors': view_factors.copy(),
                'visible_facets': visible_indices.copy()
            })
            
        chunk_view_factors.append(view_factors)
    
    return chunk_view_factors, warnings

def calculate_shape_model_view_factors_parallel(shape_model, thermal_data, simulation, config, n_rays=10000):
    """
    Parallel version of calculate_shape_model_view_factors using joblib.
    Now includes detailed progress tracking.
    """
    shape_model_hash = get_shape_model_hash(shape_model)
    view_factors_filename = get_view_factors_filename(shape_model_hash)

    # Try to load existing view factors first
    if os.path.exists(view_factors_filename):
        with np.load(view_factors_filename, allow_pickle=True) as data:
            conditional_print(config.silent_mode, "Loading existing view factors...")
            return list(data['view_factors'])

    conditional_print(config.silent_mode, "No existing view factors found.")
    
    # Validate and get actual number of jobs to use
    actual_n_jobs = config.validate_jobs()
    n_facets = len(shape_model)
    
    # Calculate chunk size if not specified
    if config.chunk_size <= 0:
        # Aim for at least 4 chunks per core
        config.chunk_size = max(1, n_facets // (actual_n_jobs * 4))
    
    # Create chunks
    n_chunks = (n_facets + config.chunk_size - 1) // config.chunk_size
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
             for i in range(n_chunks)]
    
    # Calculate total workload for progress estimation
    total_facets = len(shape_model)
    
    conditional_print(config.silent_mode, 
                     f"\nCalculating view factors using {actual_n_jobs} parallel jobs:")
    conditional_print(config.silent_mode,
                     f"Total facets: {total_facets:,}")
    conditional_print(config.silent_mode,
                     f"Chunk size: {config.chunk_size:,}")
    conditional_print(config.silent_mode,
                     f"Number of chunks: {n_chunks:,}")
    
    # Create progress bar
    if not config.silent_mode:
        from tqdm import tqdm
        pbar = tqdm(total=total_facets, 
                   desc="Processing facets",
                   unit="facets",
                   unit_scale=True,
                   bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} facets [{elapsed}<{remaining}, {rate_fmt}]")
    
    def update_progress(result):
        """Callback function to update progress bar"""
        if not config.silent_mode:
            chunk_size = len(result[0])  # Get size of completed chunk
            pbar.update(chunk_size)
    
    # Process chunks in parallel with progress tracking
    try:
        results = Parallel(n_jobs=actual_n_jobs, verbose=0, backend='loky')(
            delayed(process_view_factors_chunk)(
                shape_model, thermal_data, start_idx, end_idx, n_rays
            ) for start_idx, end_idx in chunks
        )
        
        # Combine results and collect warnings
        all_view_factors = []
        all_warnings = []
        
        for chunk_view_factors, chunk_warnings in results:
            all_view_factors.extend(chunk_view_factors)
            all_warnings.extend(chunk_warnings)
        
        # Close progress bar
        if not config.silent_mode:
            pbar.close()
        
        # Report any warnings
        if all_warnings and not config.silent_mode:
            conditional_print(config.silent_mode, "\nWarnings during view factor calculation:")
            for warning in all_warnings:
                conditional_print(config.silent_mode, 
                                f"Warning: Invalid view factor for facet {warning['facet']}")
                conditional_print(config.silent_mode, f"View factors: {warning['view_factors']}")
                conditional_print(config.silent_mode, f"Visible facets: {warning['visible_facets']}")
        
        # Save the calculated view factors
        os.makedirs("view_factors", exist_ok=True)
        np.savez_compressed(view_factors_filename, 
                           view_factors=np.array(all_view_factors, dtype=object))
        
        return all_view_factors
        
    except KeyboardInterrupt:
        if not config.silent_mode:
            pbar.close()
        print("\nCalculation interrupted by user. Progress was not saved.")
        raise
    except Exception as e:
        if not config.silent_mode:
            pbar.close()
        print(f"\nError during calculation: {str(e)}")
        raise
    finally:
        if not config.silent_mode:
            pbar.close()

def get_shape_model_hash(shape_model):
    # Create a hash based on the shape model data
    model_data = []
    for facet in shape_model:
        # Flatten the vertices and concatenate with normal and area
        facet_data = np.concatenate([facet.vertices.flatten(), facet.normal, [facet.area]])
        model_data.append(facet_data)
    
    # Convert to a 2D numpy array
    model_data_array = np.array(model_data)
    
    # Create a hash of the array
    return hashlib.sha256(model_data_array.tobytes()).hexdigest()

def get_view_factors_filename(shape_model_hash):
    return f"view_factors/view_factors_{shape_model_hash}.npz"

def get_visible_facets_filename(shape_model_hash):
    return f"visible_facets/visible_facets_{shape_model_hash}.npz"

def calculate_black_body_temp(insolation, emissivity, albedo):
    """
    Calculate the black body temperature based on insolation and surface properties.
    
    Args:
    insolation (float): Incoming solar radiation in W/m^2
    emissivity (float): Surface emissivity
    albedo (float): Surface albedo
    
    Returns:
    float: Black body temperature in Kelvin
    """
    stefan_boltzmann = 5.67e-8  # Stefan-Boltzmann constant in W/(m^2·K^4)
    
    # Calculate temperature using the Stefan-Boltzmann law
    temperature = (insolation / (emissivity * stefan_boltzmann)) ** 0.25
    
    return temperature

@jit(nopython=True)
def calculate_shadowing(subject_positions, sunlight_directions, shape_model_vertices, visible_facet_indices):
    '''
    This function calculates whether a facet is in shadow at a given time step. It cycles through all visible facets and passes their vertices to rays_triangles_intersections which determines whether they fall on the sunlight direction vector (starting at the facet position). If they do, the facet is in shadow. 
    
    It returns the illumination factor for the facet at that time step. 0 if the facet is in shadow, 1 if it is not.
    '''

    # Ensure triangles_vertices is an array of shape (m, 3, 3)
    triangles_vertices = shape_model_vertices[visible_facet_indices]

    # Call the intersection function
    intersections, t_values = rays_triangles_intersection(
        subject_positions,
        sunlight_directions,
        triangles_vertices
    )

    # Check for any intersection
    if intersections.any():
        return 0  # The facet is in shadow
        
    return 1 # The facet is not in shadow

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

# def apply_scattering(thermal_data, shape_model, simulation, config):
#     original_insolation = thermal_data.insolation.copy()
#     total_scattered_light = np.zeros_like(original_insolation)
    
#     for iteration in range(config.n_scatters):
#         conditional_print(config.silent_mode,  f"Scattering iteration {iteration + 1} of {config.n_scatters}...")
        
#         if iteration == 0:
#             input_light = original_insolation
#         else:
#             input_light = scattered_light
        
#         scattered_light = np.zeros_like(original_insolation)

#         for i, facet in enumerate(shape_model):
#             for t in range(simulation.timesteps_per_day):
#                 current_light = input_light[i, t]
                
#                 if current_light > 0:
#                     visible_facets = thermal_data.visible_facets[i]
#                     view_factors = thermal_data.secondary_radiation_view_factors[i]
                    
#                     for vf_index, vf in zip(visible_facets, view_factors):
#                         scattered_light[vf_index, t] += current_light * vf * (simulation.albedo) / np.pi

#         total_scattered_light += scattered_light
    
#     thermal_data.insolation = original_insolation + total_scattered_light
    
#     return thermal_data

def process_scattering_chunk(start_idx, end_idx, input_light, visible_facets_list, 
                           view_factors_list, timesteps_per_day, albedo):
    """
    Process a chunk of facets for scattering calculation.
    Only accepts raw numpy arrays and basic Python types.
    """
    chunk_scattered_light = np.zeros_like(input_light)
    
    for i in range(start_idx, end_idx):
        visible_facets = visible_facets_list[i]
        view_factors = view_factors_list[i]
        
        for t in range(timesteps_per_day):
            current_light = input_light[i, t]
            
            if current_light > 0:
                for vf_index, vf in zip(visible_facets, view_factors):
                    chunk_scattered_light[vf_index, t] += current_light * vf * albedo / np.pi
                    
    return chunk_scattered_light

def apply_scattering_parallel(thermal_data, shape_model, simulation, config):
    """
    Parallel version of apply_scattering that only passes numpy arrays between processes.
    """
    original_insolation = thermal_data.insolation.copy()
    total_scattered_light = np.zeros_like(original_insolation)
    n_facets = len(shape_model)
    
    # Convert Numba typed lists to regular Python lists of numpy arrays
    visible_facets_list = [np.array(x) for x in thermal_data.visible_facets]
    view_factors_list = [np.array(x) for x in thermal_data.secondary_radiation_view_factors]
    
    # Validate and get actual number of jobs to use
    actual_n_jobs = config.validate_jobs()
    
    # Calculate chunk size if not specified
    if config.chunk_size <= 0:
        # Aim for at least 4 chunks per core
        config.chunk_size = max(1, n_facets // (actual_n_jobs * 4))
    
    # Create chunks
    n_chunks = (n_facets + config.chunk_size - 1) // config.chunk_size
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
             for i in range(n_chunks)]
    
    for iteration in range(config.n_scatters):
        conditional_print(config.silent_mode, 
                        f"Scattering iteration {iteration + 1} of {config.n_scatters} "
                        f"using {actual_n_jobs} parallel jobs...")
        
        if iteration == 0:
            input_light = original_insolation
        else:
            input_light = scattered_light
        
        # Process chunks in parallel, passing only numpy arrays and basic Python types
        chunk_results = Parallel(n_jobs=actual_n_jobs, verbose=1)(
            delayed(process_scattering_chunk)(
                start_idx, end_idx,
                input_light,
                visible_facets_list,
                view_factors_list,
                simulation.timesteps_per_day,
                simulation.albedo
            )
            for start_idx, end_idx in chunks
        )
        
        # Combine results from all chunks
        scattered_light = np.zeros_like(original_insolation)
        for chunk_result in chunk_results:
            scattered_light += chunk_result
            
        total_scattered_light += scattered_light
    
    thermal_data.insolation = original_insolation + total_scattered_light
    return thermal_data

def calculate_insolation(thermal_data, shape_model, simulation, config):
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
    
    for i, facet in enumerate(conditional_tqdm(shape_model, config.silent_mode, desc="Calculating insolation")):
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

                if len(facet.visible_facets) != 0 and config.include_shadowing:
                    illumination_factor = calculate_shadowing(np.array(facet.position), np.array([rotated_sunlight_directions[t]]), shape_model_vertices, thermal_data.visible_facets[i])
                
                # Calculate insolation
                insolation = simulation.solar_luminosity * (1 - simulation.albedo) * illumination_factor * cos_zenith_angle / (4 * np.pi * simulation.solar_distance_m**2)
            else:
                insolation = 0
            
            thermal_data.insolation[i, t] = insolation

    if config.n_scatters > 0:
        conditional_print(config.silent_mode,  f"Applying light scattering with {config.n_scatters} iterations...")
        thermal_data = apply_scattering_parallel(thermal_data, shape_model, simulation, config)
        conditional_print(config.silent_mode,  "Light scattering applied.")

    return thermal_data

def calculate_initial_temperatures(thermal_data, silent_mode, emissivity, n_jobs=-1):
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
    results = Parallel(n_jobs=n_jobs)(delayed(process_facet)(thermal_data.insolation[i], emissivity, sigma) 
                                      for i in range(thermal_data.temperatures.shape[0])
    )

    conditional_print(silent_mode, f"Initial temperatures calculated for {thermal_data.temperatures.shape[0]} facets.")

    # Update the original shape_model with the results
    for i, temperature in conditional_tqdm(enumerate(results), silent_mode, total=len(results), desc='Saving temps'):
        thermal_data.temperatures[i, :, :] = temperature

    conditional_print(silent_mode, "Initial temperatures saved for all facets.")

    return thermal_data

@jit(nopython=True)
def calculate_secondary_radiation(temperatures, visible_facets, view_factors, self_heating_const):
    return self_heating_const * np.sum(temperatures[visible_facets]**4 * view_factors)

def export_results(shape_model_name, config, temperature_array):
    ''' 
    This function exports the final results of the model to be used in an instrument simulator. It creates a folder within /outputs with the shape model, model parameters, a plot of the temperature distribution, and final timestep temperatures.
    '''

    folder_name = f"{shape_model_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}" # Create new folder name
    os.makedirs(f"outputs/{folder_name}") # Create folder for results
    shape_mesh = mesh.Mesh.from_file(config.path_to_shape_model_file) # Load shape model
    os.system(f"cp {config.path_to_shape_model_file} outputs/{folder_name}") # Copy shape model .stl file to folder
    os.system(f"cp {config.path_to_setup_file} outputs/{folder_name}") # Copy model parameters .json file to folder
    np.savetxt(f"outputs/{folder_name}/temperatures.csv", temperature_array, delimiter=',') # Save the final timestep temperatures to .csv file

    # Plot the temperature distribution for the final timestep and save it to the folder
    temp_output_file_path = f"outputs/{folder_name}/"

@jit(nopython=True)
def calculate_energy_terms(temperature, insolation, delta_t, emissivity, beaming_factor,
                           density, specific_heat_capacity, layer_thickness, thermal_conductivity,
                           timesteps_per_day, n_layers):
    energy_terms = np.zeros((len(temperature), timesteps_per_day, 5))
    for i in range(len(temperature)):
        for time_step in range(timesteps_per_day):
            energy_terms[i, time_step, 0] = insolation[i, time_step] * delta_t
            energy_terms[i, time_step, 1] = -emissivity * beaming_factor * 5.670374419e-8 * (temperature[i, time_step, 0]**4) * delta_t
            energy_terms[i, time_step, 2] = -density * specific_heat_capacity * layer_thickness * (temperature[i, (time_step + 1) % timesteps_per_day, 0] - temperature[i, time_step, 0])
            energy_terms[i, time_step, 3] = thermal_conductivity * delta_t * (temperature[i, time_step, 1] - temperature[i, time_step, 0]) / layer_thickness
            energy_terms[i, time_step, 4] = energy_terms[i, time_step, 0] + energy_terms[i, time_step, 1] + energy_terms[i, time_step, 2] + energy_terms[i, time_step, 3]
    return energy_terms

@jit(nopython=True)
def calculate_temperatures(temperature, insolation, visible_facets_list, view_factors_list, 
                           const1, const2, const3, self_heating_const, 
                           timesteps_per_day, n_layers, include_self_heating,
                           start_index, end_index):
    
    current_day_temperature = temperature[:, start_index:end_index, :].copy()
    day_length = end_index - start_index
    
    for time_step in range(day_length):
        prev_step = time_step - 1 if time_step > 0 else -1
        for i in range(len(current_day_temperature)):
            if time_step == 0:
                prev_temp = temperature[i, start_index - 1, 0] if start_index > 0 else current_day_temperature[i, 0, 0]
                prev_temp_layer1 = temperature[i, start_index - 1, 1] if start_index > 0 else current_day_temperature[i, 0, 1]
            else:
                prev_temp = current_day_temperature[i, prev_step, 0]
                prev_temp_layer1 = current_day_temperature[i, prev_step, 1]

            insolation_term = insolation[i, time_step] * const1
            re_emitted_radiation_term = -const2 * (prev_temp**4)
            
            secondary_radiation_term = 0.0
  
            if include_self_heating:
                secondary_radiation_term = calculate_secondary_radiation(current_day_temperature[:, prev_step, 0], visible_facets_list[i], view_factors_list[i], self_heating_const)
            
            conducted_heat_term = const3 * (prev_temp_layer1 - prev_temp)
            
            new_temp = (prev_temp + 
                        insolation_term + 
                        re_emitted_radiation_term + 
                        conducted_heat_term + 
                        secondary_radiation_term)

            current_day_temperature[i, time_step, 0] = new_temp
            
            # Update subsurface temperatures, excluding the deepest layer
            for layer in range(1, n_layers - 1):
                if time_step == 0:
                    prev_layer = temperature[i, start_index - 1, layer] if start_index > 0 else current_day_temperature[i, 0, layer]
                    prev_layer_plus = temperature[i, start_index - 1, layer + 1] if start_index > 0 else current_day_temperature[i, 0, layer + 1]
                    prev_layer_minus = temperature[i, start_index - 1, layer - 1] if start_index > 0 else current_day_temperature[i, 0, layer - 1]
                else:
                    prev_layer = current_day_temperature[i, prev_step, layer]
                    prev_layer_plus = current_day_temperature[i, prev_step, layer + 1]
                    prev_layer_minus = current_day_temperature[i, prev_step, layer - 1]

                current_day_temperature[i, time_step, layer] = (
                    prev_layer + 
                    const3 * (prev_layer_plus - 
                              2 * prev_layer + 
                              prev_layer_minus)
                )
    
    return current_day_temperature

def thermophysical_body_model(thermal_data, shape_model, simulation, config):
    ''' 
    This is the main calculation function for the thermophysical body model. It calls the necessary functions to read in the shape model, set material and model properties, calculate 
    insolation and temperature arrays, and iterate until the model converges.
    '''

    convergence_error = simulation.convergence_target + 1
    day = 0 
    temperature_error = 0

    const1 = simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
    const2 = simulation.emissivity * simulation.beaming_factor * 5.67e-8 * simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
    const3 = simulation.thermal_diffusivity * simulation.delta_t / simulation.layer_thickness**2
    self_heating_const = 5.670374419e-8 * simulation.delta_t * simulation.emissivity / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity * np.pi)

    error_history = []

    # Set comparison temperatures to the initial temperatures of the first timestep (day 0, timestep 0)
    comparison_temps = thermal_data.temperatures[:, 0, 0].copy()

    while day < simulation.max_days and (day < simulation.min_days or convergence_error > simulation.convergence_target):
        current_day_start = day * simulation.timesteps_per_day
        current_day_end = (day + 1) * simulation.timesteps_per_day
        next_day_start = current_day_end
                        
        current_day_temperature = calculate_temperatures(
            thermal_data.temperatures,
            thermal_data.insolation,
            thermal_data.visible_facets,
            thermal_data.secondary_radiation_view_factors,
            const1, const2, const3, self_heating_const, 
            simulation.timesteps_per_day, simulation.n_layers,
            config.include_self_heating,
            current_day_start, current_day_end
        )

        thermal_data.temperatures[:, current_day_start:current_day_end, :] = current_day_temperature

        if config.calculate_energy_terms:
            energy_terms = calculate_energy_terms(
                current_day_temperature, 
                thermal_data.insolation, 
                simulation.delta_t, 
                simulation.emissivity,
                simulation.beaming_factor, 
                simulation.density, 
                simulation.specific_heat_capacity,
                simulation.layer_thickness, 
                simulation.thermal_conductivity,
                simulation.timesteps_per_day, 
                simulation.n_layers
            )

        # Check for invalid temperatures
        for i in range(thermal_data.temperatures.shape[0]):
            for time_step in range(simulation.timesteps_per_day):
                current_step = int(time_step + (day * simulation.timesteps_per_day))
                if np.isnan(current_day_temperature[i, time_step, 0]) or np.isinf(current_day_temperature[i, time_step, 0]) or current_day_temperature[i, time_step, 0] < 0 and not config.silent_mode:
                    conditional_print(config.silent_mode,  f"Ending run at timestep {current_step} due to facet {i} having a temperature of {current_day_temperature[i, time_step, 0]} K.\n Try increasing the number of time steps per day")

                    # Plot the energy terms for the facet
                    if config.calculate_energy_terms:
                        plt.plot(thermal_data.insolation_energy[i, :current_step], label="Insolation energy")
                        plt.plot(thermal_data.re_emitted_energy[i, :current_step], label="Re-emitted energy")
                        plt.plot(thermal_data.surface_energy_change[i, :current_step], label="Surface energy change")
                        plt.plot(thermal_data.conducted_energy[i, :current_step], label="Conducted energy")
                        plt.plot(thermal_data.unphysical_energy_loss[i, :current_step], label="Unphysical energy loss")
                        plt.legend()
                        plt.xlabel('Timestep')
                        plt.ylabel('Energy (J)')
                        plt.title(f'Energy terms for facet {i}')
                        plt.show()

                    # Plot the insolation curve for the facet
                    plt.plot(thermal_data.insolation[i])
                    plt.xlabel('Number of timesteps')
                    plt.ylabel('Insolation (W/m^2)')
                    plt.title(f'Insolation curve for facet {i}')
                    plt.show()

                    # Plot sub-surface temperatures for the facet
                    for layer in range(1, simulation.n_layers):
                        plt.plot(thermal_data.temperatures[i, :current_step+100, layer])
                    plt.xlabel('Number of timesteps')
                    plt.ylabel('Temperature (K)')
                    plt.title(f'Sub-surface temperature for facet {i}')
                    plt.legend([f"Layer {layer}" for layer in range(1, simulation.n_layers)])
                    plt.show()

                    # Create an array of 0s for all facets for all time steps in the day
                    facet_highlight_array = np.zeros((thermal_data.temperatures.shape[0], simulation.timesteps_per_day))
                    facet_highlight_array[i] = 1

                    check_remote_and_animate(config.remote, 
                                  config.path_to_shape_model_file, 
                                  facet_highlight_array, 
                                  simulation.rotation_axis, 
                                  simulation.sunlight_direction, 
                                  simulation.timesteps_per_day, 
                                  simulation.solar_distance_au,
                                  simulation.rotation_period_hours,
                                  colour_map='coolwarm', plot_title='Problematic facet', axis_label='Problem facet is red', animation_frames=200, save_animation=False, save_animation_name='problematic_facet_animation.gif', background_colour = 'black')

                    sys.exit()

        # Calculate convergence factor
        temperature_errors = np.abs(current_day_temperature[:, 0, 0] - comparison_temps)

        if config.convergence_method == 'mean':
            convergence_error = np.mean(temperature_errors)
        elif config.convergence_method == 'max':
            convergence_error = np.max(temperature_errors)
        else:
            raise ValueError("Invalid convergence_method. Use 'mean' or 'max'.")

        max_temperature_error = np.max(temperature_errors)
        mean_temperature_error = np.mean(temperature_errors) 

        conditional_print(config.silent_mode, f"Day: {day} | Mean Temperature error: {mean_temperature_error:.6f} K | Max Temp Error: {max_temperature_error:.6f} K")

        comparison_temps = current_day_temperature[:, 0, 0].copy()
        
        error_history.append(convergence_error)
        day += 1

    # Decrement the day counter
    day -= 1

    # Remove unused days from thermal_data
    thermal_data.temperatures = thermal_data.temperatures[:, :simulation.timesteps_per_day * (day+1), :]

    final_day_temperatures = thermal_data.temperatures[:, -simulation.timesteps_per_day:, 0]

    final_timestep_temperatures = thermal_data.temperatures[:, -1, 0]
    final_day_temperatures_all_layers = thermal_data.temperatures[:, -simulation.timesteps_per_day:, :]

    if convergence_error < simulation.convergence_target:
        conditional_print(config.silent_mode,  f"Convergence achieved after {day} days.")
        if config.calculate_energy_terms:
            for i in range(len(shape_model)):
                thermal_data.insolation_energy[i] = energy_terms[i, :, 0]
                thermal_data.re_emitted_energy[i] = energy_terms[i, :, 1]
                thermal_data.surface_energy_change[i] = energy_terms[i, :, 2]
                thermal_data.conducted_energy[i] = energy_terms[i, :, 3]
                thermal_data.unphysical_energy_loss[i] = energy_terms[i, :, 4]
    else:
        conditional_print(config.silent_mode,  f"Maximum days reached without achieving convergence.")
        conditional_print(config.silent_mode,  f"Final temperature error: {mean_temperature_error} K")
        conditional_print(config.silent_mode,  "Try increasing max_days or decreasing convergence_target.")
        if config.silent_mode:
            return

        if config.calculate_energy_terms:
            plt.plot(energy_terms[i, :, 0], label="Insolation energy")
            plt.plot(energy_terms[i, :, 1], label="Re-emitted energy")
            plt.plot(energy_terms[i, :, 2], label="Surface energy change")
            plt.plot(energy_terms[i, :, 3], label="Conducted energy")
            plt.plot(energy_terms[i, :, 4], label="Unphysical energy loss")
            plt.legend()
            plt.show()

    return final_day_temperatures, final_day_temperatures_all_layers, final_timestep_temperatures, day+1, temperature_error, max_temperature_error

def main(silent_mode=False):
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.
    '''

    # Load user configuration
    config = Config()

    # Load setup parameters from JSON file
    simulation = Simulation(config.path_to_setup_file)

    # Setup simulation
    shape_model = read_shape_model(config.path_to_shape_model_file, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)
    thermal_data = ThermalData(len(shape_model), simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)

    conditional_print(config.silent_mode,  f"\nDerived model parameters:")
    conditional_print(config.silent_mode,  f"Number of timesteps per day: {simulation.timesteps_per_day}")
    conditional_print(config.silent_mode,  f"Layer thickness: {simulation.layer_thickness} m")
    conditional_print(config.silent_mode,  f"Thermal inertia: {simulation.thermal_inertia} W m^-2 K^-1 s^0.5")
    conditional_print(config.silent_mode,  f"Skin depth: {simulation.skin_depth} m")
    conditional_print(config.silent_mode,  f"\n Number of facets: {len(shape_model)}")

    # Apply roughness to the shape model
    if config.apply_roughness:
        conditional_print(config.silent_mode,  f"Applying roughness to shape model. Original size: {len(shape_model)} facets.")
        shape_model = apply_roughness(shape_model, simulation, config, subdivision_levels=2, displacement_factors=[0.5, 0.1])
        conditional_print(config.silent_mode,  f"Roughness applied to shape model. New size: {len(shape_model)} facets.")
        # Save the shape model with roughness applied with a new filename
        config.path_to_shape_model_file = f"{config.path_to_shape_model_file[:-4]}_roughness_applied.stl"
        # Save it to a new file
        save_shape_model(shape_model, config.path_to_shape_model_file, config)

        # Read in the new shape model to ensure facets are updated
        shape_model = read_shape_model(config.path_to_shape_model_file, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)

        thermal_data = ThermalData(len(shape_model), simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)

        # Visualise shape model with roughness
        if config.animate_roughness_model: 
            check_remote_and_animate(config.remote, config.path_to_shape_model_file, 
                np.ones((len(shape_model), 1)),  # Make this a 2D array
                simulation.rotation_axis, 
                simulation.sunlight_direction, 
                1, 
                simulation.solar_distance_au,
                simulation.rotation_period_hours,
                colour_map='viridis', 
                plot_title='Roughness applied to shape model', 
                axis_label='Roughness Value', 
                animation_frames=1, 
                save_animation=False, 
                save_animation_name='roughness_animation.gif', 
                background_colour='black')

    # Setup the model
    positions = np.array([facet.position for facet in shape_model])
    normals = np.array([facet.normal for facet in shape_model])
    vertices = np.array([facet.vertices for facet in shape_model])
    
    visible_indices = calculate_and_cache_visible_facets(config.silent_mode, shape_model, positions, normals, vertices, config)

    thermal_data.set_visible_facets(visible_indices)

    for i, facet in enumerate(shape_model):
        facet.visible_facets = visible_indices[i]   
        
    if config.include_self_heating or config.n_scatters > 0:
        all_view_factors = calculate_shape_model_view_factors_parallel(shape_model, thermal_data, simulation, config, n_rays=10000)
        
        thermal_data.set_secondary_radiation_view_factors(all_view_factors)

        numba_view_factors = List()
        for view_factors in thermal_data.secondary_radiation_view_factors:
            numba_view_factors.append(np.array(view_factors, dtype=np.float64))
        thermal_data.secondary_radiation_view_factors = numba_view_factors
    else:
        # Create an empty Numba List for view factors when self-heating is not included
        numba_view_factors = List()
        for _ in range(len(shape_model)):
            numba_view_factors.append(np.array([], dtype=np.float64))
        thermal_data.secondary_radiation_view_factors = numba_view_factors

    thermal_data = calculate_insolation(thermal_data, shape_model, simulation, config)

    if config.plot_insolation_curve and not config.silent_mode:
        fig_insolation = plt.figure(figsize=(10, 6))
        conditional_print(config.silent_mode,  f"Preparing insolation curve plot.\n")
        
        if config.plotted_facet_index >= len(shape_model):
            conditional_print(config.silent_mode,  f"Facet index {config.plotted_facet_index} out of range. Please select a facet index between 0 and {len(shape_model) - 1}.")
        else:
            # Get the insolation data for the facet
            insolation_data = thermal_data.insolation[config.plotted_facet_index]
            
            roll_amount = 216
            # Roll the array to center the peak
            centered_insolation = np.roll(insolation_data, roll_amount)
            
            # Create x-axis in degrees
            degrees = np.linspace(0, 360, len(insolation_data), endpoint=False)
            
            # Create DataFrame for easy CSV export
            df = pd.DataFrame({
                'Rotation (degrees)': degrees,
                'Insolation (W/m^2)': centered_insolation
            })

            # Export to CSV
            output_dir = 'insolation_data'
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
            output_csv_path = os.path.join(output_dir, f'facet_{config.plotted_facet_index}.csv')
            df.to_csv(output_csv_path, index=False)
            conditional_print(config.silent_mode,  f"Insolation data exported to {output_csv_path}")
            
            # Plot the centered insolation curve
            plt.plot(degrees, centered_insolation)
            plt.xlabel('Rotation of the body (degrees)')
            plt.ylabel('Insolation (W/m^2)')
            plt.title(f'Insolation curve for facet {config.plotted_facet_index} over one full rotation of the body')
            plt.xlim(0, 360)
            plt.xticks(np.arange(0, 361, 45))  # Set x-axis ticks every 45 degrees
            
            conditional_print(config.silent_mode,  f"Facet {config.plotted_facet_index} insolation curve plotted with peak centered.")
            plt.show()

            sys.exit()

    if config.animate_shadowing:
        conditional_print(config.silent_mode,  f"Preparing shadowing visualisation.\n")

        check_remote_and_animate(config.remote, config.path_to_shape_model_file, 
                          thermal_data.insolation, 
                          simulation.rotation_axis, 
                          simulation.sunlight_direction, 
                          simulation.timesteps_per_day, 
                          simulation.solar_distance_au,
                          simulation.rotation_period_hours,
                          colour_map='binary_r', 
                          plot_title='Shadowing on the body', 
                          axis_label='Insolation (W/m^2)', 
                          animation_frames=200, 
                          save_animation=False, 
                          save_animation_name='shadowing_animation.gif', 
                          background_colour = 'black')

    conditional_print(config.silent_mode,  f"Calculating initial temperatures.\n")
    thermal_data = calculate_initial_temperatures(thermal_data, config.silent_mode, simulation.emissivity)

    if config.plot_initial_temp_histogram and not config.silent_mode:
        fig_histogram = plt.figure()
        initial_temperatures = [thermal_data.temperatures[i, 0, 0] for i in range(len(shape_model))]
        plt.hist(initial_temperatures, bins=20)
        plt.xlabel('Initial temperature (K)')
        plt.ylabel('Number of facets')
        plt.title('Initial temperature distribution of all facets')
        fig_histogram.show()

    numba_visible_facets = List()
    for facets in thermal_data.visible_facets:
        numba_visible_facets.append(np.array(facets, dtype=np.int64))
    thermal_data.visible_facets = numba_visible_facets

    if config.animate_secondary_radiation_view_factors:
        selected_facet = 1454  # Change this to the index of the facet you're interested in
        
        # Get the indices and view factors of contributing facets
        contributing_indices = thermal_data.visible_facets[selected_facet]
        contributing_view_factors = thermal_data.secondary_radiation_view_factors[selected_facet]
        
        # Create an array of zeros for all facets
        contribution_data = np.zeros(len(shape_model))
        
        # Set the view factors for the contributing facets
        contribution_data[contributing_indices] = 1
        contribution_data[selected_facet] = 0.5

        # Print contributing facets and their view factors
        conditional_print(config.silent_mode,  f"\nContributing facets for facet {selected_facet}:")
        for index, view_factors in zip(contributing_indices, contributing_view_factors):
            conditional_print(config.silent_mode,  f"Facet {index}: view factor = {view_factors:.6f}")
        conditional_print(config.silent_mode,  f"Total number of contributing facets: {len(contributing_indices)}")
        
        conditional_print(config.silent_mode,  f"Preparing visualization of contributing facets for facet {selected_facet}.")
        check_remote_and_animate(config.remote, config.path_to_shape_model_file, 
                      contribution_data[:, np.newaxis], 
                    simulation.rotation_axis, 
                    simulation.sunlight_direction, 
                    1,               
                    simulation.solar_distance_au,              
                    simulation.rotation_period_hours,
                    colour_map='viridis', 
                    plot_title=f'Contributing Facets for Facet {selected_facet}', 
                    axis_label='View Factors Value', 
                    animation_frames=1, 
                    save_animation=False, 
                    save_animation_name=f'contributing_facets_{selected_facet}.png', 
                    background_colour='black')
        
    if config.animate_secondary_contributions:
        # Calculate the sum of secondary radiation view factors for each facet
        secondary_radiation_sum = np.array([np.sum(view_factors) for view_factors in thermal_data.secondary_radiation_view_factors])

        conditional_print(config.silent_mode,  "Preparing secondary radiation visualization.")
        check_remote_and_animate(config.remote, config.path_to_shape_model_file, 
                    secondary_radiation_sum[:, np.newaxis], 
                    simulation.rotation_axis, 
                    simulation.sunlight_direction, 
                    1,               
                    simulation.solar_distance_au,              
                    simulation.rotation_period_hours,
                    colour_map='viridis', 
                    plot_title='Secondary Radiation Contribution', 
                    axis_label='Sum of View Factors', 
                    animation_frames=1, 
                    save_animation=False, 
                    save_animation_name='secondary_radiation.png', 
                    background_colour='black')

    conditional_print(config.silent_mode,  f"Running main simulation loop.\n")
    conditional_print(config.silent_mode,  f"Convergence target: {simulation.convergence_target} K with {config.convergence_method} convergence method.\n")
    start_time = time.time()
    final_day_temperatures, final_day_temperatures_all_layers, final_timestep_temperatures, day, temperature_error, max_temp_error = thermophysical_body_model(thermal_data, shape_model, simulation, config)
    end_time = time.time()
    execution_time = end_time - start_time

    if final_timestep_temperatures is not None:
        conditional_print(config.silent_mode,  f"Convergence target achieved after {day} days.")
        conditional_print(config.silent_mode,  f"Final temperature error: {temperature_error / len(shape_model)} K")
        conditional_print(config.silent_mode,  f"Max temperature error for any facet: {max_temp_error} K")
    else:
        conditional_print(config.silent_mode,  f"Model did not converge after {day} days.")
        conditional_print(config.silent_mode,  f"Final temperature error: {temperature_error / len(shape_model)} K")

    conditional_print(config.silent_mode,  f"Execution time: {execution_time} seconds")

    if config.plot_insolation_curve and not config.silent_mode:
        fig_temperature = plt.figure(figsize=(10, 6))
        conditional_print(config.silent_mode, f"Preparing temperature curve plot.\n")
        
        if config.plotted_facet_index >= len(shape_model):
            conditional_print(config.silent_mode, f"Facet index {config.plotted_facet_index} out of range. Please select a facet index between 0 and {len(shape_model) - 1}.")
        else:
            # Get the temp data for the facet
            temperature_data = final_day_temperatures[config.plotted_facet_index]
            
            # Calculate black body temperatures
            insolation_data = thermal_data.insolation[config.plotted_facet_index]
            black_body_temps = np.array([calculate_black_body_temp(ins, simulation.emissivity, simulation.albedo) for ins in insolation_data])
            
            # Find the index of the maximum black body temperature
            max_index = np.argmax(black_body_temps)
            
            # Calculate the roll amount to center the peak
            roll_amount = len(black_body_temps) // 2 - max_index
            
            # Roll the arrays to center the peak
            centered_temperature = np.roll(temperature_data, roll_amount)
            centered_black_body = np.roll(black_body_temps, roll_amount)
            
            # Create x-axis in degrees
            degrees = np.linspace(0, 360, len(temperature_data), endpoint=False)
            
            # Create DataFrame for easy CSV export
            df = pd.DataFrame({
                'Rotation (degrees)': degrees,
                'Temperature (K)': centered_temperature,
                'Black Body Temperature (K)': centered_black_body
            })

            # Export to CSV
            output_dir = 'temperature_data'
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
            output_csv_path = os.path.join(output_dir, f'facet_{config.plotted_facet_index}_with_black_body.csv')
            df.to_csv(output_csv_path, index=False)
            conditional_print(config.silent_mode, f"Temperature data exported to {output_csv_path}")
            
            # Plot the centered temperature curves
            plt.plot(degrees, centered_temperature, label='Model Temperature')
            plt.plot(degrees, centered_black_body, label='Black Body Temperature', linestyle='--')
            plt.xlabel('Rotation of the body (degrees)')
            plt.ylabel('Temperature (K)')
            plt.title(f'Temperature curves for facet {config.plotted_facet_index} over one full rotation of the body')
            plt.xlim(0, 360)
            plt.xticks(np.arange(0, 361, 90))  # Set x-axis ticks every 90 degrees
            plt.legend()
            
            conditional_print(config.silent_mode, f"Facet {config.plotted_facet_index} temperature curves plotted with peak centered.")
            plt.show()

            sys.exit()

    if config.plot_final_day_all_layers_temp_distribution and not config.silent_mode:
        fig_final_all_layers_temp_dist = plt.figure()
        plt.plot(final_day_temperatures_all_layers[config.plotted_facet_index])
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for all layers in facet')
        fig_final_all_layers_temp_dist.show()

    if config.plot_final_day_all_layers_temp_distribution and not config.silent_mode:
        fig_all_days_all_layers_temp_dist = plt.figure()
        plt.plot(thermal_data.temperatures[config.plotted_facet_index, :, :])
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature distribution for all layers in facet for the full run')
        fig_all_days_all_layers_temp_dist.show()

    if config.plot_energy_terms and not config.silent_mode:
        fig_energy_terms = plt.figure()
        plt.plot(shape_model[config.plotted_facet_index].unphysical_energy_loss[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Unphysical energy loss')
        plt.plot(shape_model[config.plotted_facet_index].insolation_energy[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Insolation energy')
        plt.plot(shape_model[config.plotted_facet_index].re_emitted_energy[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Re-emitted energy')
        plt.plot(-shape_model[config.plotted_facet_index].surface_energy_change[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Surface energy change')
        plt.plot(shape_model[config.plotted_facet_index].conducted_energy[(day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day], label='Conducted energy')
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('Energy (J)')
        plt.title('Energy terms for facet for the final day')
        fig_energy_terms.show()

    if config.animate_final_day_temp_distribution:
        conditional_print(config.silent_mode,  f"Preparing temperature animation.\n")

        check_remote_and_animate(config.remote, config.path_to_shape_model_file, 
                      final_day_temperatures, 
                      simulation.rotation_axis, 
                      simulation.sunlight_direction, 
                      simulation.timesteps_per_day,
                      simulation.solar_distance_au,              
                      simulation.rotation_period_hours,              
                      colour_map='coolwarm', 
                      plot_title='Temperature distribution', 
                      axis_label='Temperature (K)', 
                      animation_frames=200, 
                      save_animation=False, 
                      save_animation_name='temperature_animation.gif', 
                      background_colour = 'black')

    if config.plot_final_day_comparison and not config.silent_mode:
        conditional_print(config.silent_mode,  f"Saving final day temperatures for facet to CSV file.\n")
        np.savetxt("final_day_temperatures.csv", np.column_stack((np.linspace(0, 2 * np.pi, simulation.timesteps_per_day), final_day_temperatures[config.plotted_facet_index])), delimiter=',', header='Rotation angle (rad), Temperature (K)', comments='')

        thermprojrs_data = np.loadtxt("thermprojrs_data.csv", delimiter=',', skiprows=1)

        fig_model_comparison = plt.figure()
        plt.plot(thermprojrs_data[:, 0], thermprojrs_data[:, 1], label='Thermprojrs')
        plt.plot(np.linspace(0, 2 * np.pi, simulation.timesteps_per_day), final_day_temperatures[config.plotted_facet_index], label='This model')
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for facet')
        plt.legend()
        fig_model_comparison.show()

        x_original = np.linspace(0, 2 * np.pi, simulation.timesteps_per_day)
        x_new = np.linspace(0, 2 * np.pi, thermprojrs_data.shape[config.plotted_facet_index])

        interp_func = interp1d(x_new, thermprojrs_data[:, 1], kind='linear')
        thermprojrs_interpolated = interp_func(x_original)

        plt.plot(x_original, final_day_temperatures[config.plotted_facet_index] - thermprojrs_interpolated, label='This model')
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Temperature difference (K)')
        plt.title('Temperature difference between this model and Thermprojrs for facet')
        plt.legend()
        plt.show()

        np.savetxt("final_day.csv", np.column_stack((x_original, final_day_temperatures[config.plotted_facet_index])), delimiter=',', header='Rotation angle (rad), Temperature (K)', comments='')

    conditional_print(config.silent_mode,  f"Model run complete.\n")

# Call the main program to start execution
if __name__ == "__main__":
    main()
