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

NEXT STEPS:
1) Implement roughness/beaming effects (important to do soon)
    a) Just apply a function to the shape model (e.g. fractal roughness)
    b) Parameterise the roughness - e.g. fractal dimension, roughness scale
2) Find ways to make the model more robust
    a) Calculate n_layers and layer thickness based on thermal inertia (?) - these shouldn't be input by the user
    b) Look into ML/optimisation of finite difference method to avoid instability
    c) Look into gradient descent optimisation technique
3) Write a performance report for the model
4) Remove all NOTE and TODO comments from the code
5) GPU acceleration - look into PyCUDA and PyTorch
6) Reduce precision etc to save memory and speed up calculations
7) Reduce RAM usage by only storing the last day of temperatures for each facet - add option to save all temperatures (or larger number of days e.g. 5 days) for debugging (will limit max model size)
8) Create 'silent mode' flag so that the model can be run without printing to the console from an external script
9) BUG: John Spencer's model parameters crash it at 1 AU - Suspect something to do with timestep calculation. 
10) Add option to implement sublimation energy loss
11) Build in mesh converstion for binary .STL and .OBJ files
12) Create web interface for ease of use?
13) Integrate with JPL Horizons ephemeris to get real-time insolation data
14) Come up with a way of representing output data for many rotation axes and periods for mission planning | Do this and provide recommendations to MIRMIS team
15) Add filter visualisations to thermal model
    - Simulate retrievals for temperature based on instrument
16) Run setup steps just once for each shape model (e.g. calculate view factors, visible facets, etc.)
17) Add comparison tool to compare two runs of the model. (Separate script?)

EXTENSIONS: 
Binaries: Complex shading from non-rigid geometry (Could be a paper) 
Add temporary local heat sources e.g. jets
Horizontal conduction at high resolution

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
from animate_model import animate_model
from numba import jit, njit, float64, int64, boolean
from numba.typed import List
from joblib import Parallel, delayed
from stl import mesh
from tqdm import tqdm
from typing import Tuple
from scipy.interpolate import interp1d
from collections import defaultdict

class Simulation:
    def __init__(self, config_path, calculate_energy_terms):
        self.calculate_energy_terms = calculate_energy_terms
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
        
        # Calculation method flags
        self.include_self_heating = False # Default to not include self-heating
        self.include_scattering = False # Default to not include light scattering
        self.apply_roughness = False # Default to not include sublimation
 
        # Print out the configuration
        print(f"Configuration loaded from {config_path}")
        for key, value in config.items():
            print(f"{key}: {value}")
        
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

    def set_dynamic_arrays(self, length):
        self.visible_facets = np.zeros(length)
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

def read_shape_model(filename, timesteps_per_day, n_layers, max_days, calculate_energy_terms):
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
            facet = Facet(normal, [vertex1, vertex2, vertex3], timesteps_per_day, max_days, n_layers, calculate_energy_terms)
            shape_model.append(facet)

    for facet in shape_model:
        facet.set_dynamic_arrays(len(shape_model))
    
    return shape_model

def save_shape_model(shape_model, filename):
    """
    Save the shape model to an STL file.
    
    Parameters:
    - shape_model: List of Facet objects
    - filename: String, path to save the STL file
    """
    # Create a list to store the vertices and normals
    vertices = []
    normals = []
    
    # Iterate through all facets in the shape model
    for facet in shape_model:
        # Add the vertices of the current facet
        vertices.extend(facet.vertices)
        
        # Add the normal of the current facet (repeated 3 times for each vertex)
        normals.extend([facet.normal] * 3)
    
    # Convert lists to numpy arrays
    vertices = np.array(vertices)
    normals = np.array(normals)
    
    # Create the mesh
    facets = vertices.reshape(-1, 3, 3)
    mesh_data = mesh.Mesh(np.zeros(facets.shape[0], dtype=mesh.Mesh.dtype))
    
    # Set the vectors and normals from the data
    for i, facet in enumerate(facets):
        for j in range(3):
            mesh_data.vectors[i][j] = facet[j]
        mesh_data.normals[i] = normals[i * 3]
    
    # Save the mesh to file
    mesh_data.save(filename)
    
    print(f"Shape model saved to {filename}")


def apply_roughness(shape_model, simulation, subdivision_level=3, displacement_factor=0.4):
    """
    Apply roughness to the shape model using iterative sub-facet division and displacement.
    
    Parameters:
    - shape_model: List of Facet objects
    - subdivision_level: Number of times to perform the subdivision and adjustment process
    - displacement_factor: Maximum displacement as a fraction of the triangle's max edge length
    
    Returns:
    - new_shape_model: List of new Facet objects with applied roughness

    NOTE: Adjust so that different factors can be used at different levels of subdivision. 
    """
    
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
    
    for level in range(subdivision_level):
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
            max_displacement = max_edge_length * displacement_factor * (0.5 ** level)
            
            # Use the vertex_dict to get the potentially updated vertices
            current_vertices = [vertex_dict[get_vertex_id(v)] for v in facet.vertices]
            
            subdivided = subdivide_triangle(current_vertices, max_displacement, vertex_dict)
            
            for sub_vertices in subdivided:
                # Calculate sub-facet properties
                sub_normal = np.cross(sub_vertices[1] - sub_vertices[0], sub_vertices[2] - sub_vertices[0])
                sub_normal /= np.linalg.norm(sub_normal)
                sub_position = np.mean(sub_vertices, axis=0)
                
                # Create new Facet object
                new_facet = Facet(sub_normal, sub_vertices, simulation.timesteps_per_day, simulation.max_days, simulation.n_layers, simulation.calculate_energy_terms)
                new_shape_model.append(new_facet)
        
        # Update shape_model for the next iteration
        shape_model = new_shape_model
    
    return new_shape_model

@jit(nopython=True)
def calculate_visible_facets(positions, normals):
    ''' 
    This function calculates the visible (test) facets from each subject facet. It calculates the angle between the normal vector of each facet and the line of sight to every other facet. It returns the indices of the visible facets.
    
    NB: This doesn't account for partial shadowing (e.g. a facet may be only partially covered by the shadow cast by another facet) - more of an issue for low facet count models. Could add subdivision option to the model for better partial shadowing, but probably best to just use higher facet count models.
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

def calculate_shape_model_view_factors(shape_model, thermal_data, n_rays=10000):
    #NOTE: This doesn't need to be random - could just use a grid of rays, might be faster. 

    all_view_factors = []
    
    shape_model_hash = get_shape_model_hash(shape_model)
    view_factors_filename = get_view_factors_filename(shape_model_hash)

    if os.path.exists(view_factors_filename):
        with np.load(view_factors_filename, allow_pickle=True) as data:
            print("Loading existing view factors...")
            all_view_factors = list(data['view_factors'])
    else:
        print("No existing view factors found.")
        all_view_factors = []
    
    if not all_view_factors:
        print("Calculating new view factors...")
        for i in tqdm(range(len(shape_model)), desc="Calculating secondary radiation view factors"):
            visible_indices = thermal_data.visible_facets[i]

            subject_vertices = shape_model[i].vertices
            subject_area = shape_model[i].area
            subject_normal = shape_model[i].normal
            test_vertices = np.array([shape_model[j].vertices for j in visible_indices]).reshape(-1, 3, 3)
            test_areas = np.array([shape_model[j].area for j in visible_indices])

            view_factors = calculate_view_factors(subject_vertices, subject_normal, subject_area, test_vertices, test_areas, n_rays)

            if np.any(np.isnan(view_factors)) or np.any(np.isinf(view_factors)):
                print(f"Warning: Invalid view factor for facet {i}")
                print(f"View factors: {view_factors}")
                print(f"Visible facets: {visible_indices}")
            all_view_factors.append(view_factors)

        # Save the calculated view factors
        os.makedirs("view_factors", exist_ok=True)
        np.savez_compressed(view_factors_filename, view_factors=np.array(all_view_factors, dtype=object))

    return all_view_factors

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

def calculate_and_cache_visible_facets(shape_model, positions, normals, vertices):
    shape_model_hash = get_shape_model_hash(shape_model)
    visible_facets_filename = get_visible_facets_filename(shape_model_hash)

    if os.path.exists(visible_facets_filename):
        print("Loading existing visible facets...")
        with np.load(visible_facets_filename, allow_pickle=True) as data:
            visible_indices = data['visible_indices']
        # Convert back to list of numpy arrays
        visible_indices = [np.array(indices) for indices in visible_indices]
    else:
        print("Calculating visible facets...")
        potentially_visible_indices = calculate_visible_facets(positions, normals)
        print("Eliminating obstructed facets...")
        visible_indices = eliminate_obstructed_facets(positions, vertices, potentially_visible_indices)
        
        # Save the calculated visible facets
        os.makedirs(os.path.dirname(visible_facets_filename), exist_ok=True)
        np.savez_compressed(visible_facets_filename, visible_indices=np.array(visible_indices, dtype=object))

    return visible_indices

def get_visible_facets_filename(shape_model_hash):
    return f"visible_facets/visible_facets_{shape_model_hash}.npz"

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

def apply_scattering(thermal_data, shape_model, simulation, num_iterations=5):
    for iteration in range(num_iterations):
        scattered_light = np.zeros_like(thermal_data.insolation)

        for i, facet in enumerate(shape_model):
            for t in range(simulation.timesteps_per_day):
                insolation = thermal_data.insolation[i, t]
                
                if insolation > 0:
                    visible_facets = thermal_data.visible_facets[i]
                    view_factors = thermal_data.secondary_radiation_view_factors[i]
                    
                    for vf_index, vf in zip(visible_facets, view_factors):
                        scattered_light[vf_index, t] += insolation * vf * simulation.albedo / np.pi

        thermal_data.insolation += scattered_light
    
    return thermal_data

def calculate_insolation(thermal_data, shape_model, simulation):
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
                    illumination_factor = calculate_shadowing(np.array(facet.position), np.array([rotated_sunlight_directions[t]]), shape_model_vertices, thermal_data.visible_facets[i])
                
                # Calculate insolation
                insolation = simulation.solar_luminosity * (1 - simulation.albedo) * illumination_factor * cos_zenith_angle / (4 * np.pi * simulation.solar_distance_m**2)
            else:
                insolation = 0
            
            thermal_data.insolation[i, t] = insolation

    if simulation.include_scattering:
        thermal_data = apply_scattering(thermal_data, shape_model, simulation)

    return thermal_data

def calculate_initial_temperatures(thermal_data, emissivity, n_jobs=-1):
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

    print(f"Initial temperatures calculated for {thermal_data.temperatures.shape[0]} facets.")

    # Update the original shape_model with the results
    for i, temperature in tqdm(enumerate(results), total=len(results), desc='Saving temps'):
        thermal_data.temperatures[i, :, :] = temperature

    print("Initial temperatures saved for all facets.")

    return thermal_data

@jit(nopython=True)
def calculate_secondary_radiation(temperatures, visible_facets, view_factors, self_heating_const):
    return self_heating_const * np.sum(temperatures[visible_facets]**4 * view_factors)

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

def thermophysical_body_model(thermal_data, shape_model, simulation, path_to_shape_model_file):
    ''' 
    This is the main calculation function for the thermophysical body model. It calls the necessary functions to read in the shape model, set material and model properties, calculate 
    insolation and temperature arrays, and iterate until the model converges.
    '''

    mean_temperature_error = simulation.convergence_target + 1
    day = 0 
    temperature_error = 0

    const1 = simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
    const2 = simulation.emissivity * simulation.beaming_factor * 5.67e-8 * simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
    const3 = simulation.thermal_diffusivity * simulation.delta_t / simulation.layer_thickness**2
    self_heating_const = 5.670374419e-8 * simulation.delta_t * simulation.emissivity / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity * np.pi)

    error_history = []

    # Set comparison temperatures to the initial temperatures of the first timestep (day 0, timestep 0)
    comparison_temps = thermal_data.temperatures[:, 0, 0].copy()

    while day < simulation.max_days and (day < simulation.min_days or mean_temperature_error > simulation.convergence_target):
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
            simulation.include_self_heating,
            current_day_start, current_day_end
        )

        thermal_data.temperatures[:, current_day_start:current_day_end, :] = current_day_temperature

        if simulation.calculate_energy_terms:
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
                if np.isnan(current_day_temperature[i, time_step, 0]) or np.isinf(current_day_temperature[i, time_step, 0]) or current_day_temperature[i, time_step, 0] < 0:
                    print(f"Ending run at timestep {current_step} due to facet {i} having a temperature of {current_day_temperature[i, time_step, 0]} K.\n Try increasing the number of time steps per day")

                    # Plot the energy terms for the facet
                    if simulation.calculate_energy_terms:
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

                    animate_model(path_to_shape_model_file, 
                                  facet_highlight_array, 
                                  simulation.rotation_axis, 
                                  simulation.sunlight_direction, 
                                  simulation.timesteps_per_day, 
                                  simulation.solar_distance_au,
                                  simulation.rotation_period_hours,
                                  colour_map='coolwarm', plot_title='Problematic facet', axis_label='Problem facet is red', animation_frames=200, save_animation=False, save_animation_name='problematic_facet_animation.gif', background_colour = 'black')

                    sys.exit()

        # Calculate convergence factor
        temperature_error = np.sum(np.abs(current_day_temperature[:, 0, 0] - comparison_temps)) # 
        mean_temperature_error = temperature_error / len(shape_model)

        # Ensure propagation of the temperatures to the next day
        if day < simulation.max_days - 1:
            # Set the deep layer temperature to the mean surface temperature of all timesteps of the current day
            mean_surface_temp = np.mean(current_day_temperature[:, :, 0])
            thermal_data.temperatures[:, next_day_start:next_day_start + simulation.timesteps_per_day, -1] = mean_surface_temp
    
        print(f"Day: {day} | Mean Temperature error: {mean_temperature_error:.6f} K | Convergence target: {simulation.convergence_target} K")

        comparison_temps = current_day_temperature[:, 0, 0].copy()
        
        error_history.append(mean_temperature_error)
        day += 1

    # Decrement the day counter
    day -= 1

    # Remove unused days from thermal_data
    thermal_data.temperatures = thermal_data.temperatures[:, :simulation.timesteps_per_day * (day+1), :]

    final_day_temperatures = thermal_data.temperatures[:, -simulation.timesteps_per_day:, 0]

    final_timestep_temperatures = thermal_data.temperatures[:, -1, 0]
    final_day_temperatures_all_layers = thermal_data.temperatures[:, -simulation.timesteps_per_day:, :]

    if mean_temperature_error < simulation.convergence_target:
        print(f"Convergence achieved after {day} days.")
        if simulation.calculate_energy_terms:
            for i in range(len(shape_model)):
                thermal_data.insolation_energy[i] = energy_terms[i, :, 0]
                thermal_data.re_emitted_energy[i] = energy_terms[i, :, 1]
                thermal_data.surface_energy_change[i] = energy_terms[i, :, 2]
                thermal_data.conducted_energy[i] = energy_terms[i, :, 3]
                thermal_data.unphysical_energy_loss[i] = energy_terms[i, :, 4]
    else:
        print(f"Maximum days reached without achieving convergence.")
        print(f"Final temperature error: {temperature_error / len(shape_model)} K")
        print("Try increasing max_days or decreasing convergence_target.")

        if simulation.calculate_energy_terms:
            plt.plot(energy_terms[i, :, 0], label="Insolation energy")
            plt.plot(energy_terms[i, :, 1], label="Re-emitted energy")
            plt.plot(energy_terms[i, :, 2], label="Surface energy change")
            plt.plot(energy_terms[i, :, 3], label="Conducted energy")
            plt.plot(energy_terms[i, :, 4], label="Unphysical energy loss")
            plt.legend()
            plt.show()

    return final_day_temperatures, final_day_temperatures_all_layers, final_timestep_temperatures, day+1, temperature_error

def main():
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.
    '''

    # Shape model name
    shape_model_name = "67P_not_to_scale_1666_facets.stl"

    # Get setup file and shape model
    path_to_shape_model_file = f"shape_models/{shape_model_name}"
    path_to_setup_file = "model_setups/John_Spencer_default_model_parameters.json"

    # Load setup parameters from JSON file
    simulation = Simulation(path_to_setup_file, calculate_energy_terms=False) #BUG: - numba doesn't work with energy terms being calculated
    
    shape_model = read_shape_model(path_to_shape_model_file, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, simulation.calculate_energy_terms)

    thermal_data = ThermalData(len(shape_model), simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, simulation.calculate_energy_terms)

    print(f"\nDerived model parameters:")
    print(f"Number of timesteps per day: {simulation.timesteps_per_day}")
    print(f"Layer thickness: {simulation.layer_thickness} m")
    print(f"Thermal inertia: {simulation.thermal_inertia} W m^-2 K^-1 s^0.5")
    print(f"Skin depth: {simulation.skin_depth} m")

    print(f"\n Number of facets: {len(shape_model)}")

    ################ Modelling ################
    simulation.include_self_heating = False
    simulation.include_scattering = False # TODO: Investigate dependence on number of scatters (particularly for most shaded facets)
    simulation.apply_roughness = False

    ################ PLOTTING ################ BUG: If using 2 animations, the second one doesn't work (pyvista segmenation fault)
    plot_shadowing = True
    plot_insolation_curve = False
    plot_initial_temp_histogram = False
    plot_secondary_radiation_view_factors = False
    plot_secondary_contributions = False
    plot_final_day_temp_distribution = False
    plot_final_day_all_layers_temp_distribution = False
    plot_all_days_all_layers_temp_distribution = False
    plot_energy_terms = False # Note: You must set simulation.calculate_energy_terms to True to plot energy terms
    plot_temp_distribution_for_final_day = False
    animate_final_day_temp_distribution = False
    plot_final_day_comparison = False

    # Apply roughness to the shape model
    if simulation.apply_roughness:
        print(f"Applying roughness to shape model. Original size: {len(shape_model)} facets.")
        shape_model = apply_roughness(shape_model, simulation)
        print(f"Roughness applied to shape model. New size: {len(shape_model)} facets.")
        # Save the shape model with roughness applied with a new filename
        path_to_shape_model_file = f"shape_models/{shape_model_name[:-4]}_roughness_applied.stl"
        # Save it to a new file
        save_shape_model(shape_model, path_to_shape_model_file)

        # Visualise shape model with roughness
        animate_model(path_to_shape_model_file, 
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
    
    visible_indices = calculate_and_cache_visible_facets(shape_model, positions, normals, vertices)
        
    thermal_data.set_visible_facets(visible_indices)

    if simulation.include_self_heating or simulation.include_scattering:
        all_view_factors = calculate_shape_model_view_factors(shape_model, thermal_data, n_rays=10000)
        
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

    thermal_data = calculate_insolation(thermal_data, shape_model, simulation)

    facet_index = 1066 # Index of facet to plot

    if plot_shadowing:
        print(f"Preparing shadowing visualisation.\n")

        animate_model(path_to_shape_model_file, 
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

    if plot_insolation_curve:
        fig_insolation = plt.figure()
        plt.plot(thermal_data.insolation[facet_index])
        plt.xlabel('Number of timesteps')
        plt.ylabel('Insolation (W/m^2)')
        plt.title('Insolation curve for a single facet for one full rotation of the body')
        fig_insolation.show()

    print(f"Calculating initial temperatures.\n")
    thermal_data = calculate_initial_temperatures(thermal_data, simulation.emissivity)

    if plot_initial_temp_histogram:
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

    if plot_secondary_radiation_view_factors:
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
        print(f"\nContributing facets for facet {selected_facet}:")
        for index, view_factors in zip(contributing_indices, contributing_view_factors):
            print(f"Facet {index}: view factor = {view_factors:.6f}")
        print(f"Total number of contributing facets: {len(contributing_indices)}")
        
        print(f"Preparing visualization of contributing facets for facet {selected_facet}.")
        animate_model(path_to_shape_model_file, 
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
        
    if plot_secondary_contributions:
        # Calculate the sum of secondary radiation view factors for each facet
        secondary_radiation_sum = np.array([np.sum(view_factors) for view_factors in thermal_data.secondary_radiation_view_factors])

        print("Preparing secondary radiation visualization.")
        animate_model(path_to_shape_model_file, 
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

    print(f"Running main simulation loop.\n")
    start_time = time.time()
    final_day_temperatures, final_day_temperatures_all_layers, final_timestep_temperatures, day, temperature_error = thermophysical_body_model(thermal_data, shape_model, simulation, path_to_shape_model_file)
    end_time = time.time()
    execution_time = end_time - start_time

    if final_timestep_temperatures is not None:
        print(f"Convergence target achieved after {day} days.")
        print(f"Final temperature error: {temperature_error / len(shape_model)} K")
    else:
        print(f"Model did not converge after {day} days.")
        print(f"Final temperature error: {temperature_error / len(shape_model)} K")

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
        plt.plot(thermal_data.temperatures[facet_index, :, :])
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
        plt.plot(thermal_data.temperatures[facet_index, (day - 1) * simulation.timesteps_per_day:day * simulation.timesteps_per_day, 0])
        plt.xlabel('Timestep')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature distribution for all layers in facet for the full run')
        fig_final_day_temps.show()

    if animate_final_day_temp_distribution:
        print(f"Preparing temperature animation.\n")

        animate_model(path_to_shape_model_file, 
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

    if plot_final_day_comparison:
        print(f"Saving final day temperatures for facet to CSV file.\n")
        np.savetxt("final_day_temperatures.csv", np.column_stack((np.linspace(0, 2 * np.pi, simulation.timesteps_per_day), final_day_temperatures[facet_index])), delimiter=',', header='Rotation angle (rad), Temperature (K)', comments='')

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
