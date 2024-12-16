# src/model/insolation.py

'''
This module calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.

NOTE: Currently this requires a lot of RAM - look for ways to reduce this and check if it's the main bottleneck in the code.
'''

import time
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from src.utilities.utils import (
    conditional_tqdm,
    conditional_print,
    rays_triangles_intersection,
    calculate_rotation_matrix
)   
from src.model.scattering import BRDFLookupTable
from tqdm import tqdm

def calculate_insolation(thermal_data, shape_model, simulation, config):
    ''' 
    This function calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.

    TODO: Parallelise this function.
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
                
                # Calculate insolation TODO: Add BRDF dependence to albedo calculation 
                insolation = simulation.solar_luminosity * (1 - simulation.albedo) * illumination_factor * cos_zenith_angle / (4 * np.pi * simulation.solar_distance_m**2)
            else:
                insolation = 0
            
            thermal_data.insolation[i, t] = insolation

    if config.n_scatters > 0:
        conditional_print(config.silent_mode, f"Applying light scattering with {config.n_scatters} iterations...")
        scattering_start = time.time()
        thermal_data = apply_scattering(thermal_data, shape_model, simulation, config, 
                                      rotation_matrices, rotated_sunlight_directions)
        scattering_end = time.time()
        conditional_print(config.silent_mode, f"Time taken to apply light scattering: {scattering_end - scattering_start:.2f} seconds")

    return thermal_data

@jit(nopython=True)
def calculate_shadowing(subject_positions, sunlight_directions, shape_model_vertices, visible_facet_indices):
    '''
    This function calculates whether a facet is in shadow at a given time step. It cycles through all visible facets and passes their vertices to rays_triangles_intersections which determines whether they fall on the sunlight direction vector (starting at the facet position). If they do, the facet is in shadow. 
    
    It returns the illumination factor for the facet at that time step. 0 if the facet is in shadow, 1 if it is not.

    TODO: Use Monte Carlo or even distribution ray tracing for faster shadowing calculations.
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

def calculate_brdf_values(shape_model, rotation_matrices, rotated_sunlight_directions, brdf_lut, start_idx, end_idx, visible_facets_list):
    n_timesteps = len(rotation_matrices)
    brdf_values = {}
    n_facets_in_chunk = end_idx - start_idx
    
    # Collect all unique facet indices needed for this chunk
    needed_facets = set(range(start_idx, end_idx))  # Start with facets in chunk
    for i in range(start_idx, end_idx):
        needed_facets.update(visible_facets_list[i])  # Add their visible facets
    needed_facets = sorted(needed_facets)  # Convert to sorted list
    
    # Create mapping from global to local indices
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(needed_facets)}
    
    # Pre-rotate only the needed facets
    n_needed_facets = len(needed_facets)
    rotated_normals = np.zeros((n_needed_facets, n_timesteps, 3))
    rotated_positions = np.zeros((n_needed_facets, n_timesteps, 3))
    
    # Pre-compute rotated values for needed facets
    for local_idx, global_idx in enumerate(needed_facets):
        normal_i = shape_model[global_idx].normal
        position_i = shape_model[global_idx].position
        for t in range(n_timesteps):
            rotated_normals[local_idx, t] = np.dot(rotation_matrices[t], normal_i)
            rotated_positions[local_idx, t] = np.dot(rotation_matrices[t], position_i)
    
    # Process facets in this chunk
    for i in range(start_idx, end_idx):
        local_i = global_to_local[i]
        visible_facets = visible_facets_list[i]
        brdf_values[i] = np.zeros((len(visible_facets), n_timesteps))
        
        sun_cos = np.einsum('tj,tj->t', rotated_sunlight_directions, rotated_normals[local_i])
        illuminated_timesteps = sun_cos > 0
        
        if not illuminated_timesteps.any():
            continue
            
        inc_deg = np.degrees(np.arccos(sun_cos[illuminated_timesteps]))
        
        for j, target_idx in enumerate(visible_facets):
            local_target = global_to_local[target_idx]
            
            direction_vectors = (rotated_positions[local_target, illuminated_timesteps] - 
                               rotated_positions[local_i, illuminated_timesteps])
            
            # Rest of the BRDF calculation remains the same...

def process_scattering_chunk(start_idx, end_idx, input_light, visible_facets_list, 
                           view_factors_list, timesteps_per_day, albedo, iteration,
                           brdf_lut=None, shape_model=None, rotation_matrices=None, 
                           rotated_sunlight_directions=None):
    """
    Process a chunk of facets for scattering calculation.
    """
    chunk_scattered_light = np.zeros_like(input_light)
    
    # Only calculate BRDF on first iteration
    if iteration == 0 and brdf_lut is not None:
        brdf_values = calculate_brdf_values(
            shape_model, rotation_matrices, rotated_sunlight_directions,
            brdf_lut, start_idx, end_idx, visible_facets_list
        )
    else:
        brdf_values = None  # Use default Lambertian scattering for subsequent iterations
    
    for i in range(start_idx, end_idx):
        visible_facets = visible_facets_list[i]
        view_factors = view_factors_list[i]

        for t in range(timesteps_per_day):
            current_light = input_light[i, t]
            
            if current_light > 0:
                for j, (vf_idx, vf) in enumerate(zip(visible_facets, view_factors)):
                    brdf = brdf_values[i][j, t] if brdf_values is not None else 1.0
                    chunk_scattered_light[vf_idx, t] += (
                        brdf * current_light * vf * albedo / np.pi
                    )
                    
    return chunk_scattered_light

def apply_scattering(thermal_data, shape_model, simulation, config, 
                    rotation_matrices, rotated_sunlight_directions):
    """
    Apply scattering using BRDF lookup tables. Works with any number of jobs (including 1).
    """
    # Initialize BRDF lookup table
    brdf_lut = BRDFLookupTable(config.scattering_lut)
    
    original_insolation = thermal_data.insolation.copy()
    total_scattered_light = np.zeros_like(original_insolation)
    n_facets = len(shape_model)
    
    # Convert lists to numpy arrays
    visible_facets_list = [np.array(x) for x in thermal_data.visible_facets]
    view_factors_list = [np.array(x) for x in thermal_data.secondary_radiation_view_factors]
    
    # Get number of jobs and create chunks
    actual_n_jobs = config.validate_jobs()
    
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (actual_n_jobs * 4))
    
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
             for i in range((n_facets + config.chunk_size - 1) // config.chunk_size)]
    
    for iteration in range(config.n_scatters):
        if iteration == 0:
            input_light = original_insolation
        else:
            input_light = scattered_light
        
        # Create parallel executor
        parallel = Parallel(n_jobs=actual_n_jobs, verbose=0)
        delayed_funcs = [
            delayed(process_scattering_chunk)(
                start_idx, end_idx,
                input_light,
                visible_facets_list,
                view_factors_list,
                simulation.timesteps_per_day,
                simulation.albedo, 
                iteration,
                brdf_lut if iteration == 0 else None,
                shape_model if iteration == 0 else None,
                rotation_matrices if iteration == 0 else None,
                rotated_sunlight_directions if iteration == 0 else None
            )
            for start_idx, end_idx in chunks
        ]
        
        # Process chunks with real-time progress bar
        scattered_light = np.zeros_like(original_insolation)
        with tqdm(total=len(chunks), desc=f"Iteration {iteration + 1}/{config.n_scatters}") as pbar:
            for chunk_result in parallel(delayed_funcs):
                scattered_light += chunk_result
                pbar.update(1)
                
        total_scattered_light += scattered_light
    
    thermal_data.insolation = original_insolation + total_scattered_light
    return thermal_data
