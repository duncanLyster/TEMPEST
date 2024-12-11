# src/model/insolation.py

'''
This module calculates the insolation for each facet of the body. It calculates the angle between the sun and each facet, and then calculates the insolation for each facet factoring in shadows. It writes the insolation to the data cube.
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
                
                # Calculate insolation
                insolation = simulation.solar_luminosity * (1 - simulation.albedo) * illumination_factor * cos_zenith_angle / (4 * np.pi * simulation.solar_distance_m**2)
            else:
                insolation = 0
            
            thermal_data.insolation[i, t] = insolation

    if config.n_scatters > 0:
        conditional_print(config.silent_mode, f"Applying light scattering with {config.n_scatters} iterations...")
        scattering_start = time.time()
        thermal_data = apply_scattering(thermal_data, shape_model, simulation, config)
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

def apply_scattering(thermal_data, shape_model, simulation, config):
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
    
    # Get number of jobs
    actual_n_jobs = config.validate_jobs()
    
    # Calculate chunk size
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
