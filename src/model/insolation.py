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
    rotation_matrices = np.zeros((simulation.timesteps_per_day, 3, 3), dtype=np.float64)
    rotated_sunlight_directions = np.zeros((simulation.timesteps_per_day, 3), dtype=np.float64)
    
    for t in range(simulation.timesteps_per_day):
        rotation_matrix = calculate_rotation_matrix(simulation.rotation_axis, 
                                                 (2 * np.pi / simulation.timesteps_per_day) * t)
        rotation_matrices[t] = rotation_matrix
        rotated_sunlight_directions[t] = np.dot(rotation_matrix.T, simulation.sunlight_direction)
        rotated_sunlight_directions[t] /= np.linalg.norm(rotated_sunlight_directions[t])

    # Create chunks for parallel processing
    n_facets = len(shape_model)
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (config.n_jobs * 4))
    
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
              for i in range((n_facets + config.chunk_size - 1) // config.chunk_size)]

    # Extract numpy arrays from shape model and ensure float64 dtype
    normals = np.array([facet.normal for facet in shape_model], dtype=np.float64)
    positions = np.array([facet.position for facet in shape_model], dtype=np.float64)
    shape_model_vertices = np.array([facet.vertices for facet in shape_model], dtype=np.float64)
    
    # Process chunks in parallel
    parallel = Parallel(n_jobs=config.n_jobs, verbose=0)
    results = parallel(
        delayed(process_insolation_chunk)(
            normals[start_idx:end_idx],
            positions[start_idx:end_idx],
            thermal_data.visible_facets[start_idx:end_idx],
            rotation_matrices,
            rotated_sunlight_directions,
            simulation.solar_luminosity,
            simulation.albedo,
            simulation.solar_distance_m,
            simulation.sunlight_direction.astype(np.float64),  # Ensure float64
            config.include_shadowing,
            shape_model_vertices
        )
        for start_idx, end_idx in chunks
    )

    # Combine results
    for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
        thermal_data.insolation[start_idx:end_idx] = results[chunk_idx]

    if config.n_scatters > 0:
        conditional_print(config.silent_mode, 
                        f"Applying light scattering with {config.n_scatters} iterations...")
        scattering_start = time.time()
        thermal_data = apply_scattering(thermal_data, shape_model, simulation, config,
                                      rotation_matrices, rotated_sunlight_directions)
        scattering_end = time.time()
        conditional_print(config.silent_mode, 
                        f"Time taken to apply light scattering: {scattering_end - scattering_start:.2f} seconds")

    return thermal_data

@jit(nopython=True)
def process_insolation_chunk(normals, positions, visible_facets, rotation_matrices, 
                           rotated_sunlight_directions, solar_luminosity, albedo,
                           solar_distance_m, sunlight_direction, include_shadowing,
                           shape_model_vertices):
    """
    Process insolation calculations for a chunk of facets using only numba-compatible types.
    """
    chunk_size = len(normals)
    timesteps = len(rotation_matrices)
    insolation = np.zeros((chunk_size, timesteps))
    
    # physical softening of the shadow terminator edge
    # smoothing sun size: roughly 2 degrees. (e.g. 4 timesteps out of 720)
    window_size = max(1, timesteps // 180)
    pad_left = window_size // 2
    
    for i in range(chunk_size):
        normal = normals[i]
        position = positions[i]
        
        illum_factors = np.zeros(timesteps, dtype=np.float64)
        cos_zeniths = np.zeros(timesteps, dtype=np.float64)
        
        # 1. First pass: compute binary shadow state and zenith angle
        for t in range(timesteps):
            new_normal = np.dot(rotation_matrices[t], normal)
            new_normal_norm = np.linalg.norm(new_normal)
            sun_dot_normal = np.dot(sunlight_direction, new_normal)
            
            cos_zenith_angle = sun_dot_normal / (np.linalg.norm(sunlight_direction) * new_normal_norm)
            
            if cos_zenith_angle > 0:
                cos_zeniths[t] = cos_zenith_angle
                if len(visible_facets[i]) != 0 and include_shadowing:
                    illum_factors[t] = float(calculate_shadowing(
                        position, 
                        rotated_sunlight_directions[t:t+1],  
                        shape_model_vertices,
                        visible_facets[i]
                    ))
                else:
                    illum_factors[t] = 1.0
                    
        # 2. Second pass: apply moving average to illumination_factors to simulate solar disk diameter
        # This mitigates Crank-Nicolson spike artifacts near terminators
        smoothed_illum = np.zeros(timesteps, dtype=np.float64)
        for t in range(timesteps):
            if window_size <= 1:
                smoothed_illum[t] = illum_factors[t]
            else:
                sum_val = 0.0
                for w in range(window_size):
                    idx = (t - pad_left + w) % timesteps
                    sum_val += illum_factors[idx]
                smoothed_illum[t] = sum_val / window_size
                
        # 3. Third pass: calculate final insolation combining smoothed shadow and original zenith angle
        for t in range(timesteps):
            if cos_zeniths[t] > 0 and smoothed_illum[t] > 0:
                insolation[i, t] = (
                    solar_luminosity * 
                    (1 - albedo) * 
                    smoothed_illum[t] * 
                    cos_zeniths[t] / 
                    (4 * np.pi * solar_distance_m**2)
                )
    
    return insolation

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

def calculate_brdf_values(all_normals, all_positions, rotation_matrices, rotated_sunlight_directions, brdf_lut, start_idx, end_idx, chunk_visible_facets):
    n_timesteps = len(rotation_matrices)
    brdf_values = {}
    n_facets_in_chunk = end_idx - start_idx
    
    # Collect all unique facet indices needed for this chunk
    needed_facets = set(range(start_idx, end_idx))  # Start with facets in chunk
    for local_i in range(n_facets_in_chunk):
        needed_facets.update(chunk_visible_facets[local_i])  # Add their visible facets
    needed_facets = sorted(needed_facets)  # Convert to sorted list
    
    # Create mapping from global to local indices
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(needed_facets)}
    
    # Pre-rotate only the needed facets
    n_needed_facets = len(needed_facets)
    rotated_normals = np.zeros((n_needed_facets, n_timesteps, 3))
    rotated_positions = np.zeros((n_needed_facets, n_timesteps, 3))
    
    # Pre-compute rotated values for needed facets (using compact numpy arrays)
    for local_idx, global_idx in enumerate(needed_facets):
        normal_i = all_normals[global_idx]
        position_i = all_positions[global_idx]
        for t in range(n_timesteps):
            rotated_normals[local_idx, t] = np.dot(rotation_matrices[t], normal_i)
            rotated_positions[local_idx, t] = np.dot(rotation_matrices[t], position_i)
    
    # Process facets in this chunk
    for i in range(start_idx, end_idx):
        local_i_map = global_to_local[i]
        local_i = i - start_idx
        visible_facets = chunk_visible_facets[local_i]
        brdf_values[i] = np.zeros((len(visible_facets), n_timesteps))
        
        sun_cos = np.einsum('tj,tj->t', rotated_sunlight_directions, rotated_normals[local_i_map])
        illuminated_timesteps = sun_cos > 0
        
        if not illuminated_timesteps.any():
            continue
            
        inc_deg = np.degrees(np.arccos(sun_cos[illuminated_timesteps]))
        
        for j, target_idx in enumerate(visible_facets):
            local_target = global_to_local[target_idx]
            
            direction_vectors = (rotated_positions[local_target, illuminated_timesteps] - 
                               rotated_positions[local_i_map, illuminated_timesteps])
            
            # Rest of the BRDF calculation remains the same...

def process_scattering_chunk(start_idx, end_idx, chunk_input_light, chunk_visible_facets, 
                           chunk_view_factors, n_facets, timesteps_per_day, albedo, iteration,
                           brdf_lut=None, all_normals=None, all_positions=None,
                           rotation_matrices=None, rotated_sunlight_directions=None):
    """
    Process a chunk of facets for scattering calculation.
    Accepts pre-sliced per-chunk data to minimize memory usage per worker.
    """
    chunk_scattered_light = np.zeros((n_facets, timesteps_per_day))
    
    # Only calculate BRDF on first iteration
    if iteration == 0 and brdf_lut is not None:
        brdf_values = calculate_brdf_values(
            all_normals, all_positions, rotation_matrices, rotated_sunlight_directions,
            brdf_lut, start_idx, end_idx, chunk_visible_facets
        )
    else:
        brdf_values = None  # Use default Lambertian scattering for subsequent iterations
    
    for i in range(start_idx, end_idx):
        local_i = i - start_idx
        visible_facets = chunk_visible_facets[local_i]
        view_factors = chunk_view_factors[local_i]

        for t in range(timesteps_per_day):
            current_light = chunk_input_light[local_i, t]
            
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
    Pre-extracts compact numpy arrays and slices per-chunk to minimize worker memory.
    """
    # Initialize BRDF lookup table
    brdf_lut = BRDFLookupTable(config.scattering_lut)
    
    original_insolation = thermal_data.insolation.copy()
    total_scattered_light = np.zeros_like(original_insolation)
    n_facets = len(shape_model)
    
    # Pre-extract compact numpy arrays from shape_model for BRDF calculation
    all_normals = np.array([facet.normal for facet in shape_model], dtype=np.float64)
    all_positions = np.array([facet.position for facet in shape_model], dtype=np.float64)
    
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
                input_light[start_idx:end_idx],           # Only this chunk's input
                visible_facets_list[start_idx:end_idx],   # Only this chunk's visible facets
                view_factors_list[start_idx:end_idx],     # Only this chunk's view factors
                n_facets,
                simulation.timesteps_per_day,
                simulation.albedo, 
                iteration,
                brdf_lut if iteration == 0 else None,
                all_normals if iteration == 0 else None,
                all_positions if iteration == 0 else None,
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
