'''
This module contains functions for calculating view factors between facets of a shape model. The view factors are calculated by firing rays from the vertices of a subject facet to the vertices of test facets and checking if the rays intersect with the test facets. The view factor is calculated as the number of rays that intersect with the test facets divided by the total number of rays fired from the subject facet. The view factors are used to calculate the secondary radiation between facets of the shape model.

TODO: Divide into more modules for better organisation.
'''

import os
import time
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from src.utilities.utils import (
    rays_triangles_intersection,
    normalize_vector,
    normalize_vectors,
    random_points_in_triangle,
    conditional_print,
    conditional_tqdm,
    get_shape_model_hash,
    get_view_factors_filename,
    get_visible_facets_filename
)

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
    '''
    This function calculates the unobstructed facets from the potentially visible facets. It performs a ray-triangle intersection test to determine if the line of sight to each facet is obstructed by another facet. It returns the indices of the unobstructed facets. BUG: Currently this elimintes facets that should not be eliminated, possibly due to only using the potentially visible facets for the ray-triangle intersection test, or due to the ray-triangle intersection test itself only considering centre of test facet. Do not use for now. 
    '''
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
def calculate_visible_facets_chunk(positions, normals, start_idx, end_idx):
    """Calculate visible facets for a chunk of the shape model."""
    n_facets = len(positions)
    chunk_size = end_idx - start_idx
    potentially_visible_indices = [np.empty(0, dtype=np.int64) for _ in range(chunk_size)]
    epsilon = 1e-10
    
    for i in range(chunk_size):
        actual_idx = i + start_idx
        relative_positions = positions[actual_idx] - positions
        above_horizon = np.sum(relative_positions * normals[actual_idx], axis=1) < epsilon
        facing_towards = np.sum(-relative_positions * normals, axis=1) < epsilon
        potentially_visible = above_horizon & facing_towards
        potentially_visible[actual_idx] = False
        visible_indices = np.where(potentially_visible)[0]
        potentially_visible_indices[i] = visible_indices
        
    return potentially_visible_indices

def calculate_visible_facets_parallel(positions, normals, n_jobs, chunk_size):
    """Parallel wrapper for calculate_visible_facets using joblib."""
    n_facets = len(positions)
    n_chunks = (n_facets + chunk_size - 1) // chunk_size
    
    # Create chunks of work
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_facets)) 
             for i in range(n_chunks)]
    
    # Process chunks in parallel using joblib
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(calculate_visible_facets_chunk)(positions, normals, start_idx, end_idx)
        for start_idx, end_idx in chunks
    )
    
    # Combine results
    all_visible_indices = []
    for chunk_result in results:
        all_visible_indices.extend(chunk_result)
    
    return all_visible_indices

@jit(nopython=True)
def eliminate_obstructed_facets_chunk(positions, shape_model_vertices, potentially_visible_facet_indices, 
                                    start_idx, end_idx):
    """Process a chunk of facets for obstruction elimination."""
    chunk_size = end_idx - start_idx
    unobstructed_facets = [np.empty(0, dtype=np.int64) for _ in range(chunk_size)]

    for i in range(chunk_size):
        actual_idx = i + start_idx
        potential_indices = potentially_visible_facet_indices[actual_idx]
        
        if len(potential_indices) == 0:
            continue

        subject_position = positions[actual_idx]
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

            intersections, _ = rays_triangles_intersection(
                subject_position,
                ray_directions[j:j+1],
                test_vertices
            )

            if not np.any(intersections):
                unobstructed.append(test_facet_index)

        unobstructed_facets[i] = np.array(unobstructed, dtype=np.int64)

    return unobstructed_facets

def eliminate_obstructed_facets_parallel(positions, shape_model_vertices, potentially_visible_facet_indices,
                                       n_jobs, chunk_size):
    """Parallel wrapper for eliminate_obstructed_facets using joblib."""
    n_facets = len(positions)
    n_chunks = (n_facets + chunk_size - 1) // chunk_size
    
    # Create chunks of work
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_facets)) 
             for i in range(n_chunks)]
    
    # Process chunks in parallel using joblib
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(eliminate_obstructed_facets_chunk)(
            positions, shape_model_vertices, potentially_visible_facet_indices,
            start_idx, end_idx
        )
        for start_idx, end_idx in chunks
    )
    
    # Combine results
    all_unobstructed_facets = []
    for chunk_result in results:
        all_unobstructed_facets.extend(chunk_result)
    
    return all_unobstructed_facets

def calculate_and_cache_visible_facets(silent_mode, shape_model, positions, normals, vertices, config):
    shape_model_hash = get_shape_model_hash(shape_model)
    visible_facets_filename = get_visible_facets_filename(shape_model_hash)

    ################## Calculate/load visible facets ##################

    if os.path.exists(visible_facets_filename):
        conditional_print(silent_mode, "Loading existing visible facets...")
        with np.load(visible_facets_filename, allow_pickle=True) as data:
            visible_indices = data['visible_indices']
        visible_indices = [np.array(indices) for indices in visible_indices]
    else:
        calculate_visible_facets_start = time.time()

        if config.n_jobs == 1:
            # Use serial version
            conditional_print(silent_mode, "Calculating visible facets...")
            visible_indices = calculate_visible_facets(positions, normals)
        else:
            # Use parallel version
            actual_n_jobs = config.validate_jobs()
            conditional_print(silent_mode, f"Calculating visible facets using {actual_n_jobs} parallel jobs...")
            visible_indices = calculate_visible_facets_parallel(positions, normals, config.n_jobs, config.chunk_size)

        calculate_visible_facets_end = time.time()

        conditional_print(silent_mode, f"Time taken to calculate visible facets: {calculate_visible_facets_end - calculate_visible_facets_start:.2f} seconds")

        ################## Eliminate obstructed facets ##################

        # BUG: eliminating obstructed facets is not working as expected - may be due to visibility calculation - possible route, just ignore it. Intention is to speed up code and reduce memory usage, but benefits appear to be minimal. 

        # conditional_print(silent_mode, "Eliminating obstructed facets...")

        # if config.n_jobs == 1:
        #     # Use serial version
        #     eliminate_obstructed_facets_start = time.time()
        #     visible_indices = eliminate_obstructed_facets(positions, vertices, visible_indices)
        #     eliminate_obstructed_facets_end = time.time()

        # else:
        #     # Use parallel version
        #     eliminate_obstructed_facets_start = time.time()
        #     visible_indices = eliminate_obstructed_facets_parallel(positions, vertices, visible_indices, actual_n_jobs, config.chunk_size)
        #     eliminate_obstructed_facets_end = time.time()

        # conditional_print(silent_mode, f"Time taken to eliminate obstructed facets: {eliminate_obstructed_facets_end - eliminate_obstructed_facets_start:.2f} seconds")
        
        os.makedirs(os.path.dirname(visible_facets_filename), exist_ok=True)
        np.savez_compressed(visible_facets_filename, 
                          visible_indices=np.array(visible_indices, dtype=object))

    return visible_indices

@jit(nopython=True)
def calculate_view_factors(subject_vertices, subject_normal, subject_area, test_vertices, test_areas, n_rays):
    '''
    Calculate view factors between one subject facet and multiple test vertices. The view factors are calculated by firing rays from the subject vertices to the test vertices and checking if the rays intersect with the test vertices. The view factor is calculated as the number of rays that intersect with the test vertices divided by the total number of rays fired from the subject vertices.
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

def process_view_factors_chunk(shape_model, thermal_data, start_idx, end_idx, n_rays):
    """
    Process a chunk of subject facets for view factor calculations.
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

def calculate_all_view_factors_parallel(shape_model, thermal_data, config, n_rays):
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
    
    conditional_print(config.silent_mode, 
                     f"\nCalculating view factors using {actual_n_jobs} parallel jobs:")
    conditional_print(config.silent_mode,
                     f"Total facets: {n_facets:,}")
    conditional_print(config.silent_mode,
                     f"Chunk size: {config.chunk_size:,}")
    conditional_print(config.silent_mode,
                     f"Number of chunks: {n_chunks:,}")

    # Process chunks in parallel with joblib's built-in verbose output
    try:
        results = Parallel(n_jobs=actual_n_jobs, verbose=10, backend='loky')(
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
        print("\nCalculation interrupted by user. Progress was not saved.")
        raise
    except Exception as e:
        print(f"\nError during calculation: {str(e)}")
        raise

def calculate_all_view_factors(shape_model, thermal_data, simulation, config, n_rays=1000):
    all_view_factors = []
    
    shape_model_hash = get_shape_model_hash(shape_model)
    view_factors_filename = get_view_factors_filename(shape_model_hash)

    if os.path.exists(view_factors_filename):
        with np.load(view_factors_filename, allow_pickle=True) as data:
            conditional_print(config.silent_mode,  "Loading existing view factors...")
            all_view_factors = list(data['view_factors'])
    else:
        conditional_print(config.silent_mode,  "No existing view factors found.")
        all_view_factors = []
    
    if not all_view_factors:
        conditional_print(simulation, "Calculating new view factors...")
        for i in conditional_tqdm(range(len(shape_model)), config.silent_mode, desc="Calculating secondary radiation view factors"):
            visible_indices = thermal_data.visible_facets[i]

            subject_vertices = shape_model[i].vertices
            subject_area = shape_model[i].area
            subject_normal = shape_model[i].normal
            test_vertices = np.array([shape_model[j].vertices for j in visible_indices]).reshape(-1, 3, 3)
            test_areas = np.array([shape_model[j].area for j in visible_indices])

            view_factors = calculate_view_factors(subject_vertices, subject_normal, subject_area, test_vertices, test_areas, n_rays)

            if np.any(np.isnan(view_factors)) or np.any(np.isinf(view_factors)):
                conditional_print(config.silent_mode,  f"Warning: Invalid view factor for facet {i}")
                conditional_print(config.silent_mode,  f"View factors: {view_factors}")
                conditional_print(config.silent_mode,  f"Visible facets: {visible_indices}")
            all_view_factors.append(view_factors)

        # Save the calculated view factors
        os.makedirs("view_factors", exist_ok=True)
        np.savez_compressed(view_factors_filename, view_factors=np.array(all_view_factors, dtype=object))

    return all_view_factors
