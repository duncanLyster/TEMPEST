'''
This module contains functions for calculating view factors between facets of a shape model. The view factors are calculated by firing rays from the vertices of a subject facet to the vertices of test facets and checking if the rays intersect with the test facets. The view factor is calculated as the number of rays that intersect with the test facets divided by the total number of rays fired from the subject facet. The view factors are used to calculate the secondary radiation between facets of the shape model.

TODO: Divide into more modules for better organisation.
'''

import os
import time
import numpy as np
from numba import jit, njit, prange
from joblib import Parallel, delayed
from src.utilities.locations import Locations
from src.utilities.utils import (
    rays_triangles_intersection,
    normalize_vector,
    random_points_in_triangle,
    conditional_print,
    conditional_tqdm,
    get_shape_model_hash
)

@jit(nopython=True)
def calculate_visible_facets_chunk(positions, normals, start_idx, end_idx):
    """Calculate potentially visible facets for a chunk of the shape model.
        This elimates any facets from the view factor calculations that:
            a) Are below the 'horizon' of the subject facet. 
            b) Face away from the subject facet.
    """
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

def calculate_visible_facets(positions, normals, n_jobs, chunk_size, silent_mode=False):
    """Parallel wrapper for calculate_visible_facets using joblib.

    NOTE: Idea for improvement - instead of geometrically comparing every facet - do this only for nearby facets, then use tracing of evenly 
    distributed rays to determine shadowing 'globe/sphere' for the rest of the facets.
    """
    n_facets = len(positions)
    n_chunks = (n_facets + chunk_size - 1) // chunk_size
    
    total_possible_pairs = n_facets * n_facets  # Total possible facet pairs
    
    # Create chunks of work
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n_facets)) 
             for i in range(n_chunks)]
    
    all_visible_indices = []
    total_visible_pairs = 0
    start_time = time.time()
    
    with Parallel(n_jobs=n_jobs, verbose=0) as parallel:  
        for chunk_start in range(0, len(chunks), n_jobs):
            chunk_end = min(chunk_start + n_jobs, len(chunks))
            current_chunks = chunks[chunk_start:chunk_end]
            
            results = parallel(
                delayed(calculate_visible_facets_chunk)(positions, normals, start_idx, end_idx)
                for start_idx, end_idx in current_chunks
            )
            
            # Process and save results after each batch
            for chunk_results in results:
                all_visible_indices.extend(chunk_results)
                total_visible_pairs += sum(len(indices) for indices in chunk_results)
            
            # Calculate progress and estimated time
            tasks_done = chunk_end
            percent_done = (tasks_done / n_chunks) * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (tasks_done / n_chunks)
            remaining_time = estimated_total - elapsed_time
            
            if not silent_mode:
                print(f"\rCalculating visible facets: {percent_done:0.1f}% done | "
                      f"Tasks: {tasks_done:,}/{n_chunks:,} | "
                      f"Elapsed: {elapsed_time/60:0.1f}min | "
                      f"Remaining (estimate): {remaining_time/60:0.1f}min", end="")
    
    if not silent_mode:
        print()  # New line after progress complete
        
    reduction_percent = ((total_possible_pairs - total_visible_pairs) / total_possible_pairs) * 100
    conditional_print(silent_mode, f"Facet visibility culling: {total_possible_pairs:,} → {total_visible_pairs:,} pairs")
    conditional_print(silent_mode, f"Reduced by {reduction_percent:.1f}%")
    
    return all_visible_indices

def calculate_and_cache_visible_facets(silent_mode, shape_model, positions, normals, vertices, config):
    locations = Locations()
    shape_model_hash = get_shape_model_hash(shape_model)
    visible_facets_filename = locations.get_visible_facets_path(shape_model_hash)

    ################## Calculate/load visible facets ##################

    if os.path.exists(visible_facets_filename):
        conditional_print(silent_mode, "Loading existing visible facets...")
        with np.load(visible_facets_filename, allow_pickle=True) as data:
            visible_indices = data['visible_indices']
            # Convert each array to int64 explicitly after loading, with progress
            visible_indices = [
                np.array(indices, dtype=np.int64)
                for indices in conditional_tqdm(
                    visible_indices, silent_mode, desc="Converting visible facets"
                )
            ]
    else:
        calculate_visible_facets_start = time.time()

        actual_n_jobs = config.validate_jobs()
        conditional_print(silent_mode, f"Calculating visible facets using {actual_n_jobs} parallel jobs...")
        visible_indices = calculate_visible_facets(positions, normals, config.n_jobs, config.chunk_size)

        calculate_visible_facets_end = time.time()

        conditional_print(silent_mode, f"Time taken to calculate visible facets: {calculate_visible_facets_end - calculate_visible_facets_start:.2f} seconds")
        
        os.makedirs(os.path.dirname(visible_facets_filename), exist_ok=True)
        # Save uncompressed for faster load
        np.savez(visible_facets_filename, visible_indices=np.array(visible_indices, dtype=object))

    return visible_indices

@njit(parallel=True)
def calculate_view_factors(subject_vertices, subject_normal, subject_area, test_vertices, test_areas, n_rays):
    '''
    Calculate view factors between one subject facet and multiple test vertices. The view factors are calculated by firing rays from the subject vertices to the test vertices and checking if the rays intersect with the test vertices. The view factor is calculated as the number of rays that intersect with the test vertices divided by the total number of rays fired from the subject vertices.
    '''

    ray_sources = random_points_in_triangle(subject_vertices[0], subject_vertices[1], subject_vertices[2], n_rays)
    
    ray_directions = np.random.randn(n_rays, 3)
    for i in prange(n_rays):
        ray_directions[i] = normalize_vector(ray_directions[i])
    
    valid_directions = np.zeros(n_rays, dtype=np.bool_)
    for i in prange(n_rays):
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
    for i in prange(n_rays):
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

def calculate_all_view_factors(shape_model, thermal_data, config, n_rays):
    """
    Calculate view factors for all facets using parallel processing.
    Works with any number of jobs (including 1).
    """
    shape_model_hash = get_shape_model_hash(shape_model)
    locations = Locations()
    view_factors_filename = locations.get_view_factors_path(shape_model_hash)

    # Try to load existing view factors first
    if os.path.exists(view_factors_filename):
        with np.load(view_factors_filename, allow_pickle=True) as data:
            conditional_print(config.silent_mode, "Loading existing view factors...")
            return list(data['view_factors'])

    conditional_print(config.silent_mode, "No existing view factors found.")
    
    # Get number of jobs and validate
    actual_n_jobs = config.validate_jobs()
    n_facets = len(shape_model)
    
    # Calculate chunk size
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (actual_n_jobs * 4))
    
    # Create chunks
    n_chunks = (n_facets + config.chunk_size - 1) // config.chunk_size
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
             for i in range(n_chunks)]

    conditional_print(config.silent_mode, 
                     f"\nCalculating view factors using {actual_n_jobs} parallel jobs:")
    conditional_print(config.silent_mode, f"Total facets: {n_facets:,}")
    conditional_print(config.silent_mode, f"Chunk size: {config.chunk_size:,}")
    conditional_print(config.silent_mode, f"Number of chunks: {n_chunks:,}")

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
        
        # Ensure the directory exists before saving the file
        os.makedirs(locations.view_factors, exist_ok=True)

        # Save the calculated view factors
        np.savez_compressed(view_factors_filename, 
                           view_factors=np.array(all_view_factors, dtype=object))
        
        return all_view_factors
        
    except KeyboardInterrupt:
        print("\nCalculation interrupted by user. Progress was not saved.")
        raise
    except Exception as e:
        print(f"\nError during calculation: {str(e)}")
        raise

class EPFLookupTable:
    """Lookup table for emission phase function values."""
    def __init__(self, lut_name):
        locations = Locations()
        lut_path = locations.get_emission_lut_path(lut_name)
        
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"EPF lookup table not found at {lut_path}")
        
        data = np.load(lut_path, allow_pickle=True).item()
        self.angles = data['emission_angles']
        self.values = data['table']
        
    def query(self, angle):
        """Get EPF value for a given emission angle."""
        return np.interp(angle, self.angles, self.values)

def calculate_thermal_view_factors(shape_model, thermal_data, config):
    """Calculate thermal view factors by applying EPF values to existing view factors."""
    # Initialize EPF lookup table here instead of in tempest.py
    epf_lut = EPFLookupTable(config.emission_lut)
    
    shape_model_hash = get_shape_model_hash(shape_model)
    locations = Locations()
    thermal_vf_filename = locations.get_thermal_view_factors_path(shape_model_hash)

    if os.path.exists(thermal_vf_filename):
        with np.load(thermal_vf_filename, allow_pickle=True) as data:
            conditional_print(config.silent_mode, "Loading existing thermal view factors...")
            return list(data['thermal_view_factors'])

    conditional_print(config.silent_mode, "Calculating thermal view factors...")
    
    actual_n_jobs = config.validate_jobs()
    n_facets = len(shape_model)
    
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (actual_n_jobs * 4))
    
    n_chunks = (n_facets + config.chunk_size - 1) // config.chunk_size
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
             for i in range(n_chunks)]

    try:
        results = Parallel(n_jobs=actual_n_jobs, verbose=10, backend='loky')(
            delayed(process_thermal_view_factors_chunk)(
                shape_model, thermal_data, epf_lut, start_idx, end_idx
            ) for start_idx, end_idx in chunks
        )
        
        all_thermal_view_factors = []
        for chunk_results in results:
            all_thermal_view_factors.extend(chunk_results)
        
        os.makedirs(locations.thermal_view_factors, exist_ok=True)
        np.savez_compressed(thermal_vf_filename, 
                          thermal_view_factors=np.array(all_thermal_view_factors, dtype=object))
        
        return all_thermal_view_factors
        
    except Exception as e:
        print(f"Error calculating thermal view factors: {str(e)}")
        raise

def process_thermal_view_factors_chunk(shape_model, thermal_data, epf_lut, start_idx, end_idx):
    """Process a chunk of facets for thermal view factor calculation."""
    chunk_results = []
    
    for i in range(start_idx, end_idx):
        visible_facets = thermal_data.visible_facets[i]
        view_factors = thermal_data.secondary_radiation_view_factors[i]
        
        # Ensure both arrays have the same length and are not empty
        if len(visible_facets) == 0 or len(view_factors) == 0:
            chunk_results.append(np.array([]))
            continue
        
        if len(visible_facets) != len(view_factors):
            raise ValueError(f"Mismatch between visible_facets ({len(visible_facets)}) and view_factors ({len(view_factors)}) for facet {i}")
            
        # Calculate emission angles
        target_centers = np.array([shape_model[idx].position for idx in visible_facets])
        direction_vectors = target_centers - shape_model[i].position
        direction_vectors /= np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]
        
        cos_emission = np.einsum('ij,j->i', direction_vectors, shape_model[i].normal)
        cos_emission = np.clip(cos_emission, -1.0, 1.0)
        emission_angles = np.degrees(np.arccos(cos_emission))
        
        # Get EPF values and combine with view factors
        epf_values = np.array([epf_lut.query(em) for em in emission_angles])
        chunk_results.append(view_factors * epf_values)
    
    return chunk_results
