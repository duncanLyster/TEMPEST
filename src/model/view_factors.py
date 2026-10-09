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

# Appended to cached view-factor filenames. Bump it whenever the way view factors are
# computed changes, so old caches are not silently reused.
#   v2: cosine-weighted ray directions (Aug 2026)
#   v3: rays credited to the nearest hit only (Oct 2026)
VIEW_FACTOR_CACHE_TAG = "v3"

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

@jit(nopython=True)
def _visible_from_candidates(positions, normals, cand_flat, cand_start, cand_count,
                             start_idx, end_idx):
    """Apply the mutual-horizon test to a precomputed candidate list per facet.

    Same two tests as calculate_visible_facets_chunk -- j must lie above i's
    tangent plane and i above j's -- but evaluated only against spatial
    neighbours rather than the whole model.
    """
    chunk = end_idx - start_idx
    out = [np.empty(0, dtype=np.int64) for _ in range(chunk)]
    epsilon = 1e-10

    for k in range(chunk):
        i = start_idx + k
        count = cand_count[i]
        if count == 0:
            continue
        base = cand_start[i]
        keep = np.empty(count, dtype=np.int64)
        m = 0
        pix, piy, piz = positions[i, 0], positions[i, 1], positions[i, 2]
        nix, niy, niz = normals[i, 0], normals[i, 1], normals[i, 2]
        for t in range(count):
            j = cand_flat[base + t]
            if j == i:
                continue
            rx = pix - positions[j, 0]
            ry = piy - positions[j, 1]
            rz = piz - positions[j, 2]
            # j above i's plane
            if rx * nix + ry * niy + rz * niz >= epsilon:
                continue
            # i above j's plane
            if -(rx * normals[j, 0] + ry * normals[j, 1] + rz * normals[j, 2]) >= epsilon:
                continue
            keep[m] = j
            m += 1
        out[k] = keep[:m].copy()

    return out


def auto_interaction_distance(positions, min_solar_elevation_deg=5.0):
    """Distance beyond which one facet cannot shadow another.

    Uses the full relief of the patch as the obstacle height, so the result is
    relief / tan(min_solar_elevation_deg). This is conservative, because real relief
    is not a single vertical step. For production runs, set max_interaction_distance_m
    from a convergence test instead.
    """
    if len(positions) == 0:
        return 0.0
    # Relief is the spread in radial distance (the patch sits on a sphere).
    radii = np.linalg.norm(positions, axis=1)
    relief = float(np.max(radii) - np.min(radii))
    return relief / np.tan(np.radians(max(min_solar_elevation_deg, 0.05)))


def calculate_visible_facets_local(positions, normals, max_distance, n_jobs, chunk_size,
                                   silent_mode=False):
    """Visible-facet search limited to facets within max_distance.

    The global search keeps a large fraction of all pairs on extended terrain, so
    terrain runs need a distance cutoff. Neighbours are found one chunk at a time to
    keep peak memory bounded.
    """
    from scipy.spatial import cKDTree

    n_facets = len(positions)
    start_time = time.time()

    tree = cKDTree(positions)
    chunk_size = max(1, int(chunk_size))
    n_chunks = (n_facets + chunk_size - 1) // chunk_size

    all_visible_indices = []
    total_candidates = 0

    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n_facets)

        neighbours = tree.query_ball_point(positions[lo:hi], max_distance,
                                           workers=max(1, n_jobs))
        counts = np.array([len(x) for x in neighbours], dtype=np.int64)
        starts = np.zeros(len(counts), dtype=np.int64)
        if len(counts) > 1:
            np.cumsum(counts[:-1], out=starts[1:])
        flat = np.fromiter((j for nb in neighbours for j in nb),
                           dtype=np.int64, count=int(counts.sum()))
        del neighbours
        total_candidates += int(counts.sum())

        # _visible_from_candidates indexes by global facet id, so pad to full length.
        full_counts = np.zeros(n_facets, dtype=np.int64)
        full_starts = np.zeros(n_facets, dtype=np.int64)
        full_counts[lo:hi] = counts
        full_starts[lo:hi] = starts

        all_visible_indices.extend(
            _visible_from_candidates(positions, normals, flat,
                                     full_starts, full_counts, lo, hi))
        del flat, full_counts, full_starts, counts, starts

        if not silent_mode and (ci % max(1, n_chunks // 20) == 0 or ci == n_chunks - 1):
            elapsed = time.time() - start_time
            print(f"\rCalculating visible facets (local): {100*(ci+1)/n_chunks:0.1f}% | "
                  f"Elapsed: {elapsed/60:0.1f}min", end="")
    if not silent_mode:
        print()

    conditional_print(silent_mode,
                      f"Spatial cull at {max_distance:,.0f} m: "
                      f"{total_candidates/max(n_facets,1):,.0f} candidates/facet "
                      f"(vs {n_facets:,} for the global search)")
    total_pairs = sum(len(v) for v in all_visible_indices)
    conditional_print(silent_mode,
                      f"Facet visibility: {n_facets**2:,} possible -> {total_pairs:,} pairs "
                      f"({total_pairs/max(n_facets,1):,.0f}/facet, "
                      f"{100*(1 - total_pairs/max(n_facets**2,1)):.3f}% culled) "
                      f"in {time.time()-start_time:.1f}s")
    return all_visible_indices


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

    # max_interaction_distance_m: None = global search, 0 = derive from relief,
    # > 0 = cutoff in metres. Each cutoff gets its own cache entry.
    max_distance = getattr(config, 'max_interaction_distance_m', None)
    if max_distance is not None:
        max_distance = float(max_distance)
        if max_distance == 0:
            max_distance = auto_interaction_distance(
                positions, getattr(config, 'min_solar_elevation_degrees', 5.0))
            conditional_print(silent_mode,
                              f"Auto interaction distance: {max_distance:,.0f} m "
                              f"(relief / tan({getattr(config, 'min_solar_elevation_degrees', 5.0)} deg))")
    # View factors are stored per visible facet, so their caches need the same suffix
    config.visibility_cache_suffix = f"_r{max_distance:.0f}" if max_distance is not None else ""
    shape_model_hash += config.visibility_cache_suffix

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
        if max_distance is not None:
            visible_indices = calculate_visible_facets_local(
                positions, normals, max_distance, actual_n_jobs, config.chunk_size, silent_mode)
        else:
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
    n_hat = subject_normal / np.sqrt(np.sum(subject_normal**2))  # sampling below needs a unit normal

    # Cosine-weighted (Lambertian) direction sampling: d = normalize(n_hat + u) with
    # u uniform on the unit sphere gives p(d) = cos(theta)/pi, so the hit fraction is
    # the diffuse view factor F_ij directly.  (Uniform solid-angle sampling instead
    # measures Omega_j/2pi, which under-weights face-on exchange by up to 2x and is
    # NOT a view factor; the EPF tables assume Lambertian-relative weighting.)
    ray_directions = np.empty((n_rays, 3))
    for i in prange(n_rays):
        good = False
        while not good:
            ux = np.random.normal(0.0, 1.0)
            uy = np.random.normal(0.0, 1.0)
            uz = np.random.normal(0.0, 1.0)
            un = (ux * ux + uy * uy + uz * uz) ** 0.5
            if un < 1e-12:
                continue
            dx = ux / un + n_hat[0]
            dy = uy / un + n_hat[1]
            dz = uz / un + n_hat[2]
            dn = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dn > 1e-9:
                ray_directions[i, 0] = dx / dn
                ray_directions[i, 1] = dy / dn
                ray_directions[i, 2] = dz / dn
                good = True

    # Each ray is credited only to the nearest facet it hits; anything further along
    # the ray is occluded by it.
    intersections = np.zeros((n_rays, len(test_vertices)), dtype=np.bool_)
    for i in prange(n_rays):
        ray_origin = ray_sources[i]
        ray_dir = ray_directions[i]
        intersect, t_hit = rays_triangles_intersection(ray_origin, ray_dir.reshape(1, 3), test_vertices)
        nearest = -1
        nearest_t = np.inf
        for k in range(len(test_vertices)):
            if intersect[0, k] and t_hit[0, k] < nearest_t:
                nearest_t = t_hit[0, k]
                nearest = k
        if nearest >= 0:
            intersections[i, nearest] = True

    view_factors = np.sum(intersections, axis=0).astype(np.float64) / n_rays

    return view_factors

def process_view_factors_chunk(all_vertices, all_normals, all_areas, chunk_visible_facets, 
                               start_idx, end_idx, n_rays):
    """
    Process a chunk of subject facets for view factor calculations.
    Accepts pre-extracted numpy arrays AND ONLY the relevant chunk of visible_facets
    to avoid massive pickle overhead in multiprocessing.
    Returns a list of view factors and any warnings generated.
    
    Args:
        all_vertices: full vertices array for all facets (needed for test vertices lookup)
        all_normals: normals for all facets
        all_areas: areas for all facets
        chunk_visible_facets: visible_facets only for facets in range [start_idx, end_idx)
                              (length = end_idx - start_idx)
        start_idx: starting facet index
        end_idx: ending facet index
        n_rays: number of rays for view factor calculation
    """
    chunk_view_factors = []
    warnings = []
    
    for local_i in range(len(chunk_visible_facets)):
        global_i = start_idx + local_i
        visible_indices = chunk_visible_facets[local_i]
        
        subject_vertices = all_vertices[global_i]
        subject_area = all_areas[global_i]
        subject_normal = all_normals[global_i]
        test_vertices = all_vertices[visible_indices].reshape(-1, 3, 3)
        test_areas = all_areas[visible_indices]

        view_factors = calculate_view_factors(
            subject_vertices, subject_normal, subject_area, 
            test_vertices, test_areas, n_rays
        )

        if np.any(np.isnan(view_factors)) or np.any(np.isinf(view_factors)):
            warnings.append({
                'facet': global_i,
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
    shape_model_hash = get_shape_model_hash(shape_model) + getattr(config, 'visibility_cache_suffix', '')
    locations = Locations()
    view_factors_filename = locations.get_view_factors_path(f"{shape_model_hash}_{VIEW_FACTOR_CACHE_TAG}")

    # Try to load existing view factors first (unless bypass flag is set)
    if not config.force_recalculate_view_factors and os.path.exists(view_factors_filename):
        with np.load(view_factors_filename, allow_pickle=True) as data:
            conditional_print(config.silent_mode, "Loading existing view factors...")
            return list(data['view_factors'])

    conditional_print(config.silent_mode, "No existing view factors found." if not config.force_recalculate_view_factors else "Recalculating view factors (force_recalculate_view_factors=true)...")
    
    # Get number of jobs and validate
    actual_n_jobs = config.validate_jobs()
    n_facets = len(shape_model)
    
    # Pre-extract compact numpy arrays to avoid pickling entire shape_model/thermal_data
    all_vertices = np.array([facet.vertices for facet in shape_model], dtype=np.float64)
    all_normals = np.array([facet.normal for facet in shape_model], dtype=np.float64)
    all_areas = np.array([facet.area for facet in shape_model], dtype=np.float64)
    visible_facets_list = thermal_data.visible_facets  # list of int64 arrays
    
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
        # MEMORY OPTIMIZATION: Pass only the chunk of visible_facets needed by each worker
        # instead of the entire list, to avoid duplicating ~79 GB across 32 workers
        results = Parallel(n_jobs=actual_n_jobs, verbose=10, backend='loky')(
            delayed(process_view_factors_chunk)(
                all_vertices, all_normals, all_areas,
                visible_facets_list[start_idx:end_idx],  # NEW: Only pass subset needed
                start_idx, end_idx, n_rays
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
    
    shape_model_hash = get_shape_model_hash(shape_model) + getattr(config, 'visibility_cache_suffix', '')
    locations = Locations()
    thermal_vf_filename = locations.get_thermal_view_factors_path(f"{shape_model_hash}_{VIEW_FACTOR_CACHE_TAG}")

    if not config.force_recalculate_thermal_view_factors and os.path.exists(thermal_vf_filename):
        with np.load(thermal_vf_filename, allow_pickle=True) as data:
            conditional_print(config.silent_mode, "Loading existing thermal view factors...")
            return list(data['thermal_view_factors'])

    conditional_print(config.silent_mode, "Calculating thermal view factors..." if not config.force_recalculate_thermal_view_factors else "Recalculating thermal view factors (force_recalculate_thermal_view_factors=true)...")
    
    actual_n_jobs = config.validate_jobs()
    n_facets = len(shape_model)
    
    # Pre-extract compact numpy arrays to avoid pickling entire shape_model/thermal_data
    all_positions = np.array([facet.position for facet in shape_model], dtype=np.float64)
    all_normals = np.array([facet.normal for facet in shape_model], dtype=np.float64)
    visible_facets_list = thermal_data.visible_facets
    vf_list = thermal_data.secondary_radiation_view_factors
    
    if config.chunk_size <= 0:
        config.chunk_size = max(1, n_facets // (actual_n_jobs * 4))
    
    n_chunks = (n_facets + config.chunk_size - 1) // config.chunk_size
    chunks = [(i * config.chunk_size, min((i + 1) * config.chunk_size, n_facets)) 
             for i in range(n_chunks)]

    try:
        # MEMORY OPTIMIZATION: Pass only the chunk of visible_facets and vf_list needed by each worker
        # instead of the entire lists, to avoid duplicating ~gigabytes across 32 workers
        results = Parallel(n_jobs=actual_n_jobs, verbose=10, backend='loky')(
            delayed(process_thermal_view_factors_chunk)(
                all_positions, all_normals,
                visible_facets_list[start_idx:end_idx],  # NEW: Only pass subset needed
                vf_list[start_idx:end_idx],              # NEW: Only pass subset needed
                epf_lut, start_idx, end_idx
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

def process_thermal_view_factors_chunk(all_positions, all_normals, chunk_visible_facets, chunk_vf_list,
                                       epf_lut, start_idx, end_idx):
    """Process a chunk of facets for thermal view factor calculation.
    Accepts pre-extracted numpy arrays AND ONLY the relevant chunk of visible_facets and vf_list
    to avoid massive pickle overhead in multiprocessing.
    
    Args:
        all_positions: full positions array for all facets (needed for target lookup)
        all_normals: normals for all facets
        chunk_visible_facets: visible_facets only for facets in range [start_idx, end_idx)
        chunk_vf_list: view factors only for facets in range [start_idx, end_idx)
        epf_lut: emission phase function lookup table
        start_idx: starting facet index
        end_idx: ending facet index
    """
    chunk_results = []
    
    for local_i in range(len(chunk_visible_facets)):
        global_i = start_idx + local_i
        visible_facets = chunk_visible_facets[local_i]
        view_factors = chunk_vf_list[local_i]
        
        # Ensure both arrays have the same length and are not empty
        if len(visible_facets) == 0 or len(view_factors) == 0:
            chunk_results.append(np.array([]))
            continue
        
        if len(visible_facets) != len(view_factors):
            raise ValueError(f"Mismatch between visible_facets ({len(visible_facets)}) and view_factors ({len(view_factors)}) for facet {global_i}")
            
        # Calculate emission angles
        target_centers = all_positions[visible_facets]
        direction_vectors = target_centers - all_positions[global_i]
        direction_vectors /= np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]
        
        cos_emission = np.einsum('ij,j->i', direction_vectors, all_normals[global_i])
        cos_emission = np.clip(cos_emission, -1.0, 1.0)
        emission_angles = np.degrees(np.arccos(cos_emission))
        
        # Get EPF values and combine with view factors
        epf_values = np.array([epf_lut.query(em) for em in emission_angles])
        
        chunk_results.append(view_factors * epf_values)
    
    return chunk_results
