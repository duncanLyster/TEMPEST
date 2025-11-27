''' 
This model simulates diurnal temperature variations of a solar system body based on
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

import os
import sys
import math
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
from numba.typed import List
from stl import mesh as stl_mesh_module
from scipy.interpolate import interp1d
import h5py
from joblib import Parallel, delayed

# Ensure src directory is in the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

# Imports from src modules
from src.model.calculate_phase_curve import calculate_phase_curve
from src.model.insolation import calculate_insolation
from src.model.simulation import Simulation, ThermalData
from src.model.facet import Facet
from src.model.solvers import TemperatureSolverFactory
from src.model.view_factors import (
    calculate_and_cache_visible_facets,
    calculate_all_view_factors,
    calculate_thermal_view_factors,
    calculate_view_factors
)
from src.utilities.locations import Locations
from src.utilities.config import Config
from src.utilities.utils import (
    calculate_black_body_temp,
    conditional_print,
    calculate_rotation_matrix,
    conditional_tqdm,
    rays_triangles_intersection
)
from src.utilities.plotting.plotting import check_remote_and_animate

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
        shape_mesh = stl_mesh_module.Mesh.from_file(filename)
        shape_model = []
        for i in range(len(shape_mesh.vectors)):
            normal = shape_mesh.normals[i]
            vertices = shape_mesh.vectors[i]
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

def export_results(shape_model_name, config, temperature_array):
    ''' 
    This function exports the final results of the model to be used in an instrument simulator. It creates a folder within /output with the shape model, model parameters, a plot of the temperature distribution, and final timestep temperatures.
    '''

    folder_name = f"{shape_model_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}" # Create new folder name
    os.makedirs(f"output/{folder_name}") # Create folder for results
    shape_mesh = stl_mesh_module.Mesh.from_file(config.path_to_shape_model_file) # Load shape model
    os.system(f"cp {config.path_to_shape_model_file} output/{folder_name}") # Copy shape model .stl file to folder
    os.system(f"cp {config.path_to_setup_file} output/{folder_name}") # Copy model parameters .json file to folder
    np.savetxt(f"output/{folder_name}/temperatures.csv", temperature_array, delimiter=',') # Save the final timestep temperatures to .csv file

    # Plot the temperature distribution for the final timestep and save it to the folder
    temp_output_file_path = f"output/{folder_name}/"

def parse_args():
    parser = argparse.ArgumentParser(description='TEMPEST: Thermal Model for Planetary Bodies')
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file'
    )
    return parser.parse_args()

def check_environment(config):
    """Check if the execution environment matches config settings"""
    from multiprocessing import cpu_count
    
    available_cores = cpu_count()
    requested_cores = config.config_data.get('n_jobs', 4)  # Get original requested cores
    
    if config.remote and available_cores < 8:
        print(f"\nWARNING: Remote execution requested but only {available_cores} CPU cores available")
        print(f"This might significantly impact performance")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit("Execution cancelled by user")
    
    # Always warn about core count mismatch
    if requested_cores > available_cores:
        print(f"\nWARNING: Requested {requested_cores} cores but only {available_cores} available")
        print(f"Setting n_jobs to {available_cores}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    full_run_start_time = time.time()

    # Load user configuration with specified config path
    config = Config(config_path=args.config)
    
    # Check environment
    check_environment(config)

    # Load setup parameters from JSON file
    simulation = Simulation(config)

    # Setup simulation
    try:
        shape_model = read_shape_model(
            config.path_to_shape_model_file,
            simulation.timesteps_per_day,
            simulation.n_layers,
            simulation.max_days,
            config.calculate_energy_terms
        )
    except Exception as e:
        print(f"Failed to load shape model: {e}")
        sys.exit(1)
    thermal_data = ThermalData(len(shape_model), simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)

    conditional_print(config.silent_mode,  f"\nDerived model parameters:")
    conditional_print(config.silent_mode,  f"Number of timesteps per day: {simulation.timesteps_per_day}")
    conditional_print(config.silent_mode,  f"Layer thickness: {simulation.layer_thickness} m")
    conditional_print(config.silent_mode,  f"Thermal inertia: {simulation.thermal_inertia} W m^-2 K^-1 s^0.5")
    conditional_print(config.silent_mode,  f"Skin depth: {simulation.skin_depth} m")
    conditional_print(config.silent_mode,  f"\n Number of facets: {len(shape_model)}")

    # Apply kernel-based roughness to the shape model
    if config.apply_kernel_based_roughness:
        conditional_print(config.silent_mode, f"Applying kernel-based roughness ({config.roughness_kernel}) to shape model. Original size: {len(shape_model)} facets.")
        # Time the depression generation step
        t_dep_start = time.time()
        # Initialize the roughness model for each facet
        for facet in shape_model:
            facet.generate_spherical_depression(config, simulation)
        t_dep_end = time.time()
        conditional_print(config.silent_mode, f"Depression generation time: {t_dep_end - t_dep_start:.2f} seconds")
        # Pre-load visible parent facets for selective DP
        positions = np.array([f.position for f in shape_model], dtype=np.float64)
        normals = np.array([f.normal for f in shape_model], dtype=np.float64)
        vertices = np.array([f.vertices for f in shape_model], dtype=np.float64)
        visible_indices = calculate_and_cache_visible_facets(
            config.silent_mode, shape_model, positions, normals, vertices, config
        )
        for idx, facet in enumerate(shape_model):
            facet.visible_facets = visible_indices[idx]
        # Precompute dome-to-parent view-factors (parallel across facets)
        conditional_print(config.silent_mode, "Precomputing dome-to-parent view-factors for depressions...")
        
        # Directional ray-marching for dome-to-parent view factors
        @jit(nopython=True)
        def _compute_directional_vf(dome_center, dome_normal, dome_area, parent_vertices_subset, parent_areas_subset):
            """
            Compute view factor from dome bin to visible parents using directional ray.
            """
            # Shoot ray along dome bin normal direction
            ray_dir = dome_normal.reshape(1, 3)
            intersections, t_values = rays_triangles_intersection(
                dome_center.reshape(1, 3),
                ray_dir,
                parent_vertices_subset
            )
            
            # Initialize view factors
            n_targets = len(parent_vertices_subset)
            vf = np.zeros(n_targets, dtype=np.float64)
            
            if np.any(intersections[0]):
                # Find which parent facet(s) were hit
                hit_indices = np.where(intersections[0])[0]
                if len(hit_indices) > 0:
                    # Use closest hit
                    closest_idx = hit_indices[np.argmin(t_values[0, hit_indices])]
                    
                    # Compute view factor based on solid angle approximation
                    # For directional bins, we assume radiation exits in a cone
                    # View factor ≈ (dome_area / distance²) * cos(angle)
                    distance = t_values[0, closest_idx]
                    if distance > 1e-6:
                        # Get parent facet normal (from triangle vertices)
                        parent_tri = parent_vertices_subset[closest_idx]
                        parent_normal = np.cross(
                            parent_tri[1] - parent_tri[0],
                            parent_tri[2] - parent_tri[0]
                        )
                        parent_normal_norm = np.linalg.norm(parent_normal)
                        if parent_normal_norm > 1e-9:
                            parent_normal = parent_normal / parent_normal_norm
                            
                            # Cosine factor for receiving parent
                            cos_parent = abs(np.dot(-ray_dir[0], parent_normal))
                            
                            # Solid angle approximation: Ω ≈ A·cos(θ) / r²
                            solid_angle = dome_area * cos_parent / (distance * distance)
                            
                            # View factor from solid angle
                            vf[closest_idx] = solid_angle * parent_areas_subset[closest_idx] / dome_area
                            
            return vf
        
        # helper to compute one facet's DP matrix
        def _compute_dp(facet, parent_vertices, parent_areas, vf_rays, dome_radius_factor):
            M = len(facet.dome_facets)
            subj_vertices = np.zeros((M, 3, 3), dtype=np.float64)
            subj_normals = np.zeros((M, 3), dtype=np.float64)
            subj_areas = np.zeros(M, dtype=np.float64)
            n_parents_local = parent_vertices.shape[0]
            parent_radius = math.sqrt(facet.area / math.pi)
            for j, entry in enumerate(facet.dome_facets):
                # Scale and rotate dome patch
                local_scaled = entry['vertices'] * (dome_radius_factor * parent_radius)
                world_tris = (facet.dome_rotation.dot(local_scaled.T)).T + facet.position
                subj_vertices[j] = world_tris
                subj_normals[j] = facet.dome_rotation.dot(entry['normal'])
                subj_areas[j] = entry['area'] * (dome_radius_factor * parent_radius)**2
            # compute view factors only for visible parent facets using directional rays
            F_dp_local = np.zeros((M, n_parents_local), dtype=np.float64)
            vis = facet.visible_facets
            if len(vis) > 0:
                # Use optimized directional ray marching
                for j in range(M):
                    # Get dome bin center (centroid of triangle)
                    dome_center = np.mean(subj_vertices[j], axis=0)
                    
                    # Use directional ray marching
                    vals = _compute_directional_vf(
                        dome_center,
                        subj_normals[j],
                        subj_areas[j],
                        parent_vertices[vis],
                        parent_areas[vis]
                    )
                    F_dp_local[j, vis] = vals
            return F_dp_local
        # run in parallel: prepare parent arrays
        parent_vertices = np.array([f.vertices for f in shape_model])
        parent_areas = np.array([f.area for f in shape_model], dtype=np.float64)
        # Time the dome-to-parent view-factor computation with progress
        t_dp_start = time.time()
        dp_list = []
        # Chunk facets and process chunks in parallel (avoids nested parallelism with Numba)
        conditional_print(config.silent_mode, f"Processing DP view factors with {config.n_jobs} parallel workers...")
        facet_chunks = [shape_model[i:i+config.chunk_size] for i in range(0, len(shape_model), config.chunk_size)]
        for chunk in conditional_tqdm(facet_chunks, config.silent_mode, desc="DP chunks"):
            dp_chunk = Parallel(n_jobs=config.n_jobs)(
                delayed(_compute_dp)(
                    facet,
                    parent_vertices,
                    parent_areas,
                    config.vf_rays,
                    config.kernel_dome_radius_factor
                ) for facet in chunk
            )
            dp_list.extend(dp_chunk)
        # Assign DP results
        for facet, F_dp in zip(shape_model, dp_list):
            facet.depression_global_F_dp = F_dp
        conditional_print(config.silent_mode, "Precomputed dome-to-parent view-factors for depressions")
        t_dp_end = time.time()
        conditional_print(config.silent_mode, f"DP view-factor time: {t_dp_end - t_dp_start:.2f} seconds")

    # Setup the model
    positions = np.array([facet.position for facet in shape_model])
    normals = np.array([facet.normal for facet in shape_model])
    vertices = np.array([facet.vertices for facet in shape_model])
    
    # Time the visible facets loading/conversion
    t_vis_start = time.time()
    visible_indices = calculate_and_cache_visible_facets(config.silent_mode, shape_model, positions, normals, vertices, config)
    t_vis_end = time.time()
    conditional_print(config.silent_mode, f"Visible facets load time: {t_vis_end - t_vis_start:.2f} seconds")

    thermal_data.set_visible_facets(visible_indices)

    for i, facet in enumerate(shape_model):
        facet.visible_facets = visible_indices[i]   
        
    if config.include_self_heating or config.n_scatters > 0:
        calculate_view_factors_start = time.time()
        
        # Calculate regular view factors for scattering
        all_view_factors = calculate_all_view_factors(shape_model, thermal_data, config, config.vf_rays)
        
        # Calculate thermal view factors if self-heating is enabled
        if config.include_self_heating:
            thermal_view_factors = calculate_thermal_view_factors(
                shape_model,
                thermal_data,
                config
            )
            thermal_data.set_thermal_view_factors(thermal_view_factors)
        
        calculate_view_factors_end = time.time()
        conditional_print(config.silent_mode, f"Time taken to calculate view factors: {calculate_view_factors_end - calculate_view_factors_start:.2f} seconds")
        
        # Convert thermal_view_factors to Numba lists (always, for consistency with solver)
        numba_view_factors = List()
        for view_factors in thermal_data.thermal_view_factors:
            # Ensure array is 1D and handle empty arrays
            arr = np.array(view_factors, dtype=np.float64)
            if arr.size == 0:
                numba_view_factors.append(np.array([], dtype=np.float64))
            else:
                numba_view_factors.append(arr.flatten() if arr.ndim > 1 else arr)
        thermal_data.thermal_view_factors = numba_view_factors

    thermal_data = calculate_insolation(thermal_data, shape_model, simulation, config)

    # If kernel-based roughness is enabled, process per-timestep sub-facet insolation in vectorized fashion
    if config.apply_kernel_based_roughness:
        n_facets = len(shape_model)
        # Precompute normals and parent areas
        normals_array = np.array([f.normal for f in shape_model], dtype=np.float64)
        parent_areas = np.array([f.area for f in shape_model], dtype=np.float64)
        # Stack F_dp matrices for all facets: shape (n_facets, M, n_facets)
        F_dp_all = np.stack([f.depression_global_F_dp for f in shape_model], axis=0)
        
        # Check sparsity and optimize for convex bodies
        nonzero_fraction = np.count_nonzero(F_dp_all) / F_dp_all.size
        conditional_print(config.silent_mode, 
                         f"F_dp sparsity: {nonzero_fraction*100:.2f}% non-zero ({np.count_nonzero(F_dp_all)} / {F_dp_all.size} entries)")
        
        # For sparse matrices (< 10% non-zero), use optimized sparse computation
        use_sparse = nonzero_fraction < 0.1
        if use_sparse:
            conditional_print(config.silent_mode, "Using sparse coupling optimization")
            # Precompute nonzero indices for each facet
            nonzero_targets = []
            for i in range(n_facets):
                # Find which target facets this facet can couple to (any non-zero view factors)
                targets = np.any(F_dp_all[i] > 0, axis=0).nonzero()[0]
                nonzero_targets.append(targets)
        
        # Precompute sun directions per timestep
        world_dirs = np.zeros((simulation.timesteps_per_day, 3), dtype=np.float64)
        for tt in range(simulation.timesteps_per_day):
            ang = (2 * np.pi / simulation.timesteps_per_day) * tt
            R_tt = calculate_rotation_matrix(simulation.rotation_axis, ang)
            wd = R_tt.T.dot(simulation.sunlight_direction)
            world_dirs[tt] = wd / np.linalg.norm(wd)
        conditional_print(
            config.silent_mode,
            f"Applying kernel-based roughness over {simulation.timesteps_per_day} timesteps and {n_facets} facets..."
        )
        # Preallocate storage for dome thermal fluxes (per-facet, per-bin, per-timestep)
        M = shape_model[0].depression_outgoing_flux_array_th.shape[0]
        nF = len(shape_model)
        T = simulation.timesteps_per_day
        dome_flux_th = np.zeros((nF, M, T), dtype=np.float64)
        
        # Helper function to process a single facet's depression energetics (parallelizable)
        def _process_facet_energetics(facet, flux0, wd, config, simulation):
            """Process depression energetics for a single facet and return results."""
            if flux0 is None:
                # Facet not illuminated - return zeros
                N = len(facet.sub_facets) if len(facet.sub_facets) > 0 else M
                return (np.zeros(N, dtype=np.float64),
                        np.zeros(M, dtype=np.float64),
                        np.zeros(M, dtype=np.float64))
            
            # Clear any previous packets and add current incident packet
            facet.parent_incident_energy_packets = [(flux0, wd, 'solar')]
            # Process depression energetics
            facet.process_intra_depression_energetics(config, simulation)
            # Return results
            return (facet._last_absorbed_solar.copy(),
                    facet.depression_outgoing_flux_array_vis.copy(),
                    facet.depression_outgoing_flux_array_th.copy())
        
        # Time the roughness coupling loop
        t_coup_start = time.time()
        conditional_print(config.silent_mode, f"Processing roughness with {config.n_jobs} parallel workers...")
        # Loop over timesteps
        for t in conditional_tqdm(range(simulation.timesteps_per_day), config.silent_mode, desc="Roughness timesteps"):
            # Base insolation and sun dir
            base_ins = thermal_data.insolation[:, t]
            wd = world_dirs[t]
            cos_parent = normals_array.dot(wd)
            
            # Prepare inputs for parallel processing
            flux_inputs = []
            for i in range(n_facets):
                cp = cos_parent[i]
                if cp > 0:
                    flux0 = base_ins[i] / ((1 - simulation.albedo) * cp)
                    flux_inputs.append((shape_model[i], flux0, wd, config, simulation))
                else:
                    flux_inputs.append((shape_model[i], None, wd, config, simulation))
            
            # Process all facets in parallel (verbose shows worker info on first iteration)
            # Use 'threading' backend to avoid serialization issues with class attributes
            verbose_level = 10 if t == 0 and not config.silent_mode else 0
            results = Parallel(n_jobs=config.n_jobs, verbose=verbose_level, backend='threading')(
                delayed(_process_facet_energetics)(*inputs) for inputs in flux_inputs
            )
            
            # Unpack results and update facet states
            out_vis_all = np.zeros((nF, M), dtype=np.float64)
            out_th_all = np.zeros((nF, M), dtype=np.float64)
            for i, (absorbed_solar, out_vis, out_th) in enumerate(results):
                shape_model[i].depression_thermal_data.insolation[:, t] = absorbed_solar
                out_vis_all[i] = out_vis
                out_th_all[i] = out_th
            
            # Store thermal dome flux for animation
            dome_flux_th[:, :, t] = out_th_all
            
            # Coupling calculation: choose sparse or dense based on matrix sparsity
            if use_sparse:
                # Sparse computation: only process non-zero couplings
                inc_vis = np.zeros(nF, dtype=np.float64)
                inc_th = np.zeros(nF, dtype=np.float64)
                for i in range(nF):
                    if len(nonzero_targets[i]) > 0:
                        # Only compute for targets with non-zero view factors
                        targets = nonzero_targets[i]
                        # Sum over dome bins j: inc[target] += sum_j(out[i,j] * F_dp[i,j,target])
                        # F_dp_all[i, :, targets] has shape (M, len(targets))
                        # Transpose to get (len(targets), M) then dot with (M,) to get (len(targets),)
                        inc_vis[targets] += F_dp_all[i, :, :][:, targets].T.dot(out_vis_all[i])
                        inc_th[targets] += F_dp_all[i, :, :][:, targets].T.dot(out_th_all[i])
            else:
                # Dense computation (original): for dense or small matrices
                inc_vis = np.tensordot(out_vis_all, F_dp_all, axes=([0,1], [0,1]))
                inc_th  = np.tensordot(out_th_all,  F_dp_all, axes=([0,1], [0,1]))
            
            # Update insolation with directional coupling
            thermal_data.insolation[:, t] += (inc_vis + inc_th) / parent_areas
        # End of roughness coupling
        t_coup_end = time.time()
        conditional_print(config.silent_mode, f"Roughness coupling time: {t_coup_end - t_coup_start:.2f} seconds")

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
                          config.emissivity,
                          colour_map='binary_r', 
                          plot_title='Shadowing on the body', 
                          axis_label='Insolation (W/m^2)', 
                          animation_frames=200, 
                          save_animation=False, 
                          save_animation_name='shadowing_animation.gif', 
                          background_colour = 'black')

    conditional_print(config.silent_mode,  f"Calculating initial temperatures.\n")

    initial_temperatures_start = time.time()
    solver = TemperatureSolverFactory.create(config.temp_solver)
    thermal_data = solver.initialize_temperatures(thermal_data, simulation, config)
    initial_temperatures_end = time.time()

    conditional_print(config.silent_mode,  f"Time taken to calculate initial temperatures: {initial_temperatures_end - initial_temperatures_start:.2f} seconds")

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
                    config.emissivity,
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
                    config.emissivity,
                    colour_map='viridis', 
                    plot_title='Secondary Radiation Contribution', 
                    axis_label='Sum of View Factors', 
                    animation_frames=1, 
                    save_animation=False, 
                    save_animation_name='secondary_radiation.png', 
                    background_colour='black')

    conditional_print(config.silent_mode,  f"Running main simulation loop.\n")
    conditional_print(config.silent_mode,  f"Convergence target: {simulation.convergence_target} K with {config.convergence_method} convergence method.\n")

    # Run solver
    solver_start_time = time.time()
    result = solver.solve(thermal_data, shape_model, simulation, config)
    solver_end_time = time.time()
    solver_execution_time = solver_end_time - solver_start_time
    full_run_end_time = time.time()

    if result["final_timestep_temperatures"] is not None:
        conditional_print(config.silent_mode, f"Convergence target achieved after {result['days_to_convergence']} days.")
        conditional_print(config.silent_mode, f"Final temperature error: {result['mean_temperature_error']} K")
        conditional_print(config.silent_mode, f"Max temperature error: {result['max_temperature_error']} K")
    else:
        conditional_print(config.silent_mode, f"Model did not converge after {result['days_to_convergence']} days.")
        conditional_print(config.silent_mode, f"Final temperature error: {result['mean_temperature_error']} K")

    conditional_print(config.silent_mode, f"Solver execution time: {solver_execution_time} seconds")
    conditional_print(config.silent_mode, f"Full run time: {full_run_end_time - full_run_start_time} seconds")

    if config.plot_insolation_curve and not config.silent_mode:
        fig_temperature = plt.figure(figsize=(10, 6))
        conditional_print(config.silent_mode, f"Preparing temperature curve plot.\n")
        
        if config.plotted_facet_index >= len(shape_model):
            conditional_print(config.silent_mode, f"Facet index {config.plotted_facet_index} out of range. Please select a facet index between 0 and {len(shape_model) - 1}.")
        else:
            # Get the temp data for the facet
            temperature_data = result["final_day_temperatures"][config.plotted_facet_index]
            
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
        plt.plot(result["final_day_temperatures_all_layers"][config.plotted_facet_index])
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
        plt.plot(shape_model[config.plotted_facet_index].unphysical_energy_loss[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Unphysical energy loss')
        plt.plot(shape_model[config.plotted_facet_index].insolation_energy[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Insolation energy')
        plt.plot(shape_model[config.plotted_facet_index].re_emitted_energy[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Re-emitted energy')
        plt.plot(-shape_model[config.plotted_facet_index].surface_energy_change[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Surface energy change')
        plt.plot(shape_model[config.plotted_facet_index].conducted_energy[(result['days_to_convergence'] - 1) * simulation.timesteps_per_day:result['days_to_convergence'] * simulation.timesteps_per_day], label='Conducted energy')
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('Energy (J)')
        plt.title('Energy terms for facet for the final day')
        fig_energy_terms.show()

    # If roughness enabled, run sub-facet solves and aggregate before final-day animation
    if config.apply_kernel_based_roughness:
        print("\nSolving sub-facet temperatures for roughness depressions...")
        sub_solver = TemperatureSolverFactory.create(config.temp_solver)
        old_silent = config.silent_mode
        config.silent_mode = True
        for facet in conditional_tqdm(shape_model, config.silent_mode, desc="Solving sub-facet temps"):
            sub_solver.initialize_temperatures(facet.depression_thermal_data, simulation, config)
            facet.depression_temperature_result = sub_solver.solve(
                facet.depression_thermal_data,
                facet.sub_facets,
                simulation,
                config
            )
        config.silent_mode = old_silent
        print("Sub-facet temperature solving complete.")
        # Aggregate sub-facet final-day temperatures to parent facets
        n_parents = len(shape_model)
        parent_t_final = np.zeros((n_parents, simulation.timesteps_per_day), dtype=np.float64)
        for i, facet in enumerate(shape_model):
            sub_temps = facet.depression_temperature_result["final_day_temperatures"]
            mesh = Facet._canonical_subfacet_mesh
            area_vec = np.array([entry['area'] * facet.area for entry in mesh], dtype=np.float64)
            # Take T^4 weighted mean and convert back to T
            parent_t_final[i, :] = np.power((area_vec[:, None] * np.power(sub_temps, 4)).sum(axis=0) / area_vec.sum(), 0.25)
        result["final_day_temperatures"] = parent_t_final
        result["final_timestep_temperatures"] = parent_t_final[:, -1]
        # Sub-facet temperature animation (only available with kernel-based roughness)
        if config.animate_subfacets and config.apply_kernel_based_roughness:
            idx = config.subfacet_facet_index
            facet = shape_model[idx]
            N_sub = len(facet.sub_facets)
            # Reconstruct world-space sub-facet triangles
            scale = np.sqrt(facet.area)
            R_l2w = facet.dome_rotation.T
            tri = stl_mesh_module.Mesh(np.zeros(N_sub, dtype=stl_mesh_module.Mesh.dtype))
            for j, entry in enumerate(Facet._canonical_subfacet_mesh):
                local_tri = entry["vertices"] * scale
                world_tri = (R_l2w.dot(local_tri.T)).T + facet.position
                tri.vectors[j] = world_tri
            tmp_path = f"subfacet_{idx}.stl"
            tri.save(tmp_path)
            # Animate sub-facet temperatures over a full rotation
            check_remote_and_animate(
                config.remote,
                tmp_path,
                facet.depression_temperature_result["final_day_temperatures"],
                simulation.rotation_axis,
                simulation.sunlight_direction,
                simulation.timesteps_per_day,
                simulation.solar_distance_au,
                simulation.rotation_period_hours,
                config.emissivity,
                dome_radius_factor=config.kernel_dome_radius_factor,
                colour_map="coolwarm",
                plot_title=f"Subfacet Temps (facet {idx})",
                axis_label="Temperature (K)",
                animation_frames=simulation.timesteps_per_day,
                save_animation=False,
                save_animation_name=f"subfacet_{idx}_temps.gif",
                background_colour="white"
            )
        elif config.animate_subfacets and not config.apply_kernel_based_roughness:
            conditional_print(config.silent_mode, "Warning: Subfacet animation requires kernel-based roughness to be enabled.")

    if config.animate_final_day_temp_distribution:
        conditional_print(config.silent_mode,  f"Preparing temperature animation.\n")

        # Export subfacet and dome flux data only if kernel-based roughness is enabled
        if config.apply_kernel_based_roughness:
            # Build subfacet geometry and temps in memory (no separate file write)
            all_pts, all_faces, all_temps = [], [], []
            pt_index = 0
            for facet in conditional_tqdm(shape_model, config.silent_mode, desc="Building subfacet geometry"):
                # Subfacet triangles and temps
                scale = math.sqrt(facet.area)
                R_l2w = facet.dome_rotation.T
                temps_sf = facet.depression_temperature_result["final_day_temperatures"]  # (M,)
                for j, entry in enumerate(Facet._canonical_subfacet_mesh):
                    local_tri = entry['vertices'] * scale
                    world_tri = (R_l2w.dot(local_tri.T)).T + facet.position
                    
                    # FIX: Check normal orientation after transformation
                    v1, v2, v3 = world_tri[0], world_tri[1], world_tri[2]
                    edge1 = v2 - v1
                    edge2 = v3 - v1
                    actual_normal = np.cross(edge1, edge2)
                    norm_mag = np.linalg.norm(actual_normal)
                    
                    if norm_mag > 1e-12:
                        actual_normal = actual_normal / norm_mag
                        
                        # Expected normal should align with parent facet normal
                        expected_normal = facet.normal
                        alignment = np.dot(actual_normal, expected_normal)
                        
                        # If normal is backwards, flip vertex order
                        if alignment < 0:
                            world_tri = np.array([world_tri[0], world_tri[2], world_tri[1]])  # Swap v2 and v3
                    
                    all_pts.extend(world_tri)
                    all_faces.append([3, pt_index, pt_index+1, pt_index+2])
                    pt_index += 3
                    all_temps.append(temps_sf[j])   

            # Compute final-day dome thermal flux arrays for animation (kept in memory)
            sigma = 5.670374419e-8  # Stefan-Boltzmann constant
            submesh = Facet._canonical_subfacet_mesh
            F_sd = Facet._canonical_F_sd
            M = len(submesh)
            nF = len(shape_model)
            T = simulation.timesteps_per_day
            dome_flux_th = np.zeros((nF, len(Facet._canonical_dome_mesh), T), dtype=np.float64)
            # Compute emissive power per subfacet and project onto dome bins
            for i, facet in conditional_tqdm(enumerate(shape_model), config.silent_mode, desc="Computing dome fluxes", total=len(shape_model)):
                temps_sf = facet.depression_temperature_result["final_day_temperatures"]  # shape (N_subfacets, T)
                sub_areas = np.array([entry['area'] * facet.area for entry in submesh], dtype=np.float64)
                # Radiative power W per subfacet (use simulation.emissivity)
                E_sub = simulation.emissivity * sigma * (temps_sf**4) * sub_areas[:, None]
                # Project to dome bins
                dome_flux_th[i] = F_sd.T.dot(E_sub)
            # Compute dome metadata (kept in memory)
            dome_normals = np.array([entry['normal'] for entry in Facet._canonical_dome_mesh])
            canonical_areas = np.array([entry['area'] for entry in Facet._canonical_dome_mesh])
            parent_areas = np.array([facet.area for facet in shape_model])
            dome_bin_areas = canonical_areas[None, :] * parent_areas[:, None] * config.kernel_dome_radius_factor**2
            verts2 = np.array([entry['vertices'] for entry in Facet._canonical_dome_mesh])
            norms2 = np.linalg.norm(verts2, axis=2)[:, :, None]
            unit_verts2 = verts2 / norms2
            solid_angles2 = []
            for tri in unit_verts2:
                u1, u2, u3 = tri
                num = np.linalg.det(np.stack((u1, u2, u3)))
                den = 1.0 + np.dot(u1, u2) + np.dot(u2, u3) + np.dot(u3, u1)
                solid_angles2.append(2.0 * math.atan2(num, den))
            solid_angles2 = np.array(solid_angles2)
        
        if config.apply_kernel_based_roughness:
            # Combine all animation data into one timestamped .h5 for roughness animations & TESBY
            loc = Locations()
            loc.ensure_directories_exist()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = os.path.join(loc.remote_outputs, f"animation_outputs_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            combined_h5 = os.path.join(output_dir, f"combined_animation_data_rough_{timestamp}.h5")
            with h5py.File(combined_h5, 'w') as cf:
                # — subfacet geometry + temps —
                subgrp = cf.create_group('subfacet_data')
                subgrp.create_dataset('points', data=np.array(all_pts,   dtype=np.float64))
                subgrp.create_dataset('faces',  data=np.array(all_faces, dtype=np.int64))
                subgrp.create_dataset('temps',  data=np.array(all_temps, dtype=np.float64))
                # — dome flux data —
                domegrp = cf.create_group('dome_fluxes')
                domegrp.create_dataset('dome_flux_th',           data=dome_flux_th)
                domegrp.create_dataset('dome_normals',           data=dome_normals)
                domegrp.create_dataset('dome_bin_areas',         data=dome_bin_areas)
                domegrp.create_dataset('dome_bin_solid_angles',  data=solid_angles2)
                # — copy out all animation parameters for TESBY —
                paramgrp = cf.create_group('animation_params')
                paramgrp.create_dataset('rotation_axis',        data=simulation.rotation_axis)
                paramgrp.create_dataset('sunlight_direction',   data=simulation.sunlight_direction)
                paramgrp.create_dataset('timesteps_per_day',    data=simulation.timesteps_per_day)
                paramgrp.create_dataset('solar_distance_au',    data=simulation.solar_distance_au)
                paramgrp.create_dataset('rotation_period_hours',data=simulation.rotation_period_hours)
                paramgrp.create_dataset('emissivity',           data=config.emissivity)
                paramgrp.attrs['apply_kernel_based_roughness']  = config.apply_kernel_based_roughness
                paramgrp.attrs['dome_radius_factor']            = config.kernel_dome_radius_factor
                # — embed what plotting.py would store in NPZ —
                animio = cf.create_group('animation_io')
                # For rough animations we normally plot parent final-day temperatures
                plot_array = result["final_day_temperatures"]
                animio.create_dataset('plotted_variable_array', data=plot_array)
                animio.create_dataset('rotation_axis', data=simulation.rotation_axis)
                animio.create_dataset('sunlight_direction', data=simulation.sunlight_direction)
            conditional_print(config.silent_mode,
                              f"Combined animation data saved to {combined_h5}")
            # Use the same output_dir so remote files land together
            check_remote_and_animate(
                config.remote, config.path_to_shape_model_file, 
                plot_array, 
                simulation.rotation_axis, 
                simulation.sunlight_direction, 
                simulation.timesteps_per_day,
                simulation.solar_distance_au,              
                simulation.rotation_period_hours,              
                config.emissivity,
                output_dir=output_dir,
                apply_kernel_based_roughness=config.apply_kernel_based_roughness,
                dome_radius_factor=config.kernel_dome_radius_factor,
                colour_map='coolwarm', 
                plot_title='Temperature distribution', 
                axis_label='Temperature (K)', 
                animation_frames=200, 
                save_animation=False, 
                save_animation_name='temperature_animation.gif', 
                background_colour='black')
            conditional_print(config.silent_mode, "Animation window closed, continuing...")
        else:
            # Smooth case: no combined rough HDF5; just animate
            plot_array = result["final_day_temperatures"]
            check_remote_and_animate(
                config.remote, config.path_to_shape_model_file, 
                plot_array, 
                simulation.rotation_axis, 
                simulation.sunlight_direction, 
                simulation.timesteps_per_day,
                simulation.solar_distance_au,              
                simulation.rotation_period_hours,              
                config.emissivity,
                apply_kernel_based_roughness=config.apply_kernel_based_roughness,
                dome_radius_factor=config.kernel_dome_radius_factor,
                colour_map='coolwarm', 
                plot_title='Temperature distribution', 
                axis_label='Temperature (K)', 
                animation_frames=200, 
                save_animation=False, 
                save_animation_name='temperature_animation.gif', 
                background_colour='black')
            conditional_print(config.silent_mode, "Animation window closed, continuing...")

    if config.plot_final_day_comparison and not config.silent_mode:
        conditional_print(config.silent_mode,  f"Saving final day temperatures to CSV file.\n")
        np.savetxt("final_day_temperatures.csv", np.column_stack((np.linspace(0, 2 * np.pi, simulation.timesteps_per_day), result["final_day_temperatures"][config.plotted_facet_index])), delimiter=',', header='Rotation angle (rad), Temperature (K)', comments='')

        thermprojrs_data = np.loadtxt("thermprojrs_data.csv", delimiter=',', skiprows=1)

        fig_model_comparison = plt.figure()
        plt.plot(thermprojrs_data[:, 0], thermprojrs_data[:, 1], label='Thermprojrs')
        plt.plot(np.linspace(0, 2 * np.pi, simulation.timesteps_per_day), result["final_day_temperatures"][config.plotted_facet_index], label='This model')
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Temperature (K)')
        plt.title('Final day temperature distribution for facet')
        plt.legend()
        fig_model_comparison.show()

        x_original = np.linspace(0, 2 * np.pi, simulation.timesteps_per_day)
        x_new = np.linspace(0, 2 * np.pi, thermprojrs_data.shape[config.plotted_facet_index])

        interp_func = interp1d(x_new, thermprojrs_data[:, 1], kind='linear')
        thermprojrs_interpolated = interp_func(x_original)

        plt.plot(x_original, result["final_day_temperatures"][config.plotted_facet_index] - thermprojrs_interpolated, label='This model')
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Temperature difference (K)')
        plt.title('Temperature difference between this model and Thermprojrs for facet')
        plt.legend()
        plt.show()

        np.savetxt("final_day.csv", np.column_stack((x_original, result["final_day_temperatures"][config.plotted_facet_index])), delimiter=',', header='Rotation angle (rad), Temperature (K)', comments='')

    if config.calculate_visible_phase_curve:
        phase_angles, brightness_values = calculate_phase_curve(
            shape_model,
            simulation,
            thermal_data,
            config,
            phase_curve_type='visible',
            observer_distance=1e9,
            normalized=True,
            plot=True
        )

    # Save the visible phase curve data to a CSV file
    if config.save_visible_phase_curve_data:
        locations = Locations()
        locations.ensure_directories_exist()  # Ensure output directories exist
        # Create name using shape model name and time
        filename = os.path.basename(config.path_to_shape_model_file).replace('.stl', '')
        output_csv_path = os.path.join(locations.phase_curve_data, f'{filename}_visible_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
        df = pd.DataFrame({
            'Phase Angle (degrees)': phase_angles,
            'Brightness Value': brightness_values
        })
        df.to_csv(output_csv_path, index=False)

        # Create and save the phase curve plot
        plt.figure()  # Create a new figure
        plt.plot(phase_angles, brightness_values, label='Brightness vs Phase Angle')  # Plot the data
        plt.xlabel('Phase Angle (degrees)')
        plt.ylabel('Brightness Value')
        plt.title('Visible Phase Curve')
        plt.legend()
        
        # Save the plot
        output_image_path = output_csv_path.replace('.csv', '.png')
        plt.savefig(output_image_path)  # Save the figure as a .png file
        plt.close()  # Close the figure after saving to avoid displaying it in non-interactive environments

        conditional_print(config.silent_mode,  f"Visible phase curve data exported to {output_csv_path}")

    if config.calculate_thermal_phase_curve:
        phase_angles, brightness_values = calculate_phase_curve(
            shape_model,
            simulation,
            thermal_data,
            config,
            phase_curve_type='thermal',
            observer_distance=1e8,
            normalized=False,
            plot=config.show_thermal_phase_curve
        )

    # Save the thermal phase curve data to a CSV file
    if config.save_thermal_phase_curve_data:
        locations = Locations()
        locations.ensure_directories_exist()  # Ensure output directories exist
        # Create name using shape model name and time
        filename = os.path.basename(config.path_to_shape_model_file).replace('.stl', '')
        output_csv_path = os.path.join(locations.phase_curve_data, f'{filename}_thermal_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
        df = pd.DataFrame({
            'Phase Angle (degrees)': phase_angles,
            'Brightness Value': brightness_values
        })
        df.to_csv(output_csv_path, index=False)
        
        # Create and save the thermal phase curve plot
        plt.figure()  # Create a new figure
        plt.plot(phase_angles, brightness_values, label='Thermal Brightness vs Phase Angle')
        plt.xlabel('Phase Angle (degrees)')
        plt.ylabel('Thermal Brightness Value')
        plt.title('Thermal Phase Curve')
        plt.legend()
        
        # Save the plot
        output_image_path = output_csv_path.replace('.csv', '.png')
        plt.savefig(output_image_path)  # Save the figure as a .png file
        plt.close()  # Close the figure after saving to avoid displaying it in non-interactive environments

        # Notify user
        conditional_print(config.silent_mode, f"Thermal phase curve data exported to {output_csv_path}")

    conditional_print(config.silent_mode,  f"Model run complete.\n")

    # Export insolation and temperature data for facets 0 and 1
    print("Exporting insolation and temperature for facets 0 and 1...")
    degrees = np.linspace(0, 360, simulation.timesteps_per_day, endpoint=False)
    os.makedirs('insolation_data', exist_ok=True)
    os.makedirs('temperature_data', exist_ok=True)
    for idx in [0, 1]:
        try:
            df_ins = pd.DataFrame({'rotation_deg': degrees, 'insolation_Wm2': thermal_data.insolation[idx]})
            df_ins.to_csv(f'insolation_data/facet_{idx}.csv', index=False)
            df_temp = pd.DataFrame({'rotation_deg': degrees, 'temperature_K': result['final_day_temperatures'][idx]})
            df_temp.to_csv(f'temperature_data/facet_{idx}.csv', index=False)
            print(f"  facet {idx}: insolation_data/facet_{idx}.csv, temperature_data/facet_{idx}.csv")
        except Exception as e:
            print(f"  Error exporting insolation and temperature for facet {idx}: {e}")

# Call the main program with interrupt handling
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nRun interrupted by user (Ctrl-C). Exiting.")
        sys.exit(1)
