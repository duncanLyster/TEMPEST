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

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from numba.typed import List
from stl import mesh
from scipy.interpolate import interp1d
from datetime import datetime

# Import modules from src
from src.model.calculate_phase_curve import calculate_phase_curve
from src.model.insolation import calculate_insolation
from src.model.simulation import Simulation, ThermalData
from src.model.facet import Facet
from src.utilities.config import Config
from src.utilities.utils import (
    calculate_black_body_temp,
    conditional_print
)
from src.model.view_factors import (
    calculate_and_cache_visible_facets,
    calculate_all_view_factors_parallel,
    calculate_all_view_factors
)
from src.model.temperature_solver import (
    iterative_temperature_solver,
    calculate_initial_temperatures
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

def apply_roughness(shape_model, simulation, config, subdivision_levels, displacement_factors=None):
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

def main(silent_mode=False):
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.
    '''

    full_run_start_time = time.time()

    # Load user configuration
    config = Config()

    # Load setup parameters from JSON file
    simulation = Simulation(config)

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
        shape_model = apply_roughness(shape_model, simulation, config, config.subdivision_levels, config.displacement_factors)
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

        calculate_view_factors_start = time.time()

        if config.n_jobs == 1:
            # Use serial version
            conditional_print(silent_mode, "Calculating view factors...")
            all_view_factors = calculate_all_view_factors(shape_model, thermal_data, simulation, config, config.vf_rays)
        else:
            # Use parallel version
            actual_n_jobs = config.validate_jobs()
            all_view_factors = calculate_all_view_factors_parallel(shape_model, thermal_data, config, config.vf_rays)

        calculate_view_factors_end = time.time()

        conditional_print(silent_mode, f"Time taken to calculate view factors: {calculate_view_factors_end - calculate_view_factors_start:.2f} seconds")
        
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

    initial_temperatures_start = time.time()
    thermal_data = calculate_initial_temperatures(thermal_data, config.silent_mode, simulation.emissivity)
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
    solver_start_time = time.time()
    final_day_temperatures, final_day_temperatures_all_layers, final_timestep_temperatures, day, temperature_error, max_temp_error = iterative_temperature_solver(thermal_data, shape_model, simulation, config)
    solver_end_time = time.time()
    full_run_end_time = time.time()
    solver_execution_time = solver_end_time - solver_start_time

    if final_timestep_temperatures is not None:
        conditional_print(config.silent_mode,  f"Convergence target achieved after {day} days.")
        conditional_print(config.silent_mode,  f"Final temperature error: {temperature_error / len(shape_model)} K")
        conditional_print(config.silent_mode,  f"Max temperature error for any facet: {max_temp_error} K")
    else:
        conditional_print(config.silent_mode,  f"Model did not converge after {day} days.")
        conditional_print(config.silent_mode,  f"Final temperature error: {temperature_error / len(shape_model)} K")

    conditional_print(config.silent_mode,  f"Solver execution time: {solver_execution_time} seconds")
    conditional_print(config.silent_mode,  f"Full run time: {full_run_end_time - full_run_start_time} seconds")

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

    if config.calculate_visible_phase_curve:
        phase_angles, brightness_values = calculate_phase_curve(
            shape_model,
            simulation,
            thermal_data,
            phase_curve_type='visible',
            observer_distance=1e6,
            normalized=False,
            plot=config.show_visible_phase_curve
        )

    # Save the visible phase curve data to a CSV file
    if config.save_visible_phase_curve_data:
        output_dir = 'visible_phase_curve_data'
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        # Create name using shape model name, roughness parameters, and time
        filename = os.path.basename(config.path_to_shape_model_file).replace('.stl', '')
        output_csv_path = os.path.join(output_dir, f'{filename}_{config.subdivision_levels}_{config.displacement_factors}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
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
            phase_curve_type='thermal',
            observer_distance=1e8,
            normalized=False,
            plot=config.show_thermal_phase_curve
        )

    # Save the thermal phase curve data to a CSV file
    if config.save_thermal_phase_curve_data:
        output_dir = 'thermal_phase_curve_data'
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        filename = os.path.basename(config.path_to_shape_model_file).replace('.stl', '')
        output_csv_path = os.path.join(output_dir, f'{filename}_{config.subdivision_levels}_{config.displacement_factors}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
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

# Call the main program to start execution
if __name__ == "__main__":
    main()
