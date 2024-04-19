"""
Thermophysical Model Multi-Parameter Exploration Script

This script facilitates the exploration of thermophysical models by varying multiple key parameters simultaneously,
allowing for a comprehensive analysis of different parameter combinations on model outcomes.

Dependencies:
- numpy
- tqdm
- itertools
- pandas
- seaborn
- matplotlib
- thermophysical_body_model module and its components

Author: Duncan Lyster
"""

import concurrent.futures
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tqdm
from itertools import product, combinations
from thermophysical_body_model import (
    thermophysical_body_model,
    Simulation,
    read_shape_model,
    calculate_visible_facets,
    calculate_insolation,
    calculate_initial_temperatures,
    calculate_secondary_radiation_coefficients
)
from sklearn.linear_model import LinearRegression

def run_model_with_parameters(combination, shape_model_snapshot, path_to_setup_file, param_names):
    simulation = Simulation(path_to_setup_file)  # Create a new simulation instance for thread safety

    # Update simulation parameters for the current combination
    for param_name, value in zip(param_names, combination):
        setattr(simulation, param_name, value)
    
    # Run the thermophysical model and calculate statistics
    start_time = time.time()
    final_timestep_temperatures = thermophysical_body_model(shape_model_snapshot, simulation)
    execution_time = time.time() - start_time

    return {
        'parameters': dict(zip(param_names, combination)),
        'mean_temperature': np.mean(final_timestep_temperatures),
        'temperature_iqr': np.percentile(final_timestep_temperatures, 75) - np.percentile(final_timestep_temperatures, 25),
        'execution_time': execution_time
    }

def generate_heatmaps(df, parameters, value_column, output_folder):
    for param_pair in combinations(parameters, 2):  # Get all combinations of parameter pairs
        pivot_table = df.pivot_table(index=param_pair[0], columns=param_pair[1], values=value_column, aggfunc="mean")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': value_column})
        plt.title(f'{value_column} by {param_pair[0]} and {param_pair[1]}')
        plt.xlabel(param_pair[1])
        plt.ylabel(param_pair[0])
        plt.savefig(f"{output_folder}heatmap_{param_pair[0]}_vs_{param_pair[1]}.png")  # Save each heatmap as a PNG file
        plt.close()  # Close the figure to avoid displaying it inline if running in a notebook

def main():
    # Model setup and initialization
    shape_model_name = "5km_ico_sphere_80_facets.stl"
    path_to_shape_model_file = f"shape_models/{shape_model_name}"
    path_to_setup_file = "model_setups/generic_model_parameters.json"
    simulation = Simulation(path_to_setup_file)
    shape_model = read_shape_model(path_to_shape_model_file, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days)

    # Model component calculations
    shape_model = calculate_visible_facets(shape_model)
    shape_model = calculate_insolation(shape_model, simulation)
    shape_model = calculate_initial_temperatures(shape_model, simulation.n_layers, simulation.emissivity)
    shape_model = calculate_secondary_radiation_coefficients(shape_model)

    # Parameters for analysis with their ranges
    parameter_ranges = {
        'emissivity': np.linspace(0, 1, 100),
        'albedo': np.linspace(0, 1, 1000),
        'thermal_conductivity': np.linspace(0, 1, 100),
        'density': np.linspace(450, 550, 100), 
        'specific_heat_capacity': np.linspace(500, 2500, 100),
        'beaming_factor': np.linspace(0.9, 1.1,  100),
        'layer_thickness': [0.1],
        'n_layers': [32],
        'solar_distance': np.linspace(0.5, 1.5, 100),
        'solar_luminosity': np.linspace(3E26, 3.5E26, 100),
        'sunlight_direction': [1,0,0],
        'timesteps': [100],
        'rotation_period': [12],
        'max_days': [10],
        'convergence_target': [0.1],
        'ra_degrees': [0],
        'dec_degrees': [90],
    }

    param_names = list(parameter_ranges.keys())
    param_values = [parameter_ranges[name] for name in param_names]
    all_combinations = list(product(*param_values))

    # Include param_names with each combination
    param_combinations_with_shape_model = [
        (combination, shape_model, path_to_setup_file, param_names) for combination in all_combinations
    ]

    # Parallel exploration execution
    print("Starting parameter exploration...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_model_with_parameters, *params) for params in param_combinations_with_shape_model]
        results = list(tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)))

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results)

    # Flatten the 'parameters' dictionary into separate columns for easier analysis
    parameters_df = pd.json_normalize(results_df['parameters'])
    results_df = pd.concat([results_df.drop(columns=['parameters']), parameters_df], axis=1)

    output_folder = "runner_outputs/"

    # Call the function for generating and saving heatmaps
    generate_heatmaps(results_df, param_names, 'mean_temperature', output_folder)

    print("Heatmaps generated and saved.")

    csv_file_path = "runner_outputs/thermophysical_model_results.csv"

    # Save the DataFrame to a CSV file
    results_df.to_csv(csv_file_path, index=False)

    print("Results saved to 'thermophysical_model_results.csv'.")

if __name__ == '__main__':
    main()
