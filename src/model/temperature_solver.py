# src/model/temperature_solver.py

import numpy as np
from numba import jit
from joblib import Parallel, delayed
from tqdm import tqdm
from src.utilities.utils import conditional_print, conditional_tqdm

@jit(nopython=True)
def calculate_temperatures(temperatures, layer_temperatures, insolation, visible_facets_list, 
                         view_factors_list, const1, const2, const3, self_heating_const,
                         timesteps_per_day, n_layers, include_self_heating):
    
    n_facets = temperatures.shape[0]
    current_column = 0  # Use column 0 for current timestep
    prev_column = 1    # Use column 1 for previous timestep
    
    for time_step in range(timesteps_per_day):
        # Swap columns for next iteration
        current_column, prev_column = prev_column, current_column
        
        for i in range(n_facets):
            # Surface temperature calculation
            prev_temp = layer_temperatures[i, prev_column, 0]
            prev_temp_layer1 = layer_temperatures[i, prev_column, 1]

            insolation_term = insolation[i, time_step] * const1
            re_emitted_radiation_term = -const2 * (prev_temp**4)
            
            secondary_radiation_term = 0.0
            if include_self_heating:
                secondary_radiation_term = calculate_secondary_radiation(
                    layer_temperatures[:, prev_column, 0], 
                    visible_facets_list[i], 
                    view_factors_list[i], 
                    self_heating_const
                )
            
            conducted_heat_term = const3 * (prev_temp_layer1 - prev_temp)
            
            new_temp = (prev_temp + 
                       insolation_term + 
                       re_emitted_radiation_term + 
                       conducted_heat_term + 
                       secondary_radiation_term)

            temperatures[i, time_step] = new_temp
            layer_temperatures[i, current_column, 0] = new_temp
            
            # Update subsurface temperatures
            for layer in range(1, n_layers - 1):
                prev_layer = layer_temperatures[i, prev_column, layer]
                prev_layer_plus = layer_temperatures[i, prev_column, layer + 1]
                prev_layer_minus = layer_temperatures[i, prev_column, layer - 1]

                layer_temperatures[i, current_column, layer] = (
                    prev_layer + 
                    const3 * (prev_layer_plus - 
                             2 * prev_layer + 
                             prev_layer_minus)
                )

    return temperatures

def iterative_temperature_solver(thermal_data, shape_model, simulation, config):
    ''' 
    This is the main calculation function for the thermophysical body model. It calls the necessary functions to read in the shape model, set material and model properties, calculate 
    insolation and temperature arrays, and iterate until the model converges.
    '''

    convergence_error = simulation.convergence_target + 1
    day = 0 
    temperature_error = 0

    const1 = simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
    const2 = simulation.emissivity * simulation.beaming_factor * 5.67e-8 * simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
    const3 = simulation.thermal_diffusivity * simulation.delta_t / simulation.layer_thickness**2
    self_heating_const = 5.670374419e-8 * simulation.delta_t * simulation.emissivity / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity * np.pi)

    error_history = []
    comparison_temps = thermal_data.temperatures[:, 0].copy()

    while day < simulation.max_days and (day < simulation.min_days or convergence_error > simulation.convergence_target):
        current_day_temperature = calculate_temperatures(
            thermal_data.temperatures,
            thermal_data.layer_temperatures,
            thermal_data.insolation,
            thermal_data.visible_facets,
            thermal_data.secondary_radiation_view_factors,
            const1, const2, const3, self_heating_const, 
            simulation.timesteps_per_day, simulation.n_layers,
            config.include_self_heating
        )

        # Check for invalid temperatures
        for i in range(len(shape_model)):
            for time_step in range(simulation.timesteps_per_day):
                if np.isnan(current_day_temperature[i, time_step]) or np.isinf(current_day_temperature[i, time_step]) or current_day_temperature[i, time_step] < 0:
                    conditional_print(config.silent_mode, f"Invalid temperature {current_day_temperature[i, time_step]} K detected for facet {i} at timestep {time_step}")
                    return None, None, None, day, temperature_error, None

        if config.calculate_energy_terms:
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
            
            thermal_data.insolation_energy = energy_terms[:, :, 0]
            thermal_data.re_emitted_energy = energy_terms[:, :, 1]
            thermal_data.surface_energy_change = energy_terms[:, :, 2]
            thermal_data.conducted_energy = energy_terms[:, :, 3]
            thermal_data.unphysical_energy_loss = energy_terms[:, :, 4]

        # Calculate convergence
        temperature_errors = np.abs(current_day_temperature[:, 0] - comparison_temps)
        
        if config.convergence_method == 'mean':
            convergence_error = np.mean(temperature_errors)
        else:
            convergence_error = np.max(temperature_errors)

        max_temperature_error = np.max(temperature_errors)
        mean_temperature_error = np.mean(temperature_errors)

        conditional_print(config.silent_mode, f"Day: {day} | Mean Temperature error: {mean_temperature_error:.6f} K | Max Temp Error: {max_temperature_error:.6f} K")
        
        comparison_temps = current_day_temperature[:, 0].copy()
        error_history.append(convergence_error)
        day += 1

    # Store final day temperatures for return
    final_day_temperatures = current_day_temperature
    final_timestep_temperatures = current_day_temperature[:, -1]
    final_day_temperatures_all_layers = thermal_data.layer_temperatures

    if convergence_error < simulation.convergence_target:
        conditional_print(config.silent_mode, f"Convergence achieved after {day} days.")
    else:
        conditional_print(config.silent_mode, f"Maximum days reached without achieving convergence.")
        conditional_print(config.silent_mode, f"Final temperature error: {mean_temperature_error} K")

    return (final_day_temperatures, 
            final_day_temperatures_all_layers, 
            final_timestep_temperatures, 
            day, 
            mean_temperature_error, 
            max_temperature_error)

@jit(nopython=True)
def calculate_energy_terms(temperature, insolation, delta_t, emissivity, beaming_factor,
                           density, specific_heat_capacity, layer_thickness, thermal_conductivity,
                           timesteps_per_day, n_layers):
    energy_terms = np.zeros((len(temperature), timesteps_per_day, 5))
    for i in range(len(temperature)):
        for time_step in range(timesteps_per_day):
            energy_terms[i, time_step, 0] = insolation[i, time_step] * delta_t
            energy_terms[i, time_step, 1] = -emissivity * beaming_factor * 5.670374419e-8 * (temperature[i, time_step]**4) * delta_t
            energy_terms[i, time_step, 2] = -density * specific_heat_capacity * layer_thickness * (temperature[i, (time_step + 1) % timesteps_per_day] - temperature[i, time_step])
            energy_terms[i, time_step, 3] = thermal_conductivity * delta_t * (temperature[i, time_step, 1] - temperature[i, time_step, 0]) / layer_thickness
            energy_terms[i, time_step, 4] = energy_terms[i, time_step, 0] + energy_terms[i, time_step, 1] + energy_terms[i, time_step, 2] + energy_terms[i, time_step, 3]
    return energy_terms

def calculate_initial_temperatures(thermal_data, silent_mode, emissivity, n_jobs=-1):
    ''' 
    This function calculates the initial temperature of each facet and sub-surface layer of the body based on the insolation curve for that facet.
    '''
    # Stefan-Boltzmann constant
    sigma = 5.67e-8

    def process_facet(insolation, emissivity, sigma):
        # Calculate the initial temperature based on average power in
        power_in = np.mean(insolation)
        # Calculate the temperature of the facet using the Stefan-Boltzmann law
        calculated_temp = (power_in / (emissivity * sigma))**(1/4)
        return calculated_temp

    # Parallel processing of facets
    results = Parallel(n_jobs=n_jobs)(delayed(process_facet)(thermal_data.insolation[i], emissivity, sigma) 
                                    for i in range(thermal_data.temperatures.shape[0]))

    conditional_print(silent_mode, f"Initial temperatures calculated for {thermal_data.temperatures.shape[0]} facets.")

    # Update both surface and layer temperatures with initial values
    for i, temperature in conditional_tqdm(enumerate(results), silent_mode, total=len(results), desc='Saving temps'):
        # Set surface temperatures for all timesteps
        thermal_data.temperatures[i, :] = temperature
        # Set layer temperatures for both columns and all layers
        thermal_data.layer_temperatures[i, 0, :] = temperature
        thermal_data.layer_temperatures[i, 1, :] = temperature

    conditional_print(silent_mode, "Initial temperatures saved for all facets.")

    return thermal_data

@jit(nopython=True)
def calculate_secondary_radiation(temperatures, visible_facets, view_factors, self_heating_const):
    return self_heating_const * np.sum(temperatures[visible_facets]**4 * view_factors)