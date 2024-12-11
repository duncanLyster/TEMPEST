# src/model/temperature_solver.py

import sys
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utilities.utils import conditional_print, conditional_tqdm, calculate_black_body_temp
from src.utilities.plotting.plotting import check_remote_and_animate
from src.utilities.locations import Locations
import os

@jit(nopython=True)
def calculate_temperatures(temperature, insolation, visible_facets_list, view_factors_list, 
                           const1, const2, const3, self_heating_const, 
                           timesteps_per_day, n_layers, include_self_heating,
                           start_index, end_index):
    
    current_day_temperature = temperature[:, start_index:end_index, :].copy()
    day_length = end_index - start_index
    
    for time_step in range(day_length):
        prev_step = time_step - 1 if time_step > 0 else -1
        for i in range(len(current_day_temperature)):
            if time_step == 0:
                prev_temp = temperature[i, start_index - 1, 0] if start_index > 0 else current_day_temperature[i, 0, 0]
                prev_temp_layer1 = temperature[i, start_index - 1, 1] if start_index > 0 else current_day_temperature[i, 0, 1]
            else:
                prev_temp = current_day_temperature[i, prev_step, 0]
                prev_temp_layer1 = current_day_temperature[i, prev_step, 1]

            insolation_term = insolation[i, time_step] * const1
            re_emitted_radiation_term = -const2 * (prev_temp**4)
            
            secondary_radiation_term = 0.0
  
            if include_self_heating:
                secondary_radiation_term = calculate_secondary_radiation(current_day_temperature[:, prev_step, 0], visible_facets_list[i], view_factors_list[i], self_heating_const)
            
            conducted_heat_term = const3 * (prev_temp_layer1 - prev_temp)
            
            new_temp = (prev_temp + 
                        insolation_term + 
                        re_emitted_radiation_term + 
                        conducted_heat_term + 
                        secondary_radiation_term)

            current_day_temperature[i, time_step, 0] = new_temp
            
            # Update subsurface temperatures, excluding the deepest layer
            for layer in range(1, n_layers - 1):
                if time_step == 0:
                    prev_layer = temperature[i, start_index - 1, layer] if start_index > 0 else current_day_temperature[i, 0, layer]
                    prev_layer_plus = temperature[i, start_index - 1, layer + 1] if start_index > 0 else current_day_temperature[i, 0, layer + 1]
                    prev_layer_minus = temperature[i, start_index - 1, layer - 1] if start_index > 0 else current_day_temperature[i, 0, layer - 1]
                else:
                    prev_layer = current_day_temperature[i, prev_step, layer]
                    prev_layer_plus = current_day_temperature[i, prev_step, layer + 1]
                    prev_layer_minus = current_day_temperature[i, prev_step, layer - 1]

                current_day_temperature[i, time_step, layer] = (
                    prev_layer + 
                    const3 * (prev_layer_plus - 
                              2 * prev_layer + 
                              prev_layer_minus)
                )
    
    return current_day_temperature

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

    # Set comparison temperatures to the initial temperatures of the first timestep (day 0, timestep 0)
    comparison_temps = thermal_data.temperatures[:, 0, 0].copy()

    while day < simulation.max_days and (day < simulation.min_days or convergence_error > simulation.convergence_target):
        current_day_start = day * simulation.timesteps_per_day
        current_day_end = (day + 1) * simulation.timesteps_per_day
        next_day_start = current_day_end
                        
        current_day_temperature = calculate_temperatures(
            thermal_data.temperatures,
            thermal_data.insolation,
            thermal_data.visible_facets,
            thermal_data.secondary_radiation_view_factors,
            const1, const2, const3, self_heating_const, 
            simulation.timesteps_per_day, simulation.n_layers,
            config.include_self_heating,
            current_day_start, current_day_end
        )

        thermal_data.temperatures[:, current_day_start:current_day_end, :] = current_day_temperature

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

        # Check for invalid temperatures
        for i in range(thermal_data.temperatures.shape[0]):
            for time_step in range(simulation.timesteps_per_day):
                current_step = int(time_step + (day * simulation.timesteps_per_day))
                if np.isnan(current_day_temperature[i, time_step, 0]) or np.isinf(current_day_temperature[i, time_step, 0]) or current_day_temperature[i, time_step, 0] < 0 and not config.silent_mode:
                    conditional_print(config.silent_mode,  f"Ending run at timestep {current_step} due to facet {i} having a temperature of {current_day_temperature[i, time_step, 0]} K.\n Try increasing the number of time steps per day")

                    # Plot the energy terms for the facet
                    if config.calculate_energy_terms:
                        plt.plot(thermal_data.insolation_energy[i, :current_step], label="Insolation energy")
                        plt.plot(thermal_data.re_emitted_energy[i, :current_step], label="Re-emitted energy")
                        plt.plot(thermal_data.surface_energy_change[i, :current_step], label="Surface energy change")
                        plt.plot(thermal_data.conducted_energy[i, :current_step], label="Conducted energy")
                        plt.plot(thermal_data.unphysical_energy_loss[i, :current_step], label="Unphysical energy loss")
                        plt.legend()
                        plt.xlabel('Timestep')
                        plt.ylabel('Energy (J)')
                        plt.title(f'Energy terms for facet {i}')
                        plt.show()

                    # Plot the insolation curve for the facet
                    plt.plot(thermal_data.insolation[i])
                    plt.xlabel('Number of timesteps')
                    plt.ylabel('Insolation (W/m^2)')
                    plt.title(f'Insolation curve for facet {i}')
                    plt.show()

                    # Plot sub-surface temperatures for the facet
                    for layer in range(1, simulation.n_layers):
                        plt.plot(thermal_data.temperatures[i, :current_step+100, layer])
                    plt.xlabel('Number of timesteps')
                    plt.ylabel('Temperature (K)')
                    plt.title(f'Sub-surface temperature for facet {i}')
                    plt.legend([f"Layer {layer}" for layer in range(1, simulation.n_layers)])
                    plt.show()

                    # Create an array of 0s for all facets for all time steps in the day
                    facet_highlight_array = np.zeros((thermal_data.temperatures.shape[0], simulation.timesteps_per_day))
                    facet_highlight_array[i] = 1

                    check_remote_and_animate(config.remote, 
                                  config.path_to_shape_model_file, 
                                  facet_highlight_array, 
                                  simulation.rotation_axis, 
                                  simulation.sunlight_direction, 
                                  simulation.timesteps_per_day, 
                                  simulation.solar_distance_au,
                                  simulation.rotation_period_hours,
                                  colour_map='coolwarm', plot_title='Problematic facet', axis_label='Problem facet is red', animation_frames=200, save_animation=False, save_animation_name='problematic_facet_animation.gif', background_colour = 'black')

                    sys.exit()

        # Calculate convergence factor
        temperature_errors = np.abs(current_day_temperature[:, 0, 0] - comparison_temps)

        if config.convergence_method == 'mean':
            convergence_error = np.mean(temperature_errors)
        elif config.convergence_method == 'max':
            convergence_error = np.max(temperature_errors)
        else:
            raise ValueError("Invalid convergence_method. Use 'mean' or 'max'.")

        max_temperature_error = np.max(temperature_errors)
        mean_temperature_error = np.mean(temperature_errors) 

        conditional_print(config.silent_mode, f"Day: {day} | Mean Temperature error: {mean_temperature_error:.6f} K | Max Temp Error: {max_temperature_error:.6f} K")

        comparison_temps = current_day_temperature[:, 0, 0].copy()
        
        error_history.append(convergence_error)
        day += 1

    # Decrement the day counter
    day -= 1

    # Remove unused days from thermal_data
    thermal_data.temperatures = thermal_data.temperatures[:, :simulation.timesteps_per_day * (day+1), :]

    final_day_temperatures = thermal_data.temperatures[:, -simulation.timesteps_per_day:, 0]

    final_timestep_temperatures = thermal_data.temperatures[:, -1, 0]
    final_day_temperatures_all_layers = thermal_data.temperatures[:, -simulation.timesteps_per_day:, :]

    if convergence_error < simulation.convergence_target:
        conditional_print(config.silent_mode,  f"Convergence achieved after {day} days.")
        if config.calculate_energy_terms:
            for i in range(len(shape_model)):
                thermal_data.insolation_energy[i] = energy_terms[i, :, 0]
                thermal_data.re_emitted_energy[i] = energy_terms[i, :, 1]
                thermal_data.surface_energy_change[i] = energy_terms[i, :, 2]
                thermal_data.conducted_energy[i] = energy_terms[i, :, 3]
                thermal_data.unphysical_energy_loss[i] = energy_terms[i, :, 4]
    else:
        conditional_print(config.silent_mode,  f"Maximum days reached without achieving convergence.")
        conditional_print(config.silent_mode,  f"Final temperature error: {mean_temperature_error} K")
        conditional_print(config.silent_mode,  "Try increasing max_days or decreasing convergence_target.")
        if config.silent_mode:
            return

        if config.calculate_energy_terms:
            plt.plot(energy_terms[i, :, 0], label="Insolation energy")
            plt.plot(energy_terms[i, :, 1], label="Re-emitted energy")
            plt.plot(energy_terms[i, :, 2], label="Surface energy change")
            plt.plot(energy_terms[i, :, 3], label="Conducted energy")
            plt.plot(energy_terms[i, :, 4], label="Unphysical energy loss")
            plt.legend()
            plt.show()

    return final_day_temperatures, final_day_temperatures_all_layers, final_timestep_temperatures, day+1, temperature_error, max_temperature_error

@jit(nopython=True)
def calculate_energy_terms(temperature, insolation, delta_t, emissivity, beaming_factor,
                           density, specific_heat_capacity, layer_thickness, thermal_conductivity,
                           timesteps_per_day, n_layers):
    energy_terms = np.zeros((len(temperature), timesteps_per_day, 5))
    for i in range(len(temperature)):
        for time_step in range(timesteps_per_day):
            energy_terms[i, time_step, 0] = insolation[i, time_step] * delta_t
            energy_terms[i, time_step, 1] = -emissivity * beaming_factor * 5.670374419e-8 * (temperature[i, time_step, 0]**4) * delta_t
            energy_terms[i, time_step, 2] = -density * specific_heat_capacity * layer_thickness * (temperature[i, (time_step + 1) % timesteps_per_day, 0] - temperature[i, time_step, 0])
            energy_terms[i, time_step, 3] = thermal_conductivity * delta_t * (temperature[i, time_step, 1] - temperature[i, time_step, 0]) / layer_thickness
            energy_terms[i, time_step, 4] = energy_terms[i, time_step, 0] + energy_terms[i, time_step, 1] + energy_terms[i, time_step, 2] + energy_terms[i, time_step, 3]
    return energy_terms

def calculate_initial_temperatures(thermal_data, silent_mode, emissivity, n_jobs=-1):
    ''' 
    This function calculates the initial temperature of each facet and sub-surface layer of the body based on the insolation curve for that facet. It writes the initial temperatures to the data cube.

    BUG: Crashed while running with n_jobs=1, icosphere with 1280 facets and 2 roughness subdivisions.
    '''
    # Stefan-Boltzmann constant
    sigma = 5.67e-8

    # Define the facet processing function inside the main function
    def process_facet(insolation, emissivity, sigma):
        # Calculate the initial temperature based on average power in
        power_in = np.mean(insolation)
        # Calculate the temperature of the facet using the Stefan-Boltzmann law
        calculated_temp = (power_in / (emissivity * sigma))**(1/4)

        # Return the calculated temperature for all layers
        return calculated_temp

    # Parallel processing of facets
    results = Parallel(n_jobs=n_jobs)(delayed(process_facet)(thermal_data.insolation[i], emissivity, sigma) 
                                      for i in range(thermal_data.temperatures.shape[0])
    )

    conditional_print(silent_mode, f"Initial temperatures calculated for {thermal_data.temperatures.shape[0]} facets.")

    # Update the original shape_model with the results
    for i, temperature in conditional_tqdm(enumerate(results), silent_mode, total=len(results), desc='Saving temps'):
        thermal_data.temperatures[i, :, :] = temperature

    conditional_print(silent_mode, "Initial temperatures saved for all facets.")

    return thermal_data

@jit(nopython=True)
def calculate_secondary_radiation(temperatures, visible_facets, view_factors, self_heating_const):
    return self_heating_const * np.sum(temperatures[visible_facets]**4 * view_factors)
