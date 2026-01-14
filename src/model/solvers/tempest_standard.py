import numpy as np
from numba import jit
from .base_solver import TemperatureSolver
from src.utilities.utils import conditional_print

# Standalone numba functions
@jit(nopython=True)
def calculate_secondary_radiation(temperatures, visible_facets, view_factors, self_heating_const):
    if len(visible_facets) == 0 or len(view_factors) == 0:
        return 0.0
    return self_heating_const * np.sum(temperatures[visible_facets]**4 * view_factors)


@jit(nopython=True)
def calculate_temperatures(temperatures, layer_temperatures, insolation, visible_facets_list, 
                        view_factors_list, const1, const2, const3, self_heating_const,
                        timesteps_per_day, n_layers, include_self_heating):
    
    n_facets = temperatures.shape[0]
    
    # Initialize column pointers
    current_column = 0 
    prev_column = 0    
    
    for time_step in range(timesteps_per_day):
        # 1. Column Swapping
        if time_step % 2 == 0:
            prev_column = 0
            current_column = 1
        else:
            prev_column = 1
            current_column = 0

        # Look ahead for RK2 (Heun's Method)
        next_step = time_step + 1
        if next_step >= timesteps_per_day:
            next_step = 0
        
        for i in range(n_facets):
            # --- Surface Calculation (Runge-Kutta 2nd Order) ---
            
            # State at start of step (t)
            T_n = layer_temperatures[i, prev_column, 0]
            T_layer1 = layer_temperatures[i, prev_column, 1] # Assumed constant for the substep
            
            # Insolation at start (t) and end (t+1)
            I_n = insolation[i, time_step]
            I_next = insolation[i, next_step]
            
            # --- STEP 1: Predictor (Slope at Start) ---
            # Calculate slope k1 using T_n and I_n
            # Note: Factors of 2.0 included for Half-Node physics
            
            rad_term_1 = -const2 * (T_n**4) * 2.0
            insol_term_1 = I_n * const1 * 2.0
            cond_term_1 = (const3 * 2.0) * (T_layer1 - T_n)
            
            self_heat_1 = 0.0
            if include_self_heating:
                self_heat_1 = 2.0 * calculate_secondary_radiation(
                    layer_temperatures[:, prev_column, 0], 
                    visible_facets_list[i], 
                    view_factors_list[i], 
                    self_heating_const
                )
                
            # Total Change for Predictor
            dT_1 = rad_term_1 + insol_term_1 + cond_term_1 + self_heat_1
            
            # Predicted Temp (Euler Step)
            T_pred = T_n + dT_1
            
            # --- STEP 2: Corrector (Slope at End) ---
            # Calculate slope k2 using T_pred and I_next
            
            rad_term_2 = -const2 * (T_pred**4) * 2.0
            insol_term_2 = I_next * const1 * 2.0
            cond_term_2 = (const3 * 2.0) * (T_layer1 - T_pred)
            
            # Note: For efficiency, we approximate self-heating as constant during the substep
            # (Recalculating it requires updating ALL facets' T_pred, which is expensive)
            self_heat_2 = self_heat_1 
            
            # Total Change for Corrector
            dT_2 = rad_term_2 + insol_term_2 + cond_term_2 + self_heat_2
            
            # --- Final Update (Average of Slopes) ---
            new_temp = T_n + 0.5 * (dT_1 + dT_2)

            temperatures[i, time_step] = new_temp
            layer_temperatures[i, current_column, 0] = new_temp
            
            # --- Subsurface Calculation (Standard Explicit is fine here) ---
            # Subsurface changes are slow, so RK2 is usually overkill/too expensive
            for layer in range(1, n_layers - 1):
                prev_layer = layer_temperatures[i, prev_column, layer]
                prev_layer_plus = layer_temperatures[i, prev_column, layer + 1]
                prev_layer_minus = layer_temperatures[i, prev_column, layer - 1]

                layer_temperatures[i, current_column, layer] = (
                    prev_layer + 
                    const3 * (prev_layer_plus - 2 * prev_layer + prev_layer_minus)
                )

            # Bottom Boundary (Adiabatic)
            layer_temperatures[i, current_column, n_layers - 1] = layer_temperatures[i, current_column, n_layers - 2]

    # --- End of Day Updates ---
    if current_column == 1:
        for i in range(n_facets):
            for layer in range(n_layers):
                layer_temperatures[i, 0, layer] = layer_temperatures[i, 1, layer]

    # Radiative Mean Deep Boundary
    for i in range(n_facets):
        sum_rad = 0.0
        for t in range(timesteps_per_day):
            sum_rad += temperatures[i, t]**4
        mean_rad_temp = (sum_rad / timesteps_per_day)**0.25
        layer_temperatures[i, 0, n_layers - 1] = mean_rad_temp

    return temperatures


@jit(nopython=True)
def calculate_temperatures_stable(temperatures, layer_temperatures, insolation, visible_facets_list, 
                        view_factors_list, const1, const2, const3, self_heating_const,
                        timesteps_per_day, n_layers, include_self_heating):
    
    n_facets = temperatures.shape[0]
    
    # Initialize column pointers
    current_column = 0 
    prev_column = 0    
    
    for time_step in range(timesteps_per_day):
        # Column swapping logic
        if time_step % 2 == 0:
            prev_column = 0
            current_column = 1
        else:
            prev_column = 1
            current_column = 0
        
        for i in range(n_facets):
            # --- Surface Calculation (Skin/Half-Node Model) ---
            prev_temp = layer_temperatures[i, prev_column, 0]
            prev_temp_layer1 = layer_temperatures[i, prev_column, 1]

            # 1. Flux terms multiplied by 2.0 because surface node has half-mass (dx/2)
            insolation_term = insolation[i, time_step] * const1 * 2.0
            re_emitted_radiation_term = -const2 * (prev_temp**4) * 2.0
            
            secondary_radiation_term = 0.0
            if include_self_heating:
                secondary_radiation_term = 2.0 * calculate_secondary_radiation(
                    layer_temperatures[:, prev_column, 0], 
                    visible_facets_list[i], 
                    view_factors_list[i], 
                    self_heating_const
                )
            
            # 2. Conduction multiplied by 2.0 (flux acts on half-mass)
            conducted_heat_term = (const3 * 2.0) * (prev_temp_layer1 - prev_temp)
            
            new_temp = (prev_temp + 
                    insolation_term + 
                    re_emitted_radiation_term + 
                    conducted_heat_term + 
                    secondary_radiation_term)

            temperatures[i, time_step] = new_temp
            layer_temperatures[i, current_column, 0] = new_temp
            
            # --- Subsurface Calculation ---
            for layer in range(1, n_layers - 1):
                prev_layer = layer_temperatures[i, prev_column, layer]
                prev_layer_plus = layer_temperatures[i, prev_column, layer + 1]
                prev_layer_minus = layer_temperatures[i, prev_column, layer - 1]

                layer_temperatures[i, current_column, layer] = (
                    prev_layer + 
                    const3 * (prev_layer_plus - 2 * prev_layer + prev_layer_minus)
                )

            # 3. Bottom Boundary Condition
            layer_temperatures[i, current_column, n_layers - 1] = layer_temperatures[i, current_column, n_layers - 2]

    # --- End of Day Updates ---
    
    # 1. Standardize Output Column (Ensure valid data is in col 0)
    final_col = current_column
    if final_col == 1:
        for i in range(n_facets):
            for layer in range(n_layers):
                layer_temperatures[i, 0, layer] = layer_temperatures[i, 1, layer]

    # 2. FAST CONVERGENCE: Set Deep Boundary to Radiative Mean
    # Calculate the radiative mean of the day just finished.
    # We set the bottom boundary
    # for the START of the next day.
    for i in range(n_facets):
        # Calculate Radiative Mean: (Mean(T^4))^0.25
        # This is physically where the deep temperature tends to converge.
        sum_rad = 0.0
        for t in range(timesteps_per_day):
            sum_rad += temperatures[i, t]**4
        
        mean_rad_temp = (sum_rad / timesteps_per_day)**0.25
        
        # Update the deepest layer in 'column 0' (ready for next function call)
        # Note: Only update the very bottom. The layers between will adjust via conduction in the next day.
        layer_temperatures[i, 0, n_layers - 1] = mean_rad_temp

    return temperatures

class TempestStandardSolver(TemperatureSolver):
    def __init__(self):
        super().__init__("tempest_standard")
        self.required_parameters = [
            "emissivity",
            "density",
            "specific_heat_capacity",
            "thermal_conductivity",
            "n_layers",
            "convergence_target",
            "beaming_factor"
        ]

    def solve(self, thermal_data, shape_model, simulation, config):
        ''' 
        This is the main calculation function for the thermophysical body model. It calls the necessary functions to read in the shape model, set material and model properties, calculate 
        insolation and temperature arrays, and iterate until the model converges.
        '''

        # Initialize constants
        const1 = simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
        const2 = simulation.emissivity * simulation.beaming_factor * 5.67e-8 * simulation.delta_t / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)
        const3 = simulation.thermal_diffusivity * simulation.delta_t / simulation.layer_thickness**2
        self_heating_const = 5.670374419e-8 * simulation.delta_t * simulation.emissivity**2 / (simulation.layer_thickness * simulation.density * simulation.specific_heat_capacity)

        convergence_error = simulation.convergence_target + 1
        day = 0
        error_history = []
        comparison_temps = thermal_data.temperatures[:, 0].copy()

        while day < simulation.max_days and (day < simulation.min_days or convergence_error > simulation.convergence_target):
            current_day_temperature = calculate_temperatures(
                thermal_data.temperatures,
                thermal_data.layer_temperatures,
                thermal_data.insolation,
                thermal_data.visible_facets,
                thermal_data.thermal_view_factors,
                const1, const2, const3, self_heating_const,
                simulation.timesteps_per_day, simulation.n_layers,
                config.include_self_heating
            )

            # Check for invalid temperatures
            for i in range(len(shape_model)):
                for time_step in range(simulation.timesteps_per_day):
                    if np.isnan(current_day_temperature[i, time_step]) or np.isinf(current_day_temperature[i, time_step]) or current_day_temperature[i, time_step] < 0:
                        conditional_print(config.silent_mode, f"Invalid temperature {current_day_temperature[i, time_step]} K detected for facet {i} at timestep {time_step}")
                        return {
                            "final_day_temperatures": None,
                            "final_day_temperatures_all_layers": None,
                            "final_timestep_temperatures": None,
                            "days_to_convergence": day,
                            "mean_temperature_error": None,
                            "max_temperature_error": None
                        }

            if config.calculate_energy_terms:
                energy_terms = self.calculate_energy_terms(
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

        return {
            "final_day_temperatures": current_day_temperature,
            "final_day_temperatures_all_layers": thermal_data.layer_temperatures,
            "final_timestep_temperatures": current_day_temperature[:, -1],
            "days_to_convergence": day,
            "mean_temperature_error": mean_temperature_error,
            "max_temperature_error": max_temperature_error
        } 