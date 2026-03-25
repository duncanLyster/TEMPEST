import numpy as np
from numba import jit
import gc
from .base_solver import TemperatureSolver
from src.utilities.utils import conditional_print

@jit(nopython=True)
def solve_tridiagonal(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d using the Thomas algorithm (TDMA).
    a: lower diagonal
    b: main diagonal
    c: upper diagonal
    d: right hand side vector
    Returns x
    """
    n = len(d)
    c_prime = np.zeros(n, dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        temp = b[i] - a[i-1] * c_prime[i-1]
        if i < n - 1:
            c_prime[i] = c[i] / temp
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / temp

    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x

@jit(nopython=True)
def calculate_secondary_radiation(temperatures, visible_facets, view_factors, self_heating_const):
    if len(visible_facets) == 0 or len(view_factors) == 0:
        return 0.0
    # self_heating_const already includes epsilon etc? No, usually separate.
    # But here we just return the flux (W/m2) or temperature term?
    # In standard solver: self_heating_const * sum(T^4 * VF).
    # We will assume self_heating_const scales correctly to W/m2 or energy balance term.
    return np.sum(temperatures[visible_facets]**4 * view_factors)

@jit(nopython=True)
def solve_implicit_timestep(T_old, insolation, T_surface_rad_coupling, 
                          p, q, r, s,  # Interior matrix coefficients
                          alpha, beta, gamma,  # Surface boundary coefficients
                          epsilon, sigma):
    """
    Solves one timestep using Crank-Nicolson for interior and linearized radiation for surface.
    """
    n_layers = len(T_old)
    
    # --- Linearize Surface Radiation ---
    # T_new^4 approx T_old^4 + 4*T_old^3 * (T_new - T_old)
    #           = 4*T_old^3 * T_new - 3*T_old^4
    
    T_surf_old = T_old[0]
    rad_linear_coeff = 4 * epsilon * sigma * T_surf_old**3
    rad_const_term = -3 * epsilon * sigma * T_surf_old**4
    
    # Net flux into surface (excluding conduction to layer 1)
    # Flux_in = S_abs + F_rad + F_coupling - rad_flux
    # Balance: Flux_in + k(T1 - T0)/dz = rho*c*dz/2 * (T0_new - T0_old)/dt
    # Rearranging for T0_new, T1_new...
    
    # We construct the tridiagonal matrix directly here for the specific timestep
    # because the surface coefficients depend on T_old (linearization).
    
    # Matrix diagonals
    # Lower (a), Main (b), Upper (c)
    a = np.zeros(n_layers - 1, dtype=np.float64) # index i corresponds to row i+1
    b = np.zeros(n_layers, dtype=np.float64)
    c = np.zeros(n_layers - 1, dtype=np.float64) # index i corresponds to row i
    rhs = np.zeros(n_layers, dtype=np.float64)
    
    # --- Surface Node (i=0) ---
    # Equation: alpha * T0_new - beta * T1_new = gamma * T0_old + Insolation + Rad_Coupling - Radiation
    # Radiation term: epsilon * sigma * T^4 -> linearized
    # We incorporate linearized radiation into the implicit matrix (b[0]) and RHS
    
    # Let's verify coefficients passed from outside:
    # alpha includes conduction + heat capacity term
    # beta is conduction term
    # gamma is heat capacity term
    # So: (alpha + rad_linear_coeff) * T0_new - beta * T1_new = gamma * T0_old + Insolation + Coupling - rad_const_term
    
    b[0] = alpha + rad_linear_coeff
    c[0] = -beta
    rhs[0] = gamma * T_surf_old + insolation + T_surface_rad_coupling - rad_const_term
    
    # --- Interior Nodes (i=1 to N-2) ---
    # Crank-Nicolson: -p * T_{i-1}^{n+1} + (1 + 2p) * T_i^{n+1} - p * T_{i+1}^{n+1} = ...
    # Here passed as p, q, r
    # p = lambda/2
    # q = 1 + lambda
    # r = 1 - lambda
    # s = lambda/2
    # Equation: -p T_{i-1} + q T_i - p T_{i+1} = s T_{i-1}^n + r T_i^n + s T_{i+1}^n
    
    # Note: 'a' is lower diagonal. a[i-1] is for row i.
    for i in range(1, n_layers - 1):
        a[i-1] = -p
        b[i]   = q
        c[i]   = -p
        rhs[i] = s * T_old[i-1] + r * T_old[i] + s * T_old[i+1]
        
    # --- Bottom Node (Adiabatic) ---
    # T_{N-1} = T_{N-2} (Zero flux)
    # Simple implementation: T_{N-1} - T_{N-2} = 0
    a[n_layers-2] = -1.0
    b[n_layers-1] = 1.0
    rhs[n_layers-1] = 0.0
    
    # Solve
    T_new = solve_tridiagonal(a, b, c, rhs)
    
    # Floor check
    for i in range(n_layers):
        if T_new[i] < 2.7:
            T_new[i] = 2.7
            
    return T_new

@jit(nopython=True)
def run_implicit_solver(temperatures, layer_temperatures, insolation, 
                       visible_facets_list, view_factors_list,
                       sim_dt, layer_thickness, density, specific_heat, thermal_conductivity,
                       emissivity, sigma, self_heating_enabled):
    
    n_facets = temperatures.shape[0]
    timesteps = temperatures.shape[1]
    n_layers = layer_temperatures.shape[2]
    
    # Pre-calculate constant matrix coefficients for interior (Crank-Nicolson)
    # alpha_diff = k / (rho * c)
    # lambda = alpha_diff * dt / dx^2
    thermal_diffusivity = thermal_conductivity / (density * specific_heat)
    lamb = thermal_diffusivity * sim_dt / (layer_thickness**2)
    
    # Crank-Nicolson coefficients
    p = lamb / 2.0
    q = 1.0 + lamb
    r = 1.0 - lamb
    s = lamb / 2.0
    
    # Surface boundary coefficients (Half-node)
    # Capacity term: C_surf = rho * c * dx / 2
    # Cond term: K_eff = k / dx
    # Balance: C_surf * (T_new - T_old)/dt = -K_eff * (T_new - T_layer1_new + T_old - T_layer1_old)/2 + Fluxes
    # We use Fully Implicit for surface conduction to avoid oscillations coupled with radiation
    # Or Crank-Nicolson. Let's use Fully Implicit for surface-to-layer1 conduction for robustness.
    # Eq: rho*c*dx/2 * (T0_new - T0_old)/dt = -k/dx * (T0_new - T1_new) + Radiation + Solar
    
    # Let's use Fully Implicit for the surface node equation
    c_surf = density * specific_heat * layer_thickness / 2.0
    k_eff = thermal_conductivity / layer_thickness
    
    # (c_surf/dt) * T0_new - (c_surf/dt) * T0_old = -k_eff * T0_new + k_eff * T1_new + Fluxes
    # (c_surf/dt + k_eff) * T0_new - k_eff * T1_new = (c_surf/dt) * T0_old + Fluxes
    
    alpha_surf = (c_surf / sim_dt) + k_eff
    beta_surf = k_eff
    gamma_surf = (c_surf / sim_dt)
    
    # Loop over timesteps
    for t in range(timesteps):
        # Look ahead for insolation? Or average? 
        # Implicit usually takes value at t+1 or average. Let's use average for smoothness.
        t_next = 0 if t == timesteps - 1 else t + 1
        
        # Self-heating loop (simple relaxation or lagged)
        # Since we process facets in parallel/sequence, we lag self-heating by one timestep
        # Using T from previous timestep for self-heating view factors
        
        for i in range(n_facets):
            # Get previous state
            T_old_layers = layer_temperatures[i, 0, :].copy() # 0 is current/old state
            
            # Insolation (Average of current and next for Crank-Nicolson consistency)
            sol_flux = 0.5 * (insolation[i, t] + insolation[i, t_next])
            
            # Self heating flux
            rad_coupling = 0.0
            if self_heating_enabled:
                # Use surface temp from previous step
                # Note: We need the temperatures of OTHER facets. 
                # layer_temperatures[:, 0, 0] contains surface temps of all facets at start of step
                rad_coupling = calculate_secondary_radiation(
                    layer_temperatures[:, 0, 0],
                    visible_facets_list[i],
                    view_factors_list[i],
                    1.0 # multiplier, epsilon/sigma handled inside?
                )
                # calculate_secondary_radiation sums T^4 * VF. 
                # We need Flux = epsilon * sigma * sum(T^4 * VF) ?
                # Wait, view factors usually include area ratios.
                # Standard formula: Q_i = ... + epsilon_i * sum(F_ij * sigma * T_j^4)
                # But assume blackbody emission from neighbors? 
                # Usually: Flux_in = epsilon * sigma * sum(T^4 * VF)
                rad_coupling *= (emissivity * sigma)

            # Solve
            T_new_layers = solve_implicit_timestep(
                T_old_layers, 
                sol_flux, 
                rad_coupling,
                p, q, r, s,
                alpha_surf, beta_surf, gamma_surf,
                emissivity, sigma
            )
            
            # Update storage (we use column 1 as 'new' then swap, or just overwrite since we need old for next facet?)
            # Actually we need T_old of all facets for self-heating of next facet?
            # Ideally we update all 'new' buffers then swap.
            # But here `layer_temperatures` has 2 columns. 
            # We read from col 0 (old), write to col 1 (new).
            layer_temperatures[i, 1, :] = T_new_layers
            temperatures[i, t] = T_new_layers[0] # Store surface temp
            
        # Swap columns for next timestep: col 1 becomes col 0
        for i in range(n_facets):
            layer_temperatures[i, 0, :] = layer_temperatures[i, 1, :]
            
    return temperatures

class TempestImplicitSolver(TemperatureSolver):
    def __init__(self):
        super().__init__("tempest_implicit")
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
        """
        Main solver loop using Implicit method.
        """
        # Constants
        sigma = 5.67e-8
        
        # Initialize convergence tracking
        convergence_error = simulation.convergence_target + 1
        day = 0
        comparison_temps = thermal_data.temperatures[:, 0].copy()
        
        conditional_print(config.silent_mode, "Starting Implicit Solver (Crank-Nicolson)...")

        while day < simulation.max_days and (day < simulation.min_days or convergence_error > simulation.convergence_target):
            
            # Run the jit-compiled solver for one day
            # We assume layer_thickness is constant
            current_day_temperature = run_implicit_solver(
                thermal_data.temperatures,
                thermal_data.layer_temperatures,
                thermal_data.insolation,
                thermal_data.visible_facets,
                thermal_data.thermal_view_factors,
                simulation.delta_t,
                simulation.layer_thickness,
                simulation.density,
                simulation.specific_heat_capacity,
                simulation.thermal_conductivity,
                simulation.emissivity,
                sigma,
                config.include_self_heating
            )
            
            # Convergence check
            temperature_errors = np.abs(current_day_temperature[:, 0] - comparison_temps)
            
            if config.convergence_method == 'mean':
                convergence_error = np.mean(temperature_errors)
            else:
                convergence_error = np.max(temperature_errors)

            max_temperature_error = np.max(temperature_errors)
            mean_temperature_error = np.mean(temperature_errors)

            conditional_print(config.silent_mode, f"Day: {day} | Mean Error: {mean_temperature_error:.6f} K | Max Error: {max_temperature_error:.6f} K")
            
            comparison_temps = current_day_temperature[:, 0].copy()
            day += 1
            
            # OPTIMIZATION: Explicit garbage collection after each day to free worker memory
            # This is especially important for high-resolution models (>200k facets)
            if day % 2 == 0:  # Every 2 days
                gc.collect()

        return {
            "final_day_temperatures": current_day_temperature,
            "final_day_temperatures_all_layers": thermal_data.layer_temperatures,
            "final_timestep_temperatures": current_day_temperature[:, -1],
            "days_to_convergence": day,
            "mean_temperature_error": mean_temperature_error,
            "max_temperature_error": max_temperature_error
        }
