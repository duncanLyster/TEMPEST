# src/model/simulation.py

import numpy as np
from src.utilities.locations import Locations

class Simulation:
    def __init__(self, config):
        """
        Initialize the simulation using the provided Config object.
        """
        self.config = config
        self.load_configuration()

    def load_configuration(self):
        """
        Load configuration directly from the Config object.
        """
        # Assign configuration to attributes, converting lists to numpy arrays as needed
        for key, value in self.config.config_data.items():
            if isinstance(value, list):
                value = np.array(value)
            setattr(self, key, value)
        
        # Initialization calculations based on the loaded parameters
        self.solar_distance_m = self.solar_distance_au * 1.496e11  # Convert AU to meters
        self.rotation_period_s = self.rotation_period_hours * 3600  # Convert hours to seconds
        self.angular_velocity = (2 * np.pi) / self.rotation_period_s
        self.thermal_conductivity = (self.thermal_inertia**2 / (self.density * self.specific_heat_capacity))
        self.skin_depth = (self.thermal_conductivity / (self.density * self.specific_heat_capacity * self.angular_velocity)) ** 0.5
        self.layer_thickness = 8 * self.skin_depth / self.n_layers
        self.thermal_diffusivity = self.thermal_conductivity / (self.density * self.specific_heat_capacity)
        self.timesteps_per_day = self.calculate_adaptive_timesteps() # Adaptive timestep for low thermal inertia stability
        self.delta_t = self.rotation_period_s / self.timesteps_per_day
        
        # Compute unit vector from RA and Dec
        ra_radians = np.radians(self.ra_degrees)
        dec_radians = np.radians(self.dec_degrees)
        self.rotation_axis = np.array([np.cos(ra_radians) * np.cos(dec_radians), 
                                       np.sin(ra_radians) * np.cos(dec_radians), 
                                       np.sin(dec_radians)])

    def calculate_adaptive_timesteps(self):
        """
        Calculate timesteps.
        
        If 'timesteps_per_day' is specified in the config, it is used directly.
        Otherwise, adaptive timesteps are calculated based on CFL stability limits
        (mainly for the explicit solver).
        """
        # Check if user specified timesteps_per_day in config
        # Note: self.timesteps_per_day is set by load_configuration before this is called
        user_timesteps = getattr(self, 'timesteps_per_day', None)
        
        # Stability calculation (CFL limits)
        # Use a safety factor to ensure const3 is well below 0.5 to prevent ringing
        cfl_safety_factor = 0.8
        cfl_denominator = cfl_safety_factor * (self.layer_thickness**2 / (2 * self.thermal_diffusivity))
        timesteps_cfl = int(round(self.rotation_period_s / cfl_denominator))
        
        if user_timesteps is not None:
            # Warn if user-specified timesteps violate CFL for the explicit solver
            solver_name = getattr(self, 'temp_solver', '')
            if solver_name == 'tempest_standard' and int(user_timesteps) < timesteps_cfl:
                print("\n" + "=" * 80)
                print("  WARNING: CFL STABILITY VIOLATION (explicit solver)")
                print("=" * 80)
                print(f"  You specified timesteps_per_day = {int(user_timesteps)}, but the CFL")
                print(f"  stability criterion requires at least {timesteps_cfl} timesteps.")
                print(f"  The explicit solver will likely produce unphysical results (e.g. all")
                print(f"  temperatures dropping to 2.7 K).")
                print(f"")
                print(f"  Fix: either remove 'timesteps_per_day' from your config to let TEMPEST")
                print(f"  choose automatically, or increase it to >= {timesteps_cfl}.")
                print("=" * 80 + "\n")
            return int(user_timesteps)
        delta_t_cfl = self.rotation_period_s / timesteps_cfl
        
        # Calculate insolation coefficient with CFL timestep
        const1_cfl = delta_t_cfl / (self.layer_thickness * self.density * self.specific_heat_capacity)
        
        # Adaptive constraint: limit const1 for stability
        # Lower limit to 0.01 to ensure radiative stability at high T (~400K)
        max_const1 = 0.01
        
        if const1_cfl > max_const1:
            # Calculate timestep that keeps const1 reasonable
            required_delta_t = max_const1 * self.layer_thickness * self.density * self.specific_heat_capacity
            adaptive_timesteps = int(np.ceil(self.rotation_period_s / required_delta_t))
            return adaptive_timesteps
        else:
            return timesteps_cfl

class ThermalData:
    def __init__(self, n_facets, timesteps_per_day, n_layers, max_days, calculate_energy_terms):
        # Surface temperatures for one day only
        self.temperatures = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        # Two columns for current and previous timestep subsurface temperatures
        self.layer_temperatures = np.zeros((n_facets, 2, n_layers), dtype=np.float64)
        self.insolation = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        self.visible_facets = [np.array([], dtype=np.int64) for _ in range(n_facets)]
        self.secondary_radiation_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]
        self.thermal_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]

        if calculate_energy_terms:
            # Energy terms for one day only
            self.insolation_energy = np.zeros((n_facets, timesteps_per_day))
            self.re_emitted_energy = np.zeros((n_facets, timesteps_per_day))
            self.surface_energy_change = np.zeros((n_facets, timesteps_per_day))
            self.conducted_energy = np.zeros((n_facets, timesteps_per_day))
            self.unphysical_energy_loss = np.zeros((n_facets, timesteps_per_day))

    def set_visible_facets(self, visible_facets):
        self.visible_facets = [np.array(facets, dtype=np.int64) for facets in visible_facets]

    def set_secondary_radiation_view_factors(self, view_factors):
        self.secondary_radiation_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]
        
    def set_thermal_view_factors(self, view_factors):
        """Set thermal view factors for all facets."""
        self.thermal_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]