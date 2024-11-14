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
        self.skin_depth = (self.thermal_conductivity / (self.density * self.specific_heat_capacity * self.angular_velocity))**0.5
        self.thermal_inertia = (self.density * self.specific_heat_capacity * self.thermal_conductivity)**0.5
        self.layer_thickness = 8 * self.skin_depth / self.n_layers
        self.thermal_diffusivity = self.thermal_conductivity / (self.density * self.specific_heat_capacity)
        self.timesteps_per_day = int(round(self.rotation_period_s / (self.layer_thickness**2 / (2 * self.thermal_diffusivity))))
        self.delta_t = self.rotation_period_s / self.timesteps_per_day
        
        # Compute unit vector from RA and Dec
        ra_radians = np.radians(self.ra_degrees)
        dec_radians = np.radians(self.dec_degrees)
        self.rotation_axis = np.array([np.cos(ra_radians) * np.cos(dec_radians), 
                                       np.sin(ra_radians) * np.cos(dec_radians), 
                                       np.sin(dec_radians)])

class ThermalData:
    def __init__(self, n_facets, timesteps_per_day, n_layers, max_days, calculate_energy_terms):
        self.temperatures = np.zeros((n_facets, timesteps_per_day * max_days, n_layers), dtype=np.float64)
        self.insolation = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        self.visible_facets = [np.array([], dtype=np.int64) for _ in range(n_facets)]
        self.secondary_radiation_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]

        self.calculate_energy_terms = calculate_energy_terms

        if calculate_energy_terms:
            self.insolation_energy = np.zeros((n_facets, timesteps_per_day * max_days))
            self.re_emitted_energy = np.zeros((n_facets, timesteps_per_day * max_days))
            self.surface_energy_change = np.zeros((n_facets, timesteps_per_day * max_days))
            self.conducted_energy = np.zeros((n_facets, timesteps_per_day * max_days))
            self.unphysical_energy_loss = np.zeros((n_facets, timesteps_per_day * max_days))
            
    def set_visible_facets(self, visible_facets):
        self.visible_facets = [np.array(facets, dtype=np.int64) for facets in visible_facets]

    def set_secondary_radiation_view_factors(self, view_factors):
        self.secondary_radiation_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]
