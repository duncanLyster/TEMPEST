# Base class for temperature solvers

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from src.utilities.utils import conditional_print
from src.utilities.utils import conditional_tqdm

class TemperatureSolver:
    def __init__(self, name):
        self.name = name
        self.required_parameters = []

    def initialize_temperatures(self, thermal_data, simulation, config):
        """Initialize temperature arrays based on average insolation"""
        # Stefan-Boltzmann constant
        sigma = 5.67e-8

        def process_facet(insolation, emissivity, sigma):
            # Calculate the initial temperature based on average power in
            power_in = np.mean(insolation)
            # Calculate the temperature of the facet using the Stefan-Boltzmann law
            return (power_in / (emissivity * sigma))**(1/4)

        # Parallel processing of facets
        results = Parallel(n_jobs=config.n_jobs)(
            delayed(process_facet)(thermal_data.insolation[i], simulation.emissivity, sigma) 
            for i in range(thermal_data.temperatures.shape[0])
        )

        conditional_print(config.silent_mode, f"Initial temperatures calculated for {thermal_data.temperatures.shape[0]} facets.")

        # Update both surface and layer temperatures with initial values
        for i, temperature in conditional_tqdm(enumerate(results), config.silent_mode, total=len(results), desc='Saving temps'):
            thermal_data.temperatures[i, :] = temperature
            thermal_data.layer_temperatures[i, 0, :] = temperature
            thermal_data.layer_temperatures[i, 1, :] = temperature

        conditional_print(config.silent_mode, "Initial temperatures saved for all facets.")
        return thermal_data

    def solve(self, thermal_data, shape_model, simulation, config):
        """Run the temperature solving algorithm"""
        raise NotImplementedError("Subclasses must implement solve")

    def get_required_parameters(self):
        return self.required_parameters 