'''
This file contains the Config class which is used to load and store the configuration settings from the config.yaml file.
'''

import yaml
import warnings
from multiprocessing import cpu_count
from src.utilities.locations import Locations
import os

class Config:
    def __init__(self, config_path=None):
        # Initialize Locations
        self.locations = Locations()
        
        # Use provided config path or search in standard locations
        if config_path is None:
            # Look in private first, then public
            private_config = os.path.join(self.locations.project_root, "private/data/config/config.yaml")
            public_config = os.path.join(self.locations.project_root, "data/config/config.yaml")
            
            if os.path.exists(private_config):
                config_path = private_config
            elif os.path.exists(public_config):
                config_path = public_config
            else:
                raise FileNotFoundError("No configuration file found in standard locations")
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config_data = yaml.safe_load(file)
        
        # Set custom shape model directory if specified
        if 'shape_model_directory' in self.config_data:
            self.locations.shape_model_directory = self.config_data['shape_model_directory']
        
        # Load settings from the YAML file
        self.load_settings()

    def load_settings(self):
        # General settings
        self.silent_mode = self.config_data.get('silent_mode', False)
        self.remote = self.config_data.get('remote', False)
        self.temp_solver = self.config_data.get('temp_solver', 'tempest_standard')

        # File paths
        self.path_to_shape_model_file = self.locations.get_shape_model_path(self.config_data['shape_model_file'])

        # Performance settings
        self.n_jobs = min(self.config_data.get('n_jobs', 4), cpu_count())
        self.chunk_size = self.config_data.get('chunk_size', 100)

        # Modelling parameters
        self.convergence_method = self.config_data.get('convergence_method', 'mean')
        self.include_shadowing = self.config_data.get('include_shadowing', True)
        self.n_scatters = self.config_data.get('n_scatters', 2)
        self.include_self_heating = self.config_data.get('include_self_heating', False)
        self.apply_roughness = self.config_data.get('apply_roughness', False)
        self.subdivision_levels = self.config_data.get('subdivision_levels', 3)
        self.displacement_factors = self.config_data.get('displacement_factors', [0.5, 0.5, 0.5])
        self.vf_rays = self.config_data.get('vf_rays', 100)
        self.calculate_visible_phase_curve = self.config_data.get('calculate_visible_phase_curve', True)
        self.calculate_thermal_phase_curve = self.config_data.get('calculate_thermal_phase_curve', True)
        self.scattering_lut = self.config_data.get('scattering_lut', 'lambertian.npy')
        self.emission_lut = self.config_data.get('emission_lut', 'lambertian.npy')

        # Plotting settings
        self.plotted_facet_index = self.config_data.get('plotted_facet_index', 1220)
        self.plot_insolation_curve = self.config_data.get('plot_insolation_curve', False)
        self.plot_initial_temp_histogram = self.config_data.get('plot_initial_temp_histogram', False)
        self.plot_final_day_all_layers_temp_distribution = self.config_data.get('plot_final_day_all_layers_temp_distribution', False)
        self.plot_energy_terms = self.config_data.get('plot_energy_terms', False)
        self.plot_final_day_comparison = self.config_data.get('plot_final_day_comparison', False)
        self.show_visible_phase_curve = self.config_data.get('show_visible_phase_curve', True)
        self.save_visible_phase_curve_data = self.config_data.get('save_visible_phase_curve_data', True)
        self.show_thermal_phase_curve = self.config_data.get('show_thermal_phase_curve', False)
        self.save_thermal_phase_curve_data = self.config_data.get('save_thermal_phase_curve_data', False)

        # Animations
        self.animate_roughness_model = self.config_data.get('animate_roughness_model', False)
        self.animate_shadowing = self.config_data.get('animate_shadowing', False)
        self.animate_secondary_radiation_view_factors = self.config_data.get('animate_secondary_radiation_view_factors', False)
        self.animate_secondary_contributions = self.config_data.get('animate_secondary_contributions', False)
        self.animate_final_day_temp_distribution = self.config_data.get('animate_final_day_temp_distribution', True)

        # Debugging
        self.calculate_energy_terms = self.config_data.get('calculate_energy_terms', False)

    def validate_jobs(self):
        """
        Validate the number of jobs requested and issue warnings if necessary.
        Returns the actual number of jobs to use.
        """
        available_cores = cpu_count()
        
        if self.n_jobs == -1:
            warnings.warn(
                "Using all available cores (-1). This should be used with caution on shared "
                "computing facilities as it may impact other users. Consider setting a specific "
                f"number of cores instead. Will use {available_cores} cores.",
                UserWarning
            )
            return available_cores
            
        if self.n_jobs > available_cores:
            warnings.warn(
                f"Requested {self.n_jobs} cores but only {available_cores} are available. "
                f"Reducing to {available_cores} cores.",
                UserWarning
            )
            return available_cores
            
        if self.n_jobs < 1:
            warnings.warn(
                f"Invalid number of jobs ({self.n_jobs}). Must be >= 1. Setting to 1.",
                UserWarning
            )
            return 1
            
        return self.n_jobs