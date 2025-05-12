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
        
        # Surface Roughness Model settings
        self.apply_spherical_depression_roughness = self.config_data.get('apply_spherical_depression_roughness', False)
        self.depression_subfacets_count = self.config_data.get('depression_subfacets_count', 30)
        self.depression_profile_angle_degrees = self.config_data.get('depression_profile_angle_degrees', 45)
        self.depression_MCRT_rays_per_emission_step = self.config_data.get('depression_MCRT_rays_per_emission_step', 100)
        self.depression_internal_scattering_iterations = self.config_data.get('depression_internal_scattering_iterations', 2)
        self.depression_outgoing_emission_bins = self.config_data.get('depression_outgoing_emission_bins', 36)

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