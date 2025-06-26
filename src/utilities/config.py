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
        self.emissivity = self.config_data.get('emissivity')
        self.albedo = self.config_data.get('albedo', 0.5)
        self.convergence_method = self.config_data.get('convergence_method', 'mean')
        self.convergence_target = self.config_data.get('convergence_target', 0.1)
        self.max_days = self.config_data.get('max_days', 20)
        self.min_days = self.config_data.get('min_days', 1)
        self.n_layers = self.config_data.get('n_layers', 45)
        self.include_shadowing = self.config_data.get('include_shadowing', True)
        self.n_scatters = self.config_data.get('n_scatters', 2)
        self.include_self_heating = self.config_data.get('include_self_heating', False)
        self.vf_rays = self.config_data.get('vf_rays', 1000)
        self.scattering_lut = self.config_data.get('scattering_lut', 'lambertian.npy')
        self.emission_lut = self.config_data.get('emission_lut', 'lambertian_epf.npy')
        
        # Orbital and rotational parameters
        self.solar_distance_au = self.config_data.get('solar_distance_au', 1.0)
        self.solar_luminosity = self.config_data.get('solar_luminosity', 3.86420167e+26)
        self.sunlight_direction = self.config_data.get('sunlight_direction', [1, 0, 0])
        self.rotation_period_hours = self.config_data.get('rotation_period_hours', 24)
        self.ra_degrees = self.config_data.get('ra_degrees', 0)
        self.dec_degrees = self.config_data.get('dec_degrees', 90)
        
        # Surface Roughness Model settings (kernel-based)
        self.apply_kernel_based_roughness = self.config_data.get('apply_kernel_based_roughness', False)
        self.roughness_kernel = self.config_data.get('roughness_kernel', 'spherical_cap')
        self.kernel_subfacets_count = self.config_data.get('kernel_subfacets_count', 30)
        self.kernel_profile_angle_degrees = self.config_data.get('kernel_profile_angle_degrees', 45)
        self.kernel_directional_bins = self.config_data.get('kernel_directional_bins', 36)
        # Dome geometry scale factor relative to parent facet radius
        self.kernel_dome_radius_factor = self.config_data.get('kernel_dome_radius_factor', 100.0)
        # Number of intra-facet scattering iterations
        self.intra_facet_scatters = self.config_data.get('intra_facet_scatters', 2)

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
        # Sub-facet temperatures animation
        self.animate_subfacets = self.config_data.get('animate_subfacets', False)
        self.subfacet_facet_index = self.config_data.get('subfacet_facet_index', 0)
        # Fixed camera angles for precomputed viewed-temperature animation (radians)
        self.fixed_camera_theta = self.config_data.get('fixed_camera_theta', None)
        self.fixed_camera_phi = self.config_data.get('fixed_camera_phi', None)
        # Toggle using the precomputed viewed-temperature animation
        self.use_precomputed_viewed_temps = self.config_data.get('use_precomputed_viewed_temps', False)

        # Debugging
        self.calculate_energy_terms = self.config_data.get('calculate_energy_terms', False)
        # Phase curve toggles (legacy keys)
        self.calculate_visible_phase_curve = self.config_data.get('calculate_visible_phase_curve', False)
        self.calculate_thermal_phase_curve = self.config_data.get('calculate_thermal_phase_curve', False)

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