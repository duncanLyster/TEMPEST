'''
This file contains the configuration for the model. The user can set the parameters for the model here. TODO: Update user input method to read everything in from a yaml file.
'''

from src.utilities.locations import get_data_dir
import os
from multiprocessing import cpu_count
import warnings

class Config:
    def __init__(self):
        ################ START OF USER INPUTS ################
        
        # Get setup file and shape model
        # self.path_to_shape_model_file = "shape_models/1D_square.stl"
        # self.path_to_setup_file = "model_setups/John_Spencer_default_model_parameters.json"

        self.path_to_shape_model_file = "shape_models/5km_ico_sphere_80_facets.stl"
        self.path_to_setup_file = "model_setups/John_Spencer_default_model_parameters.json"

        # self.path_to_shape_model_file = "private/Lucy/Dinkinesh/Dinkinesh.stl"
        # self.path_to_setup_file = "private/Lucy/Dinkinesh/Dinkinesh_parameters.json"

        # self.path_to_shape_model_file = "shape_models/Rubber_Duck_high_res.stl"
        # self.path_to_setup_file = "private/Lucy/Dinkinesh/Dinkinesh_parameters.json"
        
        # self.path_to_shape_model_file = "shape_models/67P_not_to_scale_1666_facets.stl"
        # self.path_to_setup_file = "private/Lucy/Dinkinesh/Dinkinesh_parameters.json"

        ################ GENERAL ################
        self.silent_mode = False
        self.remote = False

        ################ PERFORMANCE ################
        # Number of parallel jobs to run
        # The calculations that are parallelised are visible facet calculation, elimination of obstructed facets (optional), view factors, scattering, and temperature solver.
        # Use -1 to use all available cores (USE WITH CAUTION on shared computing facilities!)
        # Default to n_jobs or max available cores, whichever is smaller
        n_jobs = 4
        self.n_jobs = min(n_jobs, cpu_count())  
        self.chunk_size = 100  # Number of facets to process per parallel task TODO: It may be sensible to change this depending on the function being parallelised.

        ################ MODELLING ################
        self.convergence_method = 'mean' # 'mean' or 'max'. Mean is recommended for most cases, max is best for investigating permanently shadowed regions.
        self.include_shadowing = True # Recommended to keep this as True for most cases
        self.n_scatters = 2 # Set to 0 to disable scattering. 1 or 2 is recommended for most cases. 3 is almost always unncecessary.
        self.include_self_heating = False
        self.apply_roughness = True
        self.subdivision_levels = 3
        self.displacement_factors = [0.5, 0.5, 0.5]
        self.vf_rays = 100 # Number of rays to use for view factor calculations. 1000 is recommended for most cases.
        self.calculate_visible_phase_curve = True
        self.calculate_thermal_phase_curve = True

        ################ PLOTTING ################
        self.plotted_facet_index = 1220 # Index of facet to plot
        self.plot_insolation_curve = False
        self.plot_insolation_curve = False 
        self.plot_initial_temp_histogram = False
        self.plot_final_day_all_layers_temp_distribution = False
        self.plot_final_day_all_layers_temp_distribution = False
        self.plot_energy_terms = False # NOTE: You must set config.calculate_energy_terms to True to plot energy terms
        self.plot_final_day_comparison = False
        self.show_visible_phase_curve = True # TODO: Should be automatically disabled in remote mode
        self.save_visible_phase_curve_data = True
        self.show_thermal_phase_curve = False # TODO: Should be automatically disabled in remote mode
        self.save_thermal_phase_curve_data = False

        ################ ANIMATIONS ################        BUG: If using 2 animations, the second one doesn't work (pyvista segmenation fault)
        self.animate_roughness_model = False
        self.animate_shadowing = False
        self.animate_secondary_radiation_view_factors = False
        self.animate_secondary_contributions = False
        self.animate_final_day_temp_distribution = False

        ################ DEBUGGING ################
        self.calculate_energy_terms = False         # NOTE: Numba must be disabled if calculating energy terms - model will run much slower

        ################ END OF USER INPUTS ################

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
