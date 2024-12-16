import os

class Locations:
    def __init__(self, base_dir=None):
        # Dynamically get the absolute path to the root of the project
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        else:
            self.base_dir = os.path.abspath(base_dir)

        # Core directories
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.shape_models = os.path.join(self.data_dir, 'shape_models')
        self.model_setups = os.path.join(self.data_dir, 'model_setups')
        self.scattering_luts = os.path.join(self.data_dir, 'scattering_luts')
        self.outputs = os.path.join(self.base_dir, 'output')
        self.config = os.path.join(self.base_dir, 'config.yaml')

        # Output directories
        self.remote_outputs = os.path.join(self.outputs, 'remote_outputs')
        self.phase_curve_data = os.path.join(self.outputs, 'visible_phase_curve_data')
        self.precalculated = os.path.join(self.outputs, 'precalculated')

        # Precomputed data directories
        self.view_factors = os.path.join(self.precalculated, 'view_factors')
        self.visible_facets = os.path.join(self.precalculated, 'visible_facets')
        self.other_cached_data = os.path.join(self.outputs, 'other_cached_data')

    def get_shape_model_path(self, filename):
        return os.path.join(self.shape_models, filename)
    
    def get_setup_file_path(self, filename):
        return os.path.join(self.model_setups, filename)
    
    def get_view_factors_path(self, hash_code):
        return os.path.join(self.view_factors, f"{hash_code}.npz")

    def get_visible_facets_path(self, hash_code):
        return os.path.join(self.visible_facets, f"{hash_code}.npz")
    
    def get_output_path(self, filename):
        return os.path.join(self.outputs, filename)
        
    def get_config_path(self):
        return self.config

    def get_scattering_lut_path(self, filename):
        return os.path.join(self.scattering_luts, filename)
    
    def ensure_directories_exist(self):
        """
        Ensure that all necessary directories exist.
        """
        dirs = [
            self.shape_models, self.model_setups, self.scattering_luts, self.outputs,
            self.remote_outputs, self.phase_curve_data, self.precalculated,
            self.view_factors, self.visible_facets, self.other_cached_data
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
