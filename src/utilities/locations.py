import os

class Locations:
    def __init__(self, base_dir='TEMPEST'):
        self.base_dir = base_dir

        # Core directories
        self.shape_models = os.path.join(self.base_dir, 'src', 'data', 'shape_models')
        self.model_setups = os.path.join(self.base_dir, 'src', 'data', 'model_setups')
        self.assets = os.path.join(self.base_dir, 'src', 'data', 'assets')
        self.outputs = os.path.join(self.base_dir, 'output')

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
    
    def ensure_directories_exist(self):
        """
        Ensure that all necessary directories exist.
        """
        dirs = [
            self.shape_models, self.model_setups, self.assets, self.outputs,
            self.remote_outputs, self.phase_curve_data, self.precalculated,
            self.view_factors, self.visible_facets, self.other_cached_data
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)