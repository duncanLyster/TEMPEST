import os

class Locations:
    def __init__(self):
        self.project_root = self._find_project_root()
        
        # Base directories
        self.data = os.path.join(self.project_root, "data")
        self.private = os.path.join(self.project_root, "private")
        
        # Public data subdirectories
        self.shape_models = os.path.join(self.data, "shape_models")
        self.config = os.path.join(self.data, "config")
        self.model_setups = os.path.join(self.data, "model_setups")
        self.outputs = os.path.join(self.data, "output")
        self.scattering_luts = os.path.join(self.data, "scattering_luts")
        
        # Output directories
        self.remote_outputs = os.path.join(self.outputs, 'remote_outputs')
        self.phase_curve_data = os.path.join(self.outputs, 'visible_phase_curve_data')
        self.precalculated = os.path.join(self.outputs, 'precalculated')

        # Precomputed data directories
        self.view_factors = os.path.join(self.precalculated, 'view_factors')
        self.visible_facets = os.path.join(self.precalculated, 'visible_facets')
        self.other_cached_data = os.path.join(self.outputs, 'other_cached_data')
        self.thermal_view_factors = os.path.join(self.data, 'thermal_view_factors')

    def _find_project_root(self):
        """Find the project root directory by looking for .git folder or fallback to parent of src directory or current working directory"""
        current = os.path.abspath(os.path.dirname(__file__))
        while True:
            # If .git folder exists, this is the project root.
            if os.path.exists(os.path.join(current, '.git')):
                return current
            # If we're at the src directory, project root is its parent.
            if os.path.basename(current) == "src":
                return os.path.dirname(current)
            parent = os.path.dirname(current)
            # If we've reached the filesystem root or cannot move up further, fallback to cwd.
            if parent == current:
                return os.getcwd()
            current = parent
        raise RuntimeError("Could not find project root directory")

    def get_shape_model_path(self, filename):
        # If a custom directory is provided in config, use that instead of default shape_models directory
        if hasattr(self, 'shape_model_directory'):
            return os.path.join(self.shape_model_directory, filename)
        
        # Fall back to default location
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

    def get_emission_lut_path(self, lut_name):
        """Get the path to an emission LUT file."""
        return os.path.join(self.data, 'emission_luts', lut_name)

    def get_thermal_view_factors_path(self, shape_model_hash):
        """Get the path for cached thermal view factors."""
        return os.path.join(self.thermal_view_factors, f'thermal_view_factors_{shape_model_hash}.npz')
