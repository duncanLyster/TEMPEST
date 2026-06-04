"""
Module for caching and managing scattered light insolation results.

This module provides functionality to save and load pre-computed insolation arrays
(including direct and scattered light components) for a given shape model and rotation setup.
This significantly speeds up reruns where only thermal properties (like thermal inertia) change.

The cache validates that illumination settings match before reusing cached results.
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from src.utilities.utils import conditional_print


class InsolationCache:
    """
    Manages caching of insolation results with validation.
    
    Cached files are stored based on a hash of:
    - Shape model file path
    - Solar distance (AU)
    - Sunlight direction
    - Albedo
    - Rotation period
    - Number of scattering iterations
    - Timesteps per day
    - Include shadowing flag
    """
    
    def __init__(self, cache_dir):
        """
        Initialize the insolation cache.
        
        Args:
            cache_dir: Directory where cached insolation files will be stored
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    @staticmethod
    def _create_illumination_hash(config, simulation, shape_model_path):
        """
        Create a hash based on illumination and geometric setup parameters.
        
        Parameters affecting insolation that are NOT thermal properties:
        - Shape model file
        - Solar distance
        - Sunlight direction
        - Albedo
        - Rotation period / rotation axis
        - Number of scattering iterations
        - Timesteps per day
        - Include shadowing
        - Scattering LUT filename
        
        Args:
            config: Configuration object
            simulation: Simulation object
            shape_model_path: Path to the shape model file
            
        Returns:
            str: Hex hash of the illumination setup
        """
        # Create a dictionary of all illumination-relevant parameters
        params = {
            'shape_model': os.path.basename(shape_model_path),
            'solar_distance_au': float(config.solar_distance_au),
            'sunlight_direction': [float(x) for x in simulation.sunlight_direction],
            'albedo': float(simulation.albedo),
            'rotation_period_hours': float(config.rotation_period_hours),
            'rotation_axis': [float(x) for x in simulation.rotation_axis],
            'n_scatters': int(config.n_scatters),
            'timesteps_per_day': int(simulation.timesteps_per_day),
            'include_shadowing': bool(config.include_shadowing),
            'scattering_lut': str(config.scattering_lut),
            'obliquity_degrees': float(getattr(config, 'obliquity_degrees', 0)),
            'north_pole_solar_longitude_degrees': float(getattr(config, 'north_pole_solar_longitude_degrees', 0)),
            'shape_model_rotation_x_degrees': float(getattr(config, 'shape_model_rotation_x_degrees', 0)),
            'shape_model_rotation_z_degrees': float(getattr(config, 'shape_model_rotation_z_degrees', 0)),
        }
        
        # Convert to JSON for consistent hashing
        params_json = json.dumps(params, sort_keys=True)
        
        # Create hash
        hash_obj = hashlib.sha256(params_json.encode())
        return hash_obj.hexdigest()
    
    def get_insolation_cache_path(self, hash_code):
        """Get the full path for a cached insolation file."""
        return os.path.join(self.cache_dir, f"insolation_{hash_code}.npz")
    
    def get_metadata_path(self, hash_code):
        """Get the path for metadata associated with a cache file."""
        return os.path.join(self.cache_dir, f"insolation_{hash_code}_metadata.json")
    
    def save_insolation(self, insolation_array, config, simulation, shape_model_path, 
                       silent_mode=False):
        """
        Save an insolation array to cache.
        
        Args:
            insolation_array: NumPy array of shape (n_facets, timesteps_per_day)
            config: Configuration object
            simulation: Simulation object
            shape_model_path: Path to the shape model file
            silent_mode: Whether to suppress console output
            
        Returns:
            str: Path to the saved cache file
        """
        hash_code = self._create_illumination_hash(config, simulation, shape_model_path)
        cache_path = self.get_insolation_cache_path(hash_code)
        metadata_path = self.get_metadata_path(hash_code)
        
        # Save the insolation array
        np.savez_compressed(cache_path, insolation=insolation_array)
        
        # Save metadata for reference
        metadata = {
            'shape_model': os.path.basename(shape_model_path),
            'solar_distance_au': float(config.solar_distance_au),
            'albedo': float(simulation.albedo),
            'rotation_period_hours': float(config.rotation_period_hours),
            'n_scatters': int(config.n_scatters),
            'timesteps_per_day': int(simulation.timesteps_per_day),
            'n_facets': int(insolation_array.shape[0]),
            'include_shadowing': bool(config.include_shadowing),
            'scattering_lut': str(config.scattering_lut),
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        conditional_print(silent_mode, 
                         f"Cached insolation results to: {cache_path}")
        
        return cache_path
    
    def load_insolation(self, config, simulation, shape_model_path, 
                       expected_n_facets, silent_mode=False):
        """
        Load cached insolation if it exists and is valid.
        
        Args:
            config: Configuration object
            simulation: Simulation object
            shape_model_path: Path to the shape model file
            expected_n_facets: Expected number of facets (for validation)
            silent_mode: Whether to suppress console output
            
        Returns:
            np.ndarray or None: Cached insolation array, or None if cache not found/invalid
        """
        hash_code = self._create_illumination_hash(config, simulation, shape_model_path)
        cache_path = self.get_insolation_cache_path(hash_code)
        metadata_path = self.get_metadata_path(hash_code)
        
        # Check if cache file exists
        if not os.path.exists(cache_path):
            conditional_print(silent_mode, 
                             f"No cached insolation found for this configuration.")
            return None
        
        # Load and validate metadata
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Validate number of facets
                if metadata.get('n_facets') != expected_n_facets:
                    conditional_print(silent_mode,
                                    f"Cached insolation has {metadata.get('n_facets')} facets, "
                                    f"but shape model has {expected_n_facets} facets. Cache invalid.")
                    return None
                    
            except (json.JSONDecodeError, IOError) as e:
                conditional_print(silent_mode,
                                f"Could not read cache metadata: {e}. Cache may be corrupted.")
                return None
        
        # Load the insolation array
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                insolation = data['insolation']
            
            # Final validation
            if insolation.shape[0] != expected_n_facets:
                conditional_print(silent_mode,
                                f"Loaded insolation has {insolation.shape[0]} facets, "
                                f"expected {expected_n_facets}. Cache invalid.")
                return None
            
            conditional_print(silent_mode,
                            f"Loaded cached insolation from: {cache_path}")
            return insolation
            
        except (OSError, KeyError) as e:
            conditional_print(silent_mode,
                            f"Could not load cached insolation: {e}")
            return None
    
    def clear_cache(self, silent_mode=False):
        """
        Clear all cached insolation files.
        
        Args:
            silent_mode: Whether to suppress console output
        """
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            conditional_print(silent_mode, f"Cleared insolation cache at: {self.cache_dir}")
    
    def list_cached_results(self):
        """
        List all cached insolation results with their metadata.
        
        Returns:
            list: List of dictionaries containing cache information
        """
        results = []
        
        for filename in os.listdir(self.cache_dir):
            if filename.startswith('insolation_') and filename.endswith('_metadata.json'):
                metadata_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    results.append(metadata)
                except (json.JSONDecodeError, IOError):
                    pass
        
        return results
