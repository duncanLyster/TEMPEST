# src/model/simulation.py

import numpy as np
from src.utilities.locations import Locations
from typing import Optional

class Simulation:
    def __init__(self, config):
        """
        Initialize the simulation using the provided Config object.
        """
        self.config = config
        self.spice_manager = None
        self.use_spice = config.use_spice
        self.load_configuration()
        
        # Initialize SPICE if enabled
        if self.use_spice:
            self._initialize_spice()

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
        self.thermal_conductivity = (self.thermal_inertia**2 / (self.density * self.specific_heat_capacity))
        self.skin_depth = (self.thermal_conductivity / (self.density * self.specific_heat_capacity * self.angular_velocity)) ** 0.5
        self.layer_thickness = 8 * self.skin_depth / self.n_layers
        self.thermal_diffusivity = self.thermal_conductivity / (self.density * self.specific_heat_capacity)
        self.timesteps_per_day = self.calculate_adaptive_timesteps() # Adaptive timestep for low thermal inertia stability
        self.delta_t = self.rotation_period_s / self.timesteps_per_day
        
        # Compute unit vector from RA and Dec (only used in non-SPICE mode)
        ra_radians = np.radians(self.ra_degrees)
        dec_radians = np.radians(self.dec_degrees)
        self.rotation_axis = np.array([np.cos(ra_radians) * np.cos(dec_radians), 
                                       np.sin(ra_radians) * np.cos(dec_radians), 
                                       np.sin(dec_radians)])
        
        # Initialize arrays for SPICE-derived geometry (populated later if use_spice=True)
        self.spice_solar_distances = None
        self.spice_sun_directions = None
        self.spice_body_orientations = None
        self.spice_et_times = None

    def calculate_adaptive_timesteps(self):
        """
        Calculate timesteps with stability constraints for low thermal inertia.
        
        This method adds an adaptive constraint that limits const1 to reasonable
        values, ensuring stability for low thermal inertia materials.
        """
        # Original CFL calculation
        cfl_denominator = self.layer_thickness**2 / (2 * self.thermal_diffusivity)
        timesteps_cfl = int(round(self.rotation_period_s / cfl_denominator))
        delta_t_cfl = self.rotation_period_s / timesteps_cfl
        
        # Calculate insolation coefficient with CFL timestep
        const1_cfl = delta_t_cfl / (self.layer_thickness * self.density * self.specific_heat_capacity)
        
        # Adaptive constraint: limit const1 for stability
        max_const1 = 0.1  # Maximum allowed insolation coefficient
        
        if const1_cfl > max_const1:
            # Calculate timestep that keeps const1 reasonable
            required_delta_t = max_const1 * self.layer_thickness * self.density * self.specific_heat_capacity
            adaptive_timesteps = int(np.ceil(self.rotation_period_s / required_delta_t))
            
            # Apply additional safety factor for very low thermal inertia
            if self.thermal_inertia < 100:
                safety_factor = 0.5
                adaptive_timesteps = int(adaptive_timesteps / safety_factor)
            
            return adaptive_timesteps
        else:
            return timesteps_cfl
    
    def _initialize_spice(self):
        """Initialize SPICE manager and precompute geometry."""
        from src.model.spice_interface import SpiceManager
        
        # Validate SPICE configuration
        self.config.validate_spice_config()
        
        # Create SPICE manager
        self.spice_manager = SpiceManager(
            kernel_paths=self.config.spice_kernels,
            target_body=self.config.spice_target_body,
            observer=self.config.spice_observer,
            aberration_correction=self.config.spice_illumination_aberration
        )
        
        # Convert start time to ephemeris time
        et_start = self.spice_manager.time_str_to_et(self.config.spice_start_time)
        
        # If duration is provided, use it; otherwise use the rotation period
        if self.config.spice_duration_hours:
            duration_seconds = self.config.spice_duration_hours * 3600
        else:
            duration_seconds = self.rotation_period_s
        
        # Create time array based on update frequency
        if self.config.spice_update_frequency == 'static':
            # Single geometry for entire simulation
            et_array = np.array([et_start])
        elif self.config.spice_update_frequency == 'per_day':
            # One geometry per day (using max_days)
            et_array = np.linspace(et_start, et_start + duration_seconds, 
                                  self.config.max_days + 1)
        else:  # per_timestep
            # One geometry per timestep
            et_array = np.linspace(et_start, et_start + duration_seconds,
                                  self.timesteps_per_day)
        
        # Precompute all geometry
        geometry = self.spice_manager.get_geometry_at_timesteps(
            et_array, compute_orientations=True
        )
        
        self.spice_et_times = geometry['et_times']
        self.spice_solar_distances = geometry['solar_distances']
        self.spice_sun_directions = geometry['sun_directions']
        
        # Body orientations might not be available for all bodies
        if 'body_orientations' in geometry:
            self.spice_body_orientations = geometry['body_orientations']
        
        # Update simulation parameters based on average SPICE values
        avg_solar_distance_m = np.mean(self.spice_solar_distances)
        self.solar_distance_au = avg_solar_distance_m / 1.496e11
        self.solar_distance_m = avg_solar_distance_m
        
    def get_geometry_at_timestep(self, timestep: int) -> dict:
        """
        Get geometry (sun direction, distance, orientation) at a timestep.
        
        Args:
            timestep: Timestep index (0 to timesteps_per_day-1)
            
        Returns:
            Dictionary with keys:
                - 'sun_direction': Unit vector pointing to Sun
                - 'solar_distance_m': Distance to Sun in meters
                - 'body_orientation': 3x3 rotation matrix (if available)
        """
        if not self.use_spice:
            # Non-SPICE mode: use rotation-based geometry
            from src.utilities.utils import calculate_rotation_matrix
            
            angle = (2 * np.pi / self.timesteps_per_day) * timestep
            rotation_matrix = calculate_rotation_matrix(self.rotation_axis, angle)
            
            # Sun direction in body frame
            sun_dir = rotation_matrix.T.dot(self.sunlight_direction)
            sun_dir = sun_dir / np.linalg.norm(sun_dir)
            
            return {
                'sun_direction': sun_dir,
                'solar_distance_m': self.solar_distance_m,
                'body_orientation': rotation_matrix
            }
        else:
            # SPICE mode: use precomputed geometry
            if self.config.spice_update_frequency == 'static':
                idx = 0
            elif self.config.spice_update_frequency == 'per_day':
                # Map timestep to day (assuming one full rotation per call)
                idx = min(timestep // self.timesteps_per_day, len(self.spice_et_times) - 1)
            else:  # per_timestep
                idx = min(timestep, len(self.spice_et_times) - 1)
            
            result = {
                'sun_direction': self.spice_sun_directions[idx],
                'solar_distance_m': self.spice_solar_distances[idx]
            }
            
            if self.spice_body_orientations is not None:
                result['body_orientation'] = self.spice_body_orientations[idx]
                
            return result
    
    def cleanup_spice(self):
        """Clean up SPICE resources."""
        if self.spice_manager is not None:
            self.spice_manager.cleanup()
            self.spice_manager = None

class ThermalData:
    def __init__(self, n_facets, timesteps_per_day, n_layers, max_days, calculate_energy_terms):
        # Surface temperatures for one day only
        self.temperatures = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        # Two columns for current and previous timestep subsurface temperatures
        self.layer_temperatures = np.zeros((n_facets, 2, n_layers), dtype=np.float64)
        self.insolation = np.zeros((n_facets, timesteps_per_day), dtype=np.float64)
        self.visible_facets = [np.array([], dtype=np.int64) for _ in range(n_facets)]
        self.secondary_radiation_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]
        self.thermal_view_factors = [np.array([], dtype=np.float64) for _ in range(n_facets)]

        if calculate_energy_terms:
            # Energy terms for one day only
            self.insolation_energy = np.zeros((n_facets, timesteps_per_day))
            self.re_emitted_energy = np.zeros((n_facets, timesteps_per_day))
            self.surface_energy_change = np.zeros((n_facets, timesteps_per_day))
            self.conducted_energy = np.zeros((n_facets, timesteps_per_day))
            self.unphysical_energy_loss = np.zeros((n_facets, timesteps_per_day))

    def set_visible_facets(self, visible_facets):
        self.visible_facets = [np.array(facets, dtype=np.int64) for facets in visible_facets]

    def set_secondary_radiation_view_factors(self, view_factors):
        self.secondary_radiation_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]
        
    def set_thermal_view_factors(self, view_factors):
        """Set thermal view factors for all facets."""
        self.thermal_view_factors = [np.array(view_factor, dtype=np.float64) for view_factor in view_factors]