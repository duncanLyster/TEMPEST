# src/model/fracture_heating.py

import numpy as np
import math

class FractureHeating:
    """
    Implements spatially varying base temperatures based on distance from fractures.
    
    This allows modeling of subsurface heating beneath localized fractures on icy bodies
    such as Europa and Enceladus, enabling maps of surface temperature that reflect 
    spatially non-uniform internal heat flow.
    """
    
    def __init__(self, config):
        """
        Initialize fracture heating parameters from config.
        """
        # Get parameters from config_data dict to handle both Config objects and simulation objects
        config_data = getattr(config, 'config_data', None)
        if config_data is None:
            # If config is a simulation object, it has attributes directly
            config_data = config.__dict__
        
        self.enabled = config_data.get('enable_fracture_heating', False)
        self.fracture_direction = config_data.get('fracture_direction', 'x')  # 'x', 'y', or 'z'
        self.fracture_position = config_data.get('fracture_position', 0.0)  # position of fracture line
        self.peak_temperature = config_data.get('fracture_peak_temperature', 225.0)  # K
        self.background_temperature = config_data.get('fracture_background_temperature', 100.0)  # K
        self.characteristic_distance = config_data.get('fracture_characteristic_distance', 100.0)  # m
        self.temperature_profile = config_data.get('fracture_temperature_profile', 'exponential')  # 'exponential' or 'linear'
        self.coordinate_scale = config_data.get('fracture_coordinate_scale', 1000.0)  # Scale factor to convert shape model units to meters (km->m = 1000)
        self.rotation_angle_degrees = config_data.get('fracture_rotation_degrees', 0.0)  # Rotation angle to align fracture
        self.fracture_offset_x = config_data.get('fracture_offset_x', 0.0)  # X offset for fracture center (m, applied before rotation)
        self.fracture_offset_y = config_data.get('fracture_offset_y', 0.0)  # Y offset for fracture center (m, applied before rotation)
        
    def calculate_distance_from_fracture(self, facet_positions):
        """
        Calculate distance from each facet to the fracture line.
        
        Args:
            facet_positions: numpy array of shape (n_facets, 3) with facet centroid positions
            
        Returns:
            distances: numpy array of shape (n_facets,) with distances from fracture
        """
        # Scale coordinates from shape model units to meters
        scaled_positions = facet_positions * self.coordinate_scale
        
        # Apply lateral offsets to translate fracture center
        offset_positions = scaled_positions.copy()
        offset_positions[:, 0] -= self.fracture_offset_x  # Subtract to move fracture center to desired location
        offset_positions[:, 1] -= self.fracture_offset_y
        
        # Apply rotation if specified (clockwise rotation in x-y plane)
        if self.rotation_angle_degrees != 0.0:
            angle_rad = np.radians(self.rotation_angle_degrees)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            
            # Rotate coordinates (clockwise rotation matrix)
            x_rotated = offset_positions[:, 0] * cos_angle + offset_positions[:, 1] * sin_angle
            y_rotated = -offset_positions[:, 0] * sin_angle + offset_positions[:, 1] * cos_angle
            z_rotated = offset_positions[:, 2]  # Z unchanged for x-y plane rotation
            
            transformed_positions = np.column_stack((x_rotated, y_rotated, z_rotated))
        else:
            transformed_positions = offset_positions
        
        # Calculate distance from fracture line
        if self.fracture_direction.lower() == 'x':
            # Fracture is along x=fracture_position line (perpendicular to x-axis)
            distances = np.abs(transformed_positions[:, 0] - self.fracture_position)
        elif self.fracture_direction.lower() == 'y':
            # Fracture is along y=fracture_position line (perpendicular to y-axis)
            distances = np.abs(transformed_positions[:, 1] - self.fracture_position)
        elif self.fracture_direction.lower() == 'z':
            # Fracture is along z=fracture_position line (perpendicular to z-axis)
            distances = np.abs(transformed_positions[:, 2] - self.fracture_position)
        else:
            raise ValueError(f"Invalid fracture_direction: {self.fracture_direction}. Must be 'x', 'y', or 'z'")
            
        return distances
    
    def calculate_base_temperature(self, distance):
        """
        Calculate base temperature as a function of distance from fracture.
        
        Args:
            distance: distance from fracture (meters)
            
        Returns:
            temperature: base temperature (K)
        """
        if self.temperature_profile == 'exponential':
            # Exponential decay: T = T_bg + (T_peak - T_bg) * exp(-distance / characteristic_distance)
            decay_factor = np.exp(-distance / self.characteristic_distance)
            temperature = self.background_temperature + (self.peak_temperature - self.background_temperature) * decay_factor
        elif self.temperature_profile == 'linear':
            # Linear decay with cutoff at characteristic_distance
            if distance <= self.characteristic_distance:
                temperature = self.peak_temperature - (self.peak_temperature - self.background_temperature) * (distance / self.characteristic_distance)
            else:
                temperature = self.background_temperature
        else:
            raise ValueError(f"Invalid temperature_profile: {self.temperature_profile}. Must be 'exponential' or 'linear'")
            
        return max(temperature, self.background_temperature)  # Ensure temperature doesn't go below background
    
    def calculate_spatially_varying_temperatures(self, facet_positions, mean_insolations, emissivity):
        """
        Calculate spatially varying base temperatures for all facets.
        
        Args:
            facet_positions: numpy array of shape (n_facets, 3) with facet centroid positions
            mean_insolations: numpy array of shape (n_facets,) with mean insolation for each facet
            emissivity: surface emissivity
            
        Returns:
            temperatures: numpy array of shape (n_facets,) with base temperatures
        """
        if not self.enabled:
            # Fall back to standard Stefan-Boltzmann calculation
            sigma = 5.67e-8
            temperatures = (mean_insolations / (emissivity * sigma))**(1/4)
            return temperatures
        
        # Calculate distances from fracture
        distances = self.calculate_distance_from_fracture(facet_positions)
        
        # Calculate base temperatures for each facet
        n_facets = len(facet_positions)
        temperatures = np.zeros(n_facets)
        
        for i in range(n_facets):
            temperatures[i] = self.calculate_base_temperature(distances[i])
            
        return temperatures
    
    def get_fracture_info(self):
        """
        Get information about the fracture heating configuration.
        
        Returns:
            dict with fracture heating parameters
        """
        return {
            'enabled': self.enabled,
            'fracture_direction': self.fracture_direction,
            'fracture_position': self.fracture_position,
            'peak_temperature': self.peak_temperature,
            'background_temperature': self.background_temperature,
            'characteristic_distance': self.characteristic_distance,
            'temperature_profile': self.temperature_profile,
            'coordinate_scale': self.coordinate_scale,
            'rotation_angle_degrees': self.rotation_angle_degrees,
            'fracture_offset_x': self.fracture_offset_x,
            'fracture_offset_y': self.fracture_offset_y
        }
