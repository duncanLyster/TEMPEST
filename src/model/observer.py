# src/model/observer.py

"""
Observer Module for TEMPEST

This module defines the Observer class for representing spacecraft, telescopes,
or other observing platforms that view the target body.
"""

import numpy as np
from typing import Optional, List, Tuple
import warnings


class Observer:
    """
    Represents an observer (spacecraft, telescope, etc.) viewing a target body.
    
    The Observer can retrieve its position from SPICE or use a manually
    specified position. It handles visibility calculations and viewing geometry.
    """
    
    def __init__(self, name: str, 
                 spice_manager=None,
                 manual_position: Optional[np.ndarray] = None,
                 fov_degrees: Optional[float] = None):
        """
        Initialize an Observer.
        
        Args:
            name: Name or NAIF ID of the observer
            spice_manager: SpiceManager instance for SPICE-based positions
            manual_position: Manually specified position [x, y, z] in meters
                           (used if spice_manager is None)
            fov_degrees: Field of view in degrees (optional, for future use)
        """
        self.name = name
        self.spice_manager = spice_manager
        self.manual_position = np.array(manual_position) if manual_position is not None else None
        self.fov_degrees = fov_degrees
        
        # Validation
        if spice_manager is None and manual_position is None:
            raise ValueError("Must provide either spice_manager or manual_position")
            
    def get_position(self, et: Optional[float] = None, 
                    in_body_frame: bool = True) -> np.ndarray:
        """
        Get observer position relative to target body.
        
        Args:
            et: Ephemeris time (required if using SPICE)
            in_body_frame: If True, return position in body-fixed frame
            
        Returns:
            Position vector in meters
        """
        if self.spice_manager is not None:
            if et is None:
                raise ValueError("Ephemeris time required when using SPICE")
            position, _ = self.spice_manager.get_observer_state(
                et, self.name, in_body_frame=in_body_frame
            )
            return position
        else:
            # Return manual position
            return self.manual_position.copy()
            
    def get_direction_to_observer(self, surface_point: np.ndarray,
                                  et: Optional[float] = None) -> np.ndarray:
        """
        Get unit vector from surface point to observer.
        
        Args:
            surface_point: Point on surface in meters (body-fixed frame)
            et: Ephemeris time (required if using SPICE)
            
        Returns:
            Unit vector pointing from surface to observer
        """
        obs_position = self.get_position(et, in_body_frame=True)
        to_observer = obs_position - surface_point
        return to_observer / np.linalg.norm(to_observer)
        
    def is_facet_visible(self, facet_position: np.ndarray, 
                        facet_normal: np.ndarray,
                        et: Optional[float] = None,
                        horizon_tolerance: float = 1e-6) -> bool:
        """
        Check if a facet is visible from the observer.
        
        Args:
            facet_position: Facet center position in meters (body-fixed frame)
            facet_normal: Facet normal vector (unit vector)
            et: Ephemeris time (required if using SPICE)
            horizon_tolerance: Tolerance for horizon check
            
        Returns:
            True if facet is visible from observer
        """
        to_observer = self.get_direction_to_observer(facet_position, et)
        
        # Facet is visible if it's facing the observer
        cos_angle = np.dot(facet_normal, to_observer)
        return cos_angle > horizon_tolerance
        
    def calculate_emission_angle(self, facet_position: np.ndarray,
                                 facet_normal: np.ndarray,
                                 et: Optional[float] = None) -> float:
        """
        Calculate emission angle for a facet.
        
        Args:
            facet_position: Facet center position in meters
            facet_normal: Facet normal vector (unit vector)
            et: Ephemeris time (required if using SPICE)
            
        Returns:
            Emission angle in degrees
        """
        to_observer = self.get_direction_to_observer(facet_position, et)
        cos_emission = np.dot(facet_normal, to_observer)
        return np.degrees(np.arccos(np.clip(cos_emission, -1.0, 1.0)))
        
    def get_visible_facets(self, shape_model, et: Optional[float] = None,
                          check_self_shadowing: bool = False) -> List[int]:
        """
        Get indices of facets visible from the observer.
        
        Args:
            shape_model: List of Facet objects
            et: Ephemeris time (required if using SPICE)
            check_self_shadowing: If True, check for shadowing by other facets
                                 (not implemented yet, for future use)
            
        Returns:
            List of visible facet indices
        """
        visible_indices = []
        
        for idx, facet in enumerate(shape_model):
            if self.is_facet_visible(facet.position, facet.normal, et):
                visible_indices.append(idx)
                
        if check_self_shadowing:
            warnings.warn("Self-shadowing check not yet implemented for observer")
            
        return visible_indices
        
    def calculate_projected_area(self, facet_area: float,
                                facet_position: np.ndarray,
                                facet_normal: np.ndarray,
                                et: Optional[float] = None) -> float:
        """
        Calculate projected area of a facet as seen from the observer.
        
        Args:
            facet_area: Physical area of the facet in m^2
            facet_position: Facet center position in meters
            facet_normal: Facet normal vector (unit vector)
            et: Ephemeris time (required if using SPICE)
            
        Returns:
            Projected area accounting for viewing angle and distance
        """
        to_observer = self.get_direction_to_observer(facet_position, et)
        
        # Cosine of emission angle
        cos_emission = np.dot(facet_normal, to_observer)
        
        if cos_emission <= 0:
            return 0.0
            
        # Get observer distance
        obs_position = self.get_position(et, in_body_frame=True)
        distance = np.linalg.norm(obs_position - facet_position)
        
        # Projected area (accounting for viewing angle and distance)
        projected_area = facet_area * cos_emission / (distance ** 2)
        
        return projected_area
        
    def calculate_phase_angle(self, facet_position: np.ndarray,
                            sun_direction: np.ndarray,
                            et: Optional[float] = None) -> float:
        """
        Calculate phase angle at a facet.
        
        Phase angle is the angle between the Sun and observer as seen
        from the facet.
        
        Args:
            facet_position: Facet position in meters
            sun_direction: Direction to Sun (unit vector)
            et: Ephemeris time (required if using SPICE)
            
        Returns:
            Phase angle in degrees
        """
        to_observer = self.get_direction_to_observer(facet_position, et)
        
        cos_phase = np.dot(sun_direction, to_observer)
        phase_angle = np.degrees(np.arccos(np.clip(cos_phase, -1.0, 1.0)))
        
        return phase_angle
        
    def __repr__(self):
        """String representation of the observer."""
        mode = "SPICE" if self.spice_manager is not None else "Manual"
        fov_str = f", FOV={self.fov_degrees}°" if self.fov_degrees else ""
        return f"Observer(name='{self.name}', mode={mode}{fov_str})"

