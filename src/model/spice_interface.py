# src/model/spice_interface.py

"""
SPICE Interface Module for TEMPEST

This module provides a clean interface to NASA's SPICE toolkit via SpiceyPy,
handling geometry queries for solar system bodies including:
- Sun direction and distance
- Body orientation and rotation states
- Observer positions and viewing geometry
"""

import numpy as np
import spiceypy as spice
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import warnings


class SpiceManager:
    """
    Manages SPICE kernel loading and provides geometry query methods.
    
    This class handles all interactions with the SPICE toolkit, providing
    a clean interface for retrieving geometry information needed by TEMPEST.
    """
    
    def __init__(self, kernel_paths: List[str], target_body: str, 
                 observer: str = "SUN", aberration_correction: str = "LT+S"):
        """
        Initialize the SPICE manager and load kernels.
        
        Args:
            kernel_paths: List of paths to SPICE kernel files
            target_body: NAIF name or ID of target body (e.g., "BENNU", "2101955")
            observer: NAIF name or ID of observer (default: "SUN")
            aberration_correction: SPICE aberration correction string
                                  (e.g., "LT+S", "CN+S", "NONE")
        """
        self.target_body = target_body
        self.observer = observer
        self.aberration_correction = aberration_correction
        self.loaded_kernels = []
        self._body_frame = None
        self._inertial_frame = "J2000"
        
        # Load all specified kernels
        self._load_kernels(kernel_paths)
        
        # Try to determine the body-fixed frame
        self._determine_body_frame()
        
    def _load_kernels(self, kernel_paths: List[str]):
        """Load SPICE kernels from the provided paths."""
        for kernel_path in kernel_paths:
            path = Path(kernel_path)
            if not path.exists():
                warnings.warn(f"Kernel file not found: {kernel_path}")
                continue
                
            try:
                spice.furnsh(str(path.absolute()))
                self.loaded_kernels.append(str(path.absolute()))
            except Exception as e:
                warnings.warn(f"Failed to load kernel {kernel_path}: {e}")
                
        if not self.loaded_kernels:
            raise RuntimeError("No SPICE kernels were successfully loaded")
            
    def _determine_body_frame(self):
        """Determine the body-fixed reference frame for the target."""
        try:
            # Try to get the body-fixed frame using standard naming
            # Most bodies have frames like "IAU_BENNU", "IAU_EARTH", etc.
            body_id = self._get_body_id(self.target_body)
            
            # Try common frame naming conventions
            frame_candidates = [
                f"IAU_{self.target_body.upper()}",
                f"{self.target_body.upper()}_FIXED",
                f"{body_id}_FIXED"
            ]
            
            for frame in frame_candidates:
                try:
                    # Test if frame is available by trying a dummy conversion
                    spice.pxform(self._inertial_frame, frame, 0.0)
                    self._body_frame = frame
                    break
                except:
                    continue
                    
            if self._body_frame is None:
                warnings.warn(
                    f"Could not automatically determine body-fixed frame for {self.target_body}. "
                    "Body orientation queries may not work correctly."
                )
        except Exception as e:
            warnings.warn(f"Error determining body frame: {e}")
            
    def _get_body_id(self, body_name: str) -> int:
        """Get NAIF ID for a body name."""
        try:
            # If already an integer, return it
            if isinstance(body_name, int):
                return body_name
            # Try to convert string to int
            try:
                return int(body_name)
            except ValueError:
                pass
            # Look up the ID from the name
            return spice.bodn2c(body_name)
        except:
            raise ValueError(f"Could not resolve NAIF ID for body: {body_name}")
            
    def time_str_to_et(self, time_str: str) -> float:
        """
        Convert a time string to ephemeris time (ET).
        
        Args:
            time_str: Time string in UTC format (e.g., "2019-03-01T00:00:00")
            
        Returns:
            Ephemeris time (seconds past J2000 epoch)
        """
        return spice.str2et(time_str)
        
    def et_to_time_str(self, et: float, format_str: str = "ISOC") -> str:
        """
        Convert ephemeris time to a time string.
        
        Args:
            et: Ephemeris time
            format_str: SPICE time format string (default: ISO calendar format)
            
        Returns:
            Time string
        """
        return spice.et2utc(et, format_str, 3)
        
    def get_sun_direction_and_distance(self, et: float, 
                                      in_body_frame: bool = True) -> Tuple[np.ndarray, float]:
        """
        Get the direction to the Sun and the distance at a given time.
        
        Args:
            et: Ephemeris time
            in_body_frame: If True, return direction in body-fixed frame;
                          if False, return in inertial frame
            
        Returns:
            Tuple of (direction vector, distance in meters)
            Direction vector is normalized and points FROM body TO sun
        """
        # Get position of Sun relative to target body
        state, _ = spice.spkezr(
            "SUN",
            et,
            self._body_frame if in_body_frame else self._inertial_frame,
            self.aberration_correction,
            self.target_body
        )
        
        # Extract position (first 3 elements are position, next 3 are velocity)
        position = np.array(state[:3])
        distance = np.linalg.norm(position) * 1000  # Convert km to meters
        direction = position / np.linalg.norm(position)
        
        return direction, distance
        
    def get_body_orientation(self, et: float) -> np.ndarray:
        """
        Get the rotation matrix from inertial frame to body-fixed frame.
        
        Args:
            et: Ephemeris time
            
        Returns:
            3x3 rotation matrix that transforms vectors from inertial
            frame to body-fixed frame
        """
        if self._body_frame is None:
            raise RuntimeError(
                "Body-fixed frame not available. Cannot compute body orientation."
            )
            
        # Get rotation matrix from inertial to body-fixed frame
        rotation_matrix = spice.pxform(
            self._inertial_frame,
            self._body_frame,
            et
        )
        
        return np.array(rotation_matrix)
        
    def get_observer_state(self, et: float, observer_name: Optional[str] = None,
                          in_body_frame: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get observer position and velocity relative to target body.
        
        Args:
            et: Ephemeris time
            observer_name: Name of observer (uses self.observer if None)
            in_body_frame: If True, return in body-fixed frame
            
        Returns:
            Tuple of (position vector in meters, velocity vector in m/s)
        """
        if observer_name is None:
            observer_name = self.observer
            
        # Get state of observer relative to target
        state, _ = spice.spkezr(
            observer_name,
            et,
            self._body_frame if in_body_frame else self._inertial_frame,
            self.aberration_correction,
            self.target_body
        )
        
        position = np.array(state[:3]) * 1000  # Convert km to m
        velocity = np.array(state[3:6]) * 1000  # Convert km/s to m/s
        
        return position, velocity
        
    def get_illumination_angles(self, et: float, surface_point: np.ndarray,
                               normal: np.ndarray) -> Dict[str, float]:
        """
        Calculate illumination angles for a surface point.
        
        Args:
            et: Ephemeris time
            surface_point: Surface point coordinates in body-fixed frame (meters)
            normal: Surface normal vector in body-fixed frame (unit vector)
            
        Returns:
            Dictionary with keys:
                - 'incidence_deg': Solar incidence angle in degrees
                - 'emission_deg': Emission angle to observer in degrees (if observer defined)
                - 'phase_deg': Phase angle in degrees
                - 'solar_distance_m': Distance to Sun in meters
        """
        # Get Sun direction and distance
        sun_dir, sun_dist = self.get_sun_direction_and_distance(et, in_body_frame=True)
        
        # Calculate incidence angle
        cos_inc = np.dot(sun_dir, normal)
        incidence = np.degrees(np.arccos(np.clip(cos_inc, -1.0, 1.0)))
        
        result = {
            'incidence_deg': incidence,
            'solar_distance_m': sun_dist,
            'phase_deg': 0.0
        }
        
        # If observer is not the Sun, calculate emission and phase angles
        if self.observer.upper() != "SUN":
            try:
                obs_pos, _ = self.get_observer_state(et, in_body_frame=True)
                obs_dir = obs_pos - (surface_point / 1000)  # Convert surface point to km
                obs_dir = obs_dir / np.linalg.norm(obs_dir)
                
                # Emission angle
                cos_em = np.dot(obs_dir, normal)
                emission = np.degrees(np.arccos(np.clip(cos_em, -1.0, 1.0)))
                result['emission_deg'] = emission
                
                # Phase angle (angle between sun and observer as seen from surface point)
                cos_phase = np.dot(sun_dir, obs_dir)
                phase = np.degrees(np.arccos(np.clip(cos_phase, -1.0, 1.0)))
                result['phase_deg'] = phase
            except:
                warnings.warn("Could not calculate emission/phase angles")
                
        return result
        
    def get_geometry_at_timesteps(self, et_array: np.ndarray,
                                  compute_orientations: bool = True) -> Dict:
        """
        Compute geometry for multiple timesteps efficiently.
        
        Args:
            et_array: Array of ephemeris times
            compute_orientations: If True, compute body orientation matrices
            
        Returns:
            Dictionary with keys:
                - 'sun_directions': (N, 3) array of sun direction vectors
                - 'solar_distances': (N,) array of distances in meters
                - 'body_orientations': (N, 3, 3) array of rotation matrices (if requested)
                - 'et_times': Copy of input et_array
        """
        n_times = len(et_array)
        sun_directions = np.zeros((n_times, 3))
        solar_distances = np.zeros(n_times)
        
        for i, et in enumerate(et_array):
            sun_dir, sun_dist = self.get_sun_direction_and_distance(
                et, in_body_frame=True
            )
            sun_directions[i] = sun_dir
            solar_distances[i] = sun_dist
            
        result = {
            'sun_directions': sun_directions,
            'solar_distances': solar_distances,
            'et_times': et_array.copy()
        }
        
        if compute_orientations and self._body_frame is not None:
            body_orientations = np.zeros((n_times, 3, 3))
            for i, et in enumerate(et_array):
                body_orientations[i] = self.get_body_orientation(et)
            result['body_orientations'] = body_orientations
            
        return result
        
    def cleanup(self):
        """Unload all loaded SPICE kernels."""
        for kernel in self.loaded_kernels:
            try:
                spice.unload(kernel)
            except:
                pass
        self.loaded_kernels = []
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup kernels."""
        self.cleanup()
        return False
        
    def __del__(self):
        """Destructor - ensure kernels are unloaded."""
        self.cleanup()

