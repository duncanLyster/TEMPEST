"""
Forward model wrapper - runs TEMPEST with given thermal inertia.

This module orchestrates TEMPEST execution and radiance extraction
for given thermal inertia values, used in fitting optimization loops.
"""

import sys
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import yaml
import logging

# Set up logging
logger = logging.getLogger(__name__)


class ForwardModel:
    """Wrapper for running TEMPEST and extracting thermal radiance."""
    
    def __init__(
        self,
        config_path: Path,
        wavelengths: np.ndarray,
        phase_angle: float = 0.0,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize forward model with TEMPEST configuration.
        
        Args:
            config_path: Path to TEMPEST YAML config file
            wavelengths: Wavelengths for radiance computation [m]
            phase_angle: Observer phase angle [radians]
            output_dir: Optional directory for intermediate outputs
        """
        self.config_path = Path(config_path)
        self.wavelengths = wavelengths
        self.phase_angle = phase_angle
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base config
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        logger.info(f"ForwardModel initialized with config: {self.config_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_tempest(self, thermal_inertia: float) -> Tuple[np.ndarray, object, object]:
        """
        Run TEMPEST simulation with given thermal inertia.
        
        Args:
            thermal_inertia: Thermal inertia [W m^-2 K^-1 s^-0.5]
            
        Returns:
            temperatures: (n_facets, n_timesteps) temperature array [K]
            shape_model: Facet list
            simulation: Simulation object
            
        Raises:
            RuntimeError: If TEMPEST execution fails
        """
        # Create modified config with target thermal_inertia
        config = self.base_config.copy()
        config['thermal_inertia'] = float(thermal_inertia)
        
        # Save to temporary config file
        temp_config_path = self.output_dir / f"config_ti_{thermal_inertia:.0f}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Running TEMPEST with TI={thermal_inertia:.1f}")
        
        # Import TEMPEST modules directly
        tempest_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(tempest_root))
        
        try:
            # Import core TEMPEST components
            from src.utilities.config import Config
            from src.model.simulation import Simulation, ThermalData
            from src.model.solvers import TemperatureSolverFactory
            from src.model.insolation import calculate_insolation
            from src.model.view_factors import (
                calculate_and_cache_visible_facets,
                calculate_all_view_factors,
                calculate_thermal_view_factors,
            )
            from src.utilities.utils import conditional_print
            import gc
            from numba.typed import List
            
            # Step 1: Load configuration
            config_obj = Config(config_path=str(temp_config_path))
            simulation = Simulation(config_obj)
            
            # Step 2: Read shape model
            def read_shape_model_inline(filename, timesteps_per_day, n_layers, max_days, calculate_energy_terms):
                """Read STL shape model."""
                from src.model.facet import Facet
                from stl import mesh as stl_mesh_module
                
                if not Path(filename).is_file():
                    raise FileNotFoundError(f"Shape model not found: {filename}")
                
                try:
                    # Try ASCII first
                    with open(filename, 'r') as f:
                        lines = f.readlines()
                    
                    shape_model = []
                    for i in range(len(lines)):
                        if lines[i].strip().startswith('facet normal'):
                            normal = np.array([float(n) for n in lines[i].strip().split()[2:]])
                            vertex1 = np.array([float(v) for v in lines[i+2].strip().split()[1:]])
                            vertex2 = np.array([float(v) for v in lines[i+3].strip().split()[1:]])
                            vertex3 = np.array([float(v) for v in lines[i+4].strip().split()[1:]])
                            facet = Facet(normal, [vertex1, vertex2, vertex3], 
                                        timesteps_per_day, max_days, n_layers, calculate_energy_terms)
                            shape_model.append(facet)
                except UnicodeDecodeError:
                    # Fall back to binary
                    shape_mesh = stl_mesh_module.Mesh.from_file(filename)
                    shape_model = []
                    for i in range(len(shape_mesh.vectors)):
                        from src.model.facet import Facet
                        facet = Facet(
                            shape_mesh.normals[i],
                            shape_mesh.vectors[i],
                            timesteps_per_day, max_days, n_layers, calculate_energy_terms
                        )
                        shape_model.append(facet)
                
                return shape_model
            
            shape_model = read_shape_model_inline(
                config_obj.path_to_shape_model_file,
                simulation.timesteps_per_day,
                simulation.n_layers,
                simulation.max_days,
                config_obj.calculate_energy_terms
            )
            
            thermal_data = ThermalData(
                len(shape_model),
                simulation.timesteps_per_day,
                simulation.n_layers,
                simulation.max_days,
                config_obj.calculate_energy_terms
            )
            
            # Step 3: Calculate insolation
            positions = np.array([facet.position for facet in shape_model])
            normals = np.array([facet.normal for facet in shape_model])
            vertices = np.array([facet.vertices for facet in shape_model])
            
            visible_indices = calculate_and_cache_visible_facets(
                config_obj.silent_mode, shape_model, positions, normals, vertices, config_obj
            )
            thermal_data.set_visible_facets(visible_indices)
            
            for i, facet in enumerate(shape_model):
                facet.visible_facets = visible_indices[i]
            
            if config_obj.include_self_heating or config_obj.n_scatters > 0:
                all_view_factors = calculate_all_view_factors(
                    shape_model, thermal_data, config_obj, config_obj.vf_rays
                )
                thermal_data.set_secondary_radiation_view_factors(all_view_factors)
                
                if config_obj.include_self_heating:
                    thermal_view_factors = calculate_thermal_view_factors(
                        shape_model, thermal_data, config_obj
                    )
                    thermal_data.set_thermal_view_factors(thermal_view_factors)
            
            thermal_data = calculate_insolation(thermal_data, shape_model, simulation, config_obj)
            
            # Step 4: Initialize and solve
            solver = TemperatureSolverFactory.create(config_obj.temp_solver)
            thermal_data = solver.initialize_temperatures(thermal_data, simulation, config_obj)
            
            # Convert visible facets to numba format
            numba_visible_facets = List()
            for facets in thermal_data.visible_facets:
                numba_visible_facets.append(np.array(facets, dtype=np.int64))
            thermal_data.visible_facets = numba_visible_facets
            
            # Free memory and solve
            shape_model_backup = shape_model
            shape_model = None
            gc.collect()
            
            result = solver.solve(thermal_data, shape_model, simulation, config_obj)
            
            # Restore shape model
            shape_model = shape_model_backup
            
            logger.info(f"TEMPEST completed for TI={thermal_inertia:.1f}")
            
            return thermal_data.temperatures, shape_model, simulation
            
        except Exception as e:
            logger.error(f"TEMPEST execution failed: {e}")
            raise RuntimeError(f"TEMPEST execution failed: {e}")
    
    def compute_radiance(
        self,
        temperatures: np.ndarray,
        shape_model,
        simulation,
        emissivity: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute disc-integrated spectral flux from TEMPEST output.
        
        Returns flux in arbitrary units (to be scaled when fitting to data).
        Only the spectrum shape matters for parameter optimization.
        
        Args:
            temperatures: (n_facets, n_timesteps) temperature array [K]
            shape_model: Facet list
            simulation: Simulation object
            emissivity: Optional override of simulation.emissivity
            
        Returns:
            Spectral flux [arbitrary units], shape (n_wavelengths,)
        """
        from src.analysis.disc_integrated_radiance import compute_disc_integrated_radiance
        
        if emissivity is None:
            emissivity = simulation.emissivity
        
        flux = compute_disc_integrated_radiance(
            temperatures,
            shape_model,
            self.wavelengths,
            simulation,
            phase_angle=self.phase_angle,
            emissivity=emissivity,
        )
        
        return flux
    
    def evaluate(self, thermal_inertia: float) -> np.ndarray:
        """
        Full forward model evaluation: run TEMPEST and compute radiance.
        
        Args:
            thermal_inertia: Thermal inertia [W m^-2 K^-1 s^-0.5]
            
        Returns:
            Spectral radiance [W/(m^3·sr)]
        """
        temperatures, shape_model, simulation = self.run_tempest(thermal_inertia)
        radiance = self.compute_radiance(temperatures, shape_model, simulation)
        return radiance


# Simplified version for testing without full TEMPEST integration
def forward_model_isothermal(
    thermal_inertia: float,
    wavelengths: np.ndarray,
    reference_ti: float = 300.0,
    reference_temperature: float = 250.0,
) -> np.ndarray:
    """
    Simple isothermal forward model for testing.
    
    Scales temperature with thermal inertia as:
    T(TI) = T_ref × (TI / TI_ref)^(-0.25)
    
    This is a rough approximation; full TEMPEST should be used for accuracy.
    
    Args:
        thermal_inertia: Thermal inertia value [W m^-2 K^-1 s^-0.5]
        wavelengths: Wavelengths [m]
        reference_ti: Reference TI for scaling [W m^-2 K^-1 s^-0.5]
        reference_temperature: Temperature at reference TI [K]
        
    Returns:
        Spectral radiance [W/(m^3·sr)]
    """
    from retrieve_radiance import planck_radiance
    
    # Temperature scaling with thermal inertia
    # Higher TI → more heat capacity → cooler equilibrium temperature
    temperature = reference_temperature * (thermal_inertia / reference_ti) ** (-0.25)
    
    # Compute Planck radiance
    radiance = planck_radiance(wavelengths, temperature, emissivity=0.95)
    
    return radiance
