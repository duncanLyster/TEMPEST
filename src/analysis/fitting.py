"""
Thermal inertia fitting framework - scipy.optimize wrapper.

Implements chi-squared minimization for fitting TEMPEST thermal inertia
to JWST observations.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import logging
from scipy.optimize import minimize, differential_evolution

logger = logging.getLogger(__name__)


class ThermalInertiaFitter:
    """
    Optimizer for fitting thermal inertia to JWST spectroscopic observations.
    
    This class wraps scipy.optimize for parameter space exploration
    and cost function evaluation.
    """
    
    def __init__(
        self,
        forward_model: Callable,
        jwst_wavelengths: np.ndarray,
        jwst_flux: np.ndarray,
        jwst_uncertainty: np.ndarray,
        emissivity: float = 0.95,
    ):
        """
        Initialize fitter with observations and forward model.
        
        Args:
            forward_model: Function(thermal_inertia) -> radiance [W/(m^3·sr)]
            jwst_wavelengths: JWST wavelengths [m]
            jwst_flux: JWST flux density [Jy]
            jwst_uncertainty: JWST flux uncertainty [Jy]
            emissivity: Thermal emissivity [0-1]
        """
        self.forward_model = forward_model
        self.jwst_wavelengths = jwst_wavelengths
        self.jwst_flux = jwst_flux
        self.jwst_uncertainty = jwst_uncertainty
        self.emissivity = emissivity
        
        # Optimization state
        self.iteration_count = 0
        self.history = {
            'thermal_inertia': [],
            'chi_squared': [],
            'radiance': [],
        }
        
        logger.info(f"ThermalInertiaFitter initialized with {len(jwst_flux)} data points")
    
    def cost_function(self, thermal_inertia: float) -> float:
        """
        Compute chi-squared cost for given thermal inertia.
        
        Args:
            thermal_inertia: Thermal inertia [W m^-2 K^-1 s^-0.5]
            
        Returns:
            Chi-squared statistic
        """
        # Handle scipy passing arrays instead of scalars
        if hasattr(thermal_inertia, '__iter__'):
            thermal_inertia = float(thermal_inertia[0])
        else:
            thermal_inertia = float(thermal_inertia)
        
        try:
            # Compute model radiance
            radiance_model = self.forward_model(thermal_inertia)
            
            # Interpolate to JWST wavelengths
            model_flux = np.interp(
                self.jwst_wavelengths,
                self.jwst_wavelengths,  # Assume same wavelengths for now
                radiance_model,
                left=np.nan,
                right=np.nan
            )
            
            # Normalize to data
            scale = np.sum(self.jwst_flux) / np.sum(model_flux) if np.sum(model_flux) > 0 else 1.0
            model_flux *= scale
            
            # Compute chi-squared
            residuals = (self.jwst_flux - model_flux) / self.jwst_uncertainty
            chi_squared = np.sum(residuals ** 2)
            
            # Track history
            self.iteration_count += 1
            self.history['thermal_inertia'].append(thermal_inertia)
            self.history['chi_squared'].append(chi_squared)
            self.history['radiance'].append(radiance_model)
            
            if self.iteration_count % 10 == 0:
                logger.info(f"Iteration {self.iteration_count}: TI={thermal_inertia:.1f}, χ²={chi_squared:.2f}")
            
            return chi_squared
            
        except Exception as e:
            logger.error(f"Error evaluating TI={thermal_inertia:.1f}: {e}")
            return np.inf
    
    def fit_local(
        self,
        thermal_inertia_initial: float = 300.0,
        bounds: tuple = (1.0, 10000.0),
        method: str = 'Nelder-Mead',
        max_iterations: int = 500,
        tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Local optimization using scipy.optimize.minimize.
        
        Args:
            thermal_inertia_initial: Starting value [W m^-2 K^-1 s^-0.5]
            bounds: (min, max) bounds for TI
            method: Optimization method ('Nelder-Mead', 'L-BFGS-B', etc.)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with keys:
            - 'thermal_inertia_best': Best-fit TI
            - 'chi_squared_best': Best χ²
            - 'result': scipy OptimizeResult object
            - 'success': Whether optimization converged
        """
        logger.info(f"Starting local optimization (method={method})")
        logger.info(f"Initial TI: {thermal_inertia_initial:.1f}")
        logger.info(f"Bounds: {bounds}")
        
        # Initial chi-squared
        chi2_initial = self.cost_function(thermal_inertia_initial)
        logger.info(f"Initial χ²: {chi2_initial:.2f}")
        
        # Bounds as scipy expects
        if method == 'L-BFGS-B':
            scipy_bounds = [bounds]
        else:
            scipy_bounds = None
        
        # Run optimization
        result = minimize(
            self.cost_function,
            x0=[thermal_inertia_initial],
            method=method,
            bounds=scipy_bounds,
            options={
                'maxiter': max_iterations,
                'xatol': tolerance,
                'fatol': tolerance,
            },
            callback=None,
        )
        
        best_ti = float(result.x[0]) if hasattr(result.x, '__iter__') else float(result.x)
        best_chi2 = float(result.fun)
        
        logger.info(f"Local optimization complete: TI={best_ti:.1f}, χ²={best_chi2:.2f}")
        
        return {
            'thermal_inertia_best': best_ti,
            'chi_squared_best': best_chi2,
            'result': result,
            'success': result.success,
        }
    
    def fit_global(
        self,
        bounds: tuple = (1.0, 10000.0),
        population_size: int = 30,
        max_generations: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Global optimization using scipy.optimize.differential_evolution.
        
        Robust against local minima but computationally expensive.
        
        Args:
            bounds: (min, max) bounds for TI
            population_size: Population size for evolution algorithm
            max_generations: Maximum generations
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with keys:
            - 'thermal_inertia_best': Best-fit TI
            - 'chi_squared_best': Best χ²
            - 'result': scipy OptimizeResult object
            - 'success': Whether optimization converged
        """
        logger.info(f"Starting global optimization (differential_evolution)")
        logger.info(f"Population size: {population_size}, Max generations: {max_generations}")
        logger.info(f"Bounds: {bounds}")
        
        # Bounds as scipy expects
        scipy_bounds = [bounds]
        
        # Run optimization
        result = differential_evolution(
            self.cost_function,
            bounds=scipy_bounds,
            maxiter=max_generations,
            popsize=population_size,
            seed=seed,
            workers=1,  # Sequential evaluation
            updating='deferred',
            atol=0,
            tol=1e-6,
        )
        
        best_ti = float(result.x[0]) if hasattr(result.x, '__iter__') else float(result.x)
        best_chi2 = float(result.fun)
        
        logger.info(f"Global optimization complete: TI={best_ti:.1f}, χ²={best_chi2:.2f}")
        
        return {
            'thermal_inertia_best': best_ti,
            'chi_squared_best': best_chi2,
            'result': result,
            'success': result.success,
        }
    
    def save_history(self, output_path: Path):
        """Save optimization history to HDF5 file."""
        import h5py
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('thermal_inertia', data=np.array(self.history['thermal_inertia']))
            f.create_dataset('chi_squared', data=np.array(self.history['chi_squared']))
            # Radiance may have different lengths, so store as ragged array
            grp_rad = f.create_group('radiance')
            for i, rad in enumerate(self.history['radiance']):
                grp_rad.create_dataset(f"iteration_{i}", data=rad)
        
        logger.info(f"Saved optimization history to {output_path}")
    
    def plot_convergence(self, output_path: Optional[Path] = None):
        """Plot optimization convergence."""
        import matplotlib.pyplot as plt
        
        if len(self.history['chi_squared']) == 0:
            logger.warning("No optimization history to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        iterations = np.arange(len(self.history['chi_squared']))
        
        # Chi-squared vs iteration
        axes[0].semilogy(iterations, self.history['chi_squared'], 'b.-', linewidth=1)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Chi-squared')
        axes[0].set_title('Convergence: Chi-squared vs Iteration')
        axes[0].grid(True, alpha=0.3)
        
        # Thermal inertia vs iteration
        axes[1].semilogx(iterations, self.history['thermal_inertia'], 'r.-', linewidth=1)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Thermal Inertia [W m⁻² K⁻¹ s⁻⁰·⁵]')
        axes[1].set_title('Convergence: Thermal Inertia vs Iteration')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved convergence plot to {output_path}")
        
        return fig, axes
