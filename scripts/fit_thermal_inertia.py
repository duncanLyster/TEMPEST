#!/usr/bin/env python3
"""
Main script for fitting TEMPEST thermal inertia to JWST observations.

Usage:
    python scripts/fit_thermal_inertia.py --config CONFIG.yaml --jwst JWST_DATA.dat
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add TEMPEST root to path
tempest_root = Path(__file__).parent.parent
sys.path.insert(0, str(tempest_root))


def main(args):
    """Run thermal inertia fitting."""
    
    # Load JWST data
    logger.info("="*60)
    logger.info("TEMPEST JWST THERMAL INERTIA FITTING")
    logger.info("="*60)
    
    from src.observations.jwst_loader import load_jwst_spectrum, print_jwst_summary
    
    logger.info(f"Loading JWST data from: {args.jwst}")
    wavelength_um, flux, uncertainty = load_jwst_spectrum(args.jwst)
    wavelength_m = wavelength_um * 1e-6  # Convert to meters
    
    print_jwst_summary(wavelength_um, flux, uncertainty)
    
    # Initialize forward model
    logger.info(f"\nInitializing forward model with config: {args.config}")
    
    if args.use_isothermal:
        logger.info("Using simplified isothermal forward model")
        from src.analysis.forward_model import forward_model_isothermal
        forward_model = lambda ti: forward_model_isothermal(ti, wavelength_m)
    else:
        logger.info("Using full TEMPEST forward model (experimental)")
        from src.analysis.forward_model import ForwardModel
        fm = ForwardModel(args.config, wavelength_m, phase_angle=0.0)
        forward_model = fm.evaluate
    
    # Initialize fitter
    logger.info("\nInitializing optimizer")
    from src.analysis.fitting import ThermalInertiaFitter
    
    fitter = ThermalInertiaFitter(
        forward_model=forward_model,
        jwst_wavelengths=wavelength_m,
        jwst_flux=flux,
        jwst_uncertainty=uncertainty,
        emissivity=args.emissivity,
    )
    
    # Run optimization
    logger.info("\n" + "="*60)
    if args.method == 'global':
        logger.info("Running GLOBAL optimization (differential evolution)")
        result = fitter.fit_global(
            bounds=args.bounds,
            population_size=args.population_size,
            max_generations=args.max_generations,
            seed=42,
        )
    else:
        logger.info("Running LOCAL optimization (Nelder-Mead)")
        result = fitter.fit_local(
            thermal_inertia_initial=args.ti_initial,
            bounds=args.bounds,
            method=args.local_method,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
        )
    
    logger.info("="*60)
    
    # Report results
    best_ti = result['thermal_inertia_best']
    best_chi2 = result['chi_squared_best']
    success = result['success']
    
    logger.info(f"\nOPTIMIZATION RESULTS:")
    logger.info(f"  Best-fit thermal inertia: {best_ti:.2f} W m⁻² K⁻¹ s⁻⁰·⁵")
    logger.info(f"  Minimum χ²: {best_chi2:.2f}")
    logger.info(f"  Convergence: {'SUCCESS' if success else 'NO CONVERGENCE'}")
    logger.info(f"  Iterations: {len(fitter.history['chi_squared'])}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save history
    history_file = output_dir / 'optimization_history.h5'
    fitter.save_history(history_file)
    
    # Save convergence plot
    convergence_file = output_dir / 'convergence.png'
    fitter.plot_convergence(convergence_file)
    
    # Save results to text file
    results_file = output_dir / 'results.txt'
    with open(results_file, 'w') as f:
        f.write("THERMAL INERTIA FITTING RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best-fit thermal inertia: {best_ti:.2f} W m⁻² K⁻¹ s⁻⁰·⁵\n")
        f.write(f"Minimum χ²: {best_chi2:.2f}\n")
        f.write(f"Convergence: {'SUCCESS' if success else 'NO CONVERGENCE'}\n")
        f.write(f"Total iterations: {len(fitter.history['chi_squared'])}\n")
        f.write(f"Emissivity: {args.emissivity}\n")
        f.write(f"\nJWST Data:\n")
        f.write(f"  File: {args.jwst}\n")
        f.write(f"  Wavelength range: {wavelength_um.min():.2f}-{wavelength_um.max():.2f} μm\n")
        f.write(f"  Number of points: {len(wavelength_um)}\n")
    
    logger.info(f"\nResults saved to: {output_dir}/")
    logger.info(f"  - optimization_history.h5")
    logger.info(f"  - convergence.png")
    logger.info(f"  - results.txt")
    
    return result


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit TEMPEST thermal inertia to JWST observations"
    )
    
    # Required arguments
    parser.add_argument(
        '--jwst',
        required=True,
        help='Path to JWST spectroscopic data file (3-column ASCII)'
    )
    parser.add_argument(
        '--config',
        required=False,
        help='Path to TEMPEST configuration YAML (required if not using --isothermal)'
    )
    
    # Forward model options
    parser.add_argument(
        '--isothermal',
        dest='use_isothermal',
        action='store_true',
        default=True,
        help='Use simplified isothermal forward model (default)'
    )
    parser.add_argument(
        '--no-isothermal',
        dest='use_isothermal',
        action='store_false',
        help='Use full TEMPEST forward model (experimental, slower)'
    )
    
    # Optimization options
    parser.add_argument(
        '--method',
        choices=['local', 'global'],
        default='local',
        help='Optimization method (default: local)'
    )
    parser.add_argument(
        '--local-method',
        default='Nelder-Mead',
        choices=['Nelder-Mead', 'L-BFGS-B', 'Powell'],
        help='Local optimization algorithm (default: Nelder-Mead)'
    )
    parser.add_argument(
        '--ti-initial',
        type=float,
        default=300.0,
        help='Initial thermal inertia guess (default: 300)'
    )
    parser.add_argument(
        '--bounds',
        type=float,
        nargs=2,
        default=(1.0, 10000.0),
        help='Thermal inertia bounds [min max] (default: 1 10000)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=500,
        help='Maximum iterations for local optimization (default: 500)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='Convergence tolerance (default: 1e-6)'
    )
    
    # Global optimization options
    parser.add_argument(
        '--population-size',
        type=int,
        default=30,
        help='Population size for differential evolution (default: 30)'
    )
    parser.add_argument(
        '--max-generations',
        type=int,
        default=100,
        help='Maximum generations for differential evolution (default: 100)'
    )
    
    # Physics options
    parser.add_argument(
        '--emissivity',
        type=float,
        default=0.95,
        help='Thermal emissivity (default: 0.95 for C-type asteroids)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        default='outputs/jwst_fitting',
        help='Output directory for results (default: outputs/jwst_fitting)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    try:
        result = main(args)
        sys.exit(0 if result['success'] else 1)
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
