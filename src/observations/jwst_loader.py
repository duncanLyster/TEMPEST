"""
JWST spectroscopic data loading and validation.

Loads MIRI spectroscopic observations and validates data integrity.
"""

import numpy as np
from pathlib import Path
from typing import Tuple


def load_jwst_spectrum(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load JWST MIRI spectroscopic data.
    
    Expected format: 3-column ASCII file
    Column 1: Wavelength (μm)
    Column 2: Flux density (Jy)
    Column 3: Flux uncertainty (Jy)
    
    Args:
        filepath: Path to JWST data file
        
    Returns:
        wavelength (μm), flux (Jy), uncertainty (Jy) as numpy arrays
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data has wrong format or invalid values
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JWST data file not found: {filepath}")
    
    # Load data
    try:
        data = np.loadtxt(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load JWST data file: {e}")
    
    # Validate shape
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected 3-column array, got shape {data.shape}")
    
    wavelength = data[:, 0]
    flux = data[:, 1]
    uncertainty = data[:, 2]
    
    # Sort by wavelength (handles multiple bands stitched together)
    sort_idx = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    flux = flux[sort_idx]
    uncertainty = uncertainty[sort_idx]
    
    if wavelength.min() < 1 or wavelength.max() > 100:
        raise ValueError(f"Wavelengths out of expected range: {wavelength.min():.2f}-{wavelength.max():.2f} μm")
    
    # Validate flux and uncertainty
    if not np.all(flux >= 0):
        raise ValueError(f"Negative flux values detected (min={flux.min():.2e})")
    
    if not np.all(uncertainty >= 0):
        raise ValueError(f"Negative uncertainty values detected (min={uncertainty.min():.2e})")
    
    # Check for NaN/inf
    if not np.all(np.isfinite(wavelength)):
        raise ValueError("Non-finite values in wavelength")
    if not np.all(np.isfinite(flux)):
        raise ValueError("Non-finite values in flux")
    if not np.all(np.isfinite(uncertainty)):
        raise ValueError("Non-finite values in uncertainty")
    
    return wavelength, flux, uncertainty


def print_jwst_summary(wavelength: np.ndarray, flux: np.ndarray, uncertainty: np.ndarray):
    """Print summary statistics of JWST spectrum.
    
    Args:
        wavelength: Wavelength array (μm)
        flux: Flux density array (Jy)
        uncertainty: Flux uncertainty array (Jy)
    """
    frac_error = 100 * uncertainty / flux
    sn_ratio = flux / uncertainty
    
    print("\nJWST MIRI Spectrum Summary:")
    print(f"  Wavelength range: {wavelength.min():.2f}-{wavelength.max():.2f} μm")
    print(f"  Number of points: {len(wavelength)}")
    print(f"  Sampling: {np.mean(np.diff(wavelength))*1e3:.4f} nm/step")
    print(f"\nFlux statistics (Jy):")
    print(f"  Min/max: {flux.min():.2f} / {flux.max():.2f}")
    print(f"  Median: {np.median(flux):.2f}")
    print(f"  Mean: {flux.mean():.2f}")
    print(f"\nUncertainty statistics:")
    print(f"  Relative error: {frac_error.mean():.2f}% ± {frac_error.std():.2f}%")
    print(f"  S/N ratio: {sn_ratio.mean():.1f} ± {sn_ratio.std():.1f}")
    print(f"  Range: {frac_error.min():.2f}% - {frac_error.max():.2f}%")
