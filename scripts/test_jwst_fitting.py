#!/usr/bin/env python3
"""
Test and exploration script for JWST thermal fitting with TEMPEST.

This script:
1. Loads and plots JWST mid-IR observations
2. Computes disc-integrated thermal radiance from TEMPEST output
3. Tests sensitivity to thermal inertia
4. Prepares for fitting
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add TEMPEST to path
tempest_root = Path(__file__).parent.parent
sys.path.insert(0, str(tempest_root))

def load_jwst_data(filepath):
    """Load JWST MIRI spectroscopy data.
    
    Returns:
        wavelength (μm), flux (Jy), uncertainty (Jy)
    """
    data = np.loadtxt(filepath)
    wavelength = data[:, 0]
    flux = data[:, 1]
    uncertainty = data[:, 2]
    
    print(f"Loaded JWST data:")
    print(f"  Wavelength range: {wavelength.min():.2f}-{wavelength.max():.2f} μm")
    print(f"  Flux range: {flux.min():.1f}-{flux.max():.1f} Jy")
    print(f"  Uncertainty range: {uncertainty.min():.2f}-{uncertainty.max():.2f} Jy ({(uncertainty/flux*100).mean():.1f}%)")
    
    return wavelength, flux, uncertainty


def test_1_plot_jwst_data():
    """Test 1: Load and visualize JWST data."""
    print("\n" + "="*60)
    print("TEST 1: Load and plot JWST observations")
    print("="*60)
    
    jwst_file = tempest_root / 'private/data/JWST/eurybates_miri_psf3_3_3.dat'
    wavelength, flux, uncertainty = load_jwst_data(jwst_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Flux
    axes[0].plot(wavelength, flux, 'o-', markersize=2, linewidth=0.5)
    axes[0].fill_between(wavelength, flux - uncertainty, flux + uncertainty, alpha=0.3)
    axes[0].set_xlabel('Wavelength (μm)')
    axes[0].set_ylabel('Flux density (Jy)')
    axes[0].set_title('JWST MIRI Spectrum (with 1σ errors)')
    axes[0].grid(True, alpha=0.3)
    
    # Uncertainty
    axes[1].plot(wavelength, uncertainty / flux * 100, 'o', markersize=2, color='orange')
    axes[1].set_xlabel('Wavelength (μm)')
    axes[1].set_ylabel('Fractional uncertainty (%)')
    axes[1].set_title('Relative Error')
    axes[1].grid(True, alpha=0.3)
    
    # S/N
    axes[2].plot(wavelength, flux / uncertainty, 'o', markersize=2, color='red')
    axes[2].set_xlabel('Wavelength (μm)')
    axes[2].set_ylabel('Signal-to-noise ratio')
    axes[2].set_title('S/N Ratio')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(tempest_root / 'outputs/test_1_jwst_data.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/test_1_jwst_data.png")
    
    return wavelength, flux, uncertainty


def test_2_planck_radiance():
    """Test 2: Compute Planck radiance at JWST wavelengths."""
    print("\n" + "="*60)
    print("TEST 2: Planck radiance calculation")
    print("="*60)
    
    from retrieve_radiance import planck_radiance
    
    wavelength, flux, uncertainty = load_jwst_data(
        tempest_root / 'private/data/JWST/eurybates_miri_psf3_3_3.dat'
    )
    
    # Test temperatures (K)
    temperatures = [150, 200, 250, 300, 350]
    emissivity = 0.95  # C-type asteroid
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for T in temperatures:
        # Convert wavelength from μm to m for planck_radiance
        radiance = planck_radiance(wavelength * 1e-6, T, emissivity=emissivity)
        # Planck radiance is in W/(m^3·sr), normalize to plot relative shape
        radiance_norm = radiance / radiance.max()
        
        ax.plot(wavelength, radiance_norm, label=f'T={T} K', linewidth=2)
    
    # Plot JWST data normalized
    flux_norm = flux / flux.max()
    ax.errorbar(wavelength, flux_norm, yerr=uncertainty/flux.max(), 
                fmt='o', markersize=2, capsize=2, label='JWST (normalized)', color='black')
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Normalized radiance')
    ax.set_title('Planck Function vs JWST Observations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(tempest_root / 'outputs/test_2_planck_radiance.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/test_2_planck_radiance.png")
    print(f"Temperature range for radiance test: {temperatures[0]}-{temperatures[-1]} K")


def test_3_simple_model():
    """Test 3: Simple isothermal model comparison."""
    print("\n" + "="*60)
    print("TEST 3: Simple isothermal blackbody model")
    print("="*60)
    
    from retrieve_radiance import planck_radiance
    
    wavelength, flux, uncertainty = load_jwst_data(
        tempest_root / 'private/data/JWST/eurybates_miri_psf3_3_3.dat'
    )
    
    # Simple fitting: isothermal blackbody
    # Chi-squared minimization for single temperature
    def chi_squared(T, emissivity=0.95):
        radiance = planck_radiance(wavelength * 1e-6, T, emissivity=emissivity)
        # Normalize to JWST flux level
        scale = np.sum(flux) / np.sum(radiance)
        model = radiance * scale
        residuals = (flux - model) / uncertainty
        return np.sum(residuals**2)
    
    # Grid search for best temperature
    T_test = np.linspace(100, 400, 100)
    chi2_values = [chi_squared(T) for T in T_test]
    best_T = T_test[np.argmin(chi2_values)]
    
    print(f"Best-fit isothermal temperature: {best_T:.1f} K")
    print(f"Chi-squared at best fit: {min(chi2_values):.2f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chi-squared curve
    axes[0].plot(T_test, chi2_values, 'b-', linewidth=2)
    axes[0].axvline(best_T, color='r', linestyle='--', label=f'Best T = {best_T:.0f} K')
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('χ²')
    axes[0].set_title('Chi-squared vs Temperature')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Data vs model
    radiance_best = planck_radiance(wavelength * 1e-6, best_T, emissivity=0.95)
    scale = np.sum(flux) / np.sum(radiance_best)
    model_best = radiance_best * scale
    
    axes[1].errorbar(wavelength, flux, yerr=uncertainty, fmt='o', markersize=2, 
                     label='JWST observations', capsize=2)
    axes[1].plot(wavelength, model_best, 'r-', linewidth=2, label=f'Model T={best_T:.0f} K')
    axes[1].set_xlabel('Wavelength (μm)')
    axes[1].set_ylabel('Flux density (Jy)')
    axes[1].set_title('Best-fit Isothermal Model')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(tempest_root / 'outputs/test_3_simple_model.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/test_3_simple_model.png")


def main():
    """Run all tests."""
    import os
    
    # Create outputs directory
    output_dir = tempest_root / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("JWST THERMAL FITTING - TEST SUITE")
    print("="*60)
    
    try:
        test_1_plot_jwst_data()
        test_2_planck_radiance()
        test_3_simple_model()
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Inspect outputs in /outputs/")
        print("2. Run TEMPEST with thermal_inertia sweep")
        print("3. Compute disc-integrated radiance for each run")
        print("4. Build scipy.optimize fitting loop")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
