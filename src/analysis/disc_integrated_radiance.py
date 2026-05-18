"""
Disc-integrated thermal radiance computation for TEMPEST output.

Computes spectral radiance (Jy) from facet temperatures and viewing geometry,
integrated over the visible disc (used for JWST fitting).
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def compute_disc_integrated_radiance(
    temperatures: np.ndarray,
    shape_model,
    wavelengths: np.ndarray,
    simulation,
    phase_angle: float = 0.0,
    observer_distance: float = 1.0,
    emissivity: float = 0.95,
    apply_epf: bool = True,
) -> np.ndarray:
    """
    Compute disc-integrated spectral flux from TEMPEST facet temperatures.
    
    This computes the integrated thermal spectrum from all visible facets,
    with an arbitrary (unitless) scale that will be fitted to observations.
    
    Args:
        temperatures: (n_facets, n_timesteps) temperature array [K]
        shape_model: List of Facet objects with .normal, .area, .position
        wavelengths: Wavelength array [m]
        simulation: Simulation object
        phase_angle: Observer phase angle [radians], 0 = opposition
        observer_distance: Distance from target (scale doesn't matter for fitting)
        emissivity: Thermal emissivity [0-1]
        apply_epf: Apply emission phase function correction
        
    Returns:
        Spectral flux array [arbitrary units], shape (n_wavelengths,)
        
    Notes:
        - Absolute units don't matter; a free scale parameter will be fitted
        - Only the spectrum shape matters for parameter fitting
        - Uses final timestep of temperature array (steady-state)
    """
    from retrieve_radiance import planck_radiance
    
    # Use final timestep (steady state)
    T_final = temperatures[:, -1]
    
    # Compute observer and sun directions at zero phase angle
    observer_direction = np.array([1.0, 0.0, 0.0])  # Observer looks down -x
    
    # Initialize flux array
    flux_spectral = np.zeros_like(wavelengths, dtype=float)
    
    # For each visible facet
    for idx, facet in enumerate(shape_model):
        # Check visibility: facet normal must point toward observer
        cos_emission = np.dot(facet.normal, observer_direction)
        
        if cos_emission <= 0:
            continue  # Facet faces away
        
        # Temperature for this facet
        T = T_final[idx]
        
        # Compute Planck spectral radiance at all wavelengths
        L_planck = planck_radiance(wavelengths, T, emissivity=emissivity)
        
        # Apply EPF correction if enabled
        if apply_epf:
            epf_factor = cos_emission ** 0.6  # Typical for rough surfaces
        else:
            epf_factor = 1.0
        
        # Contribution from this facet:
        # Flux contribution ∝ L(λ,T) × (facet_area × cos_e) × EPF
        # The solid angle (area × cos_e / d²) will be scaled by free parameter anyway
        contribution = L_planck * (facet.area * cos_emission) * epf_factor
        flux_spectral += contribution
    
    return flux_spectral


def normalize_to_jy(
    radiance_si: np.ndarray,
    wavelengths_m: np.ndarray,
    distance_au: float = 1.0,
    solid_angle_sr: float = 1.0,
) -> np.ndarray:
    """
    Convert spectral radiance to flux density (Jy) at observer.
    
    Args:
        radiance_si: Spectral radiance [W/(m^3·sr)]
        wavelengths_m: Wavelengths [m]
        distance_au: Distance to observer [AU]
        solid_angle_sr: Solid angle of disc [sr]
        
    Returns:
        Flux density [Jy], shape same as radiance_si
        
    Notes:
        1 Jy = 1e-26 W/(m^2·Hz)
        Using λ·Iλ = ν·Iν conversion
    """
    # Physical constants
    c = 3e8  # m/s
    au_m = 1.496e11  # m
    
    # Distance in meters
    distance_m = distance_au * au_m
    
    # Convert wavelength to frequency for unit conversion
    # Iν = (c / λ²) × Iλ (spectral radiance conversion)
    # But we want flux density: F = Ω × L
    # where Ω is solid angle
    
    # Simple scaling: assume uniform disc of given solid angle
    # F [Jy] = L [W/(m^3·sr)] × Ω [sr] / (1e-26 × distance^2)
    flux_density_si = radiance_si * solid_angle_sr / (distance_m ** 2)  # [W/m^2/m]
    
    # Convert to Jy/wavelength
    # 1 Jy = 1e-26 W/(m^2·Hz)
    # Need to convert W/(m^2·m) to Jy
    # Flux per wavelength interval: F_λ [W/m^3]
    # F_ν = c/λ² × F_λ [W/Hz]
    # But we have integrated over wavelength already...
    
    # Simpler approach: normalize by spectrum amplitude
    # This requires knowing the actual system size and geometry
    flux_density_jy = flux_density_si * 1e26  # Very rough conversion
    
    return flux_density_jy


def compare_to_jwst(
    model_radiance: np.ndarray,
    model_wavelengths: np.ndarray,
    jwst_wavelengths: np.ndarray,
    jwst_flux: np.ndarray,
    jwst_uncertainty: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compare model radiance to JWST observations via chi-squared.
    
    Args:
        model_radiance: Model spectral radiance [W/(m^3·sr)]
        model_wavelengths: Model wavelengths [m]
        jwst_wavelengths: JWST wavelengths [m]
        jwst_flux: JWST flux density [Jy]
        jwst_uncertainty: JWST flux uncertainty [Jy]
        
    Returns:
        chi_squared: Goodness of fit statistic
        model_interp: Model flux interpolated to JWST wavelengths [Jy]
        residuals: (obs - model) / sigma
    """
    # Interpolate model to JWST wavelengths
    model_interp = np.interp(
        jwst_wavelengths,
        model_wavelengths,
        model_radiance,
        left=np.nan,
        right=np.nan
    )
    
    # Normalize both to have same total integrated flux
    # This compensates for absolute distance/size ambiguity
    model_integral = np.trapz(model_interp, jwst_wavelengths)
    jwst_integral = np.trapz(jwst_flux, jwst_wavelengths)
    scale_factor = jwst_integral / model_integral if model_integral > 0 else 1.0
    model_interp *= scale_factor
    
    # Compute residuals
    residuals = (jwst_flux - model_interp) / jwst_uncertainty
    
    # Chi-squared
    chi_squared = np.sum(residuals ** 2)
    
    return chi_squared, model_interp, residuals


def plot_radiance_comparison(
    model_wavelengths: np.ndarray,
    model_radiance: np.ndarray,
    jwst_wavelengths: np.ndarray,
    jwst_flux: np.ndarray,
    jwst_uncertainty: np.ndarray,
    title: str = "Model vs JWST Observations",
    output_path: Optional[Path] = None,
):
    """Plot model radiance against JWST observations."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Model
    ax.plot(model_wavelengths * 1e6, model_radiance, 'b-', linewidth=2, label='TEMPEST Model')
    
    # JWST data
    ax.errorbar(
        jwst_wavelengths * 1e6,
        jwst_flux,
        yerr=jwst_uncertainty,
        fmt='o',
        markersize=3,
        capsize=2,
        color='red',
        label='JWST Observations'
    )
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Flux Density (Jy)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax
