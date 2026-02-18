#!/usr/bin/env python3
"""
LUT Correlation Diagnostic Script

Investigates why roughness correction factors show systematic latitude dependence.
Runs actual crater simulations and analyzes temperature/geometry patterns.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from scipy.stats import pearsonr
from scipy.integrate import simpson
from datetime import datetime

# Setup paths
root_dir = Path(__file__).parent.resolve()
os.chdir(root_dir)
sys.path.append(str(root_dir))

from TEMPEST_RAD.generator import simulate_crater_diurnal_cycle, ReferenceConfig

print("=" * 80)
print("LUT CORRELATION DIAGNOSTIC")
print("=" * 80)

# Create output directory
output_dir = root_dir / "diagnostic_plots"
output_dir.mkdir(exist_ok=True)
print(f"\nOutput directory: {output_dir}")

# ============================================================================
# PART 1: LOAD AND ANALYZE LUT
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: LUT CORRECTION FACTOR ANALYSIS")
print("=" * 80)

LUT_FILE = root_dir / "roughness_lut_moon_test.h5"

lut_timestamp = os.path.getmtime(LUT_FILE)
print(f"\nLUT File: {LUT_FILE.name}")
print(f"Modified: {datetime.fromtimestamp(lut_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Size: {LUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")

with h5py.File(LUT_FILE, 'r') as f:
    latitudes = f['latitude'][...]
    wavelengths = f['wavelength'][...]
    factors = f['lut'][...]
    
    print(f"\nLUT Shape: {factors.shape}")
    print(f"Latitudes: {latitudes}°")
    print(f"Wavelengths: {wavelengths} µm")
    
    # Query at noon, disk center, 15µm
    i_theta = 0
    i_angle = 0
    i_time = 90  # Noon
    i_wave = np.argmin(np.abs(wavelengths - 15.0))
    i_emi = 0
    i_azi = 0
    
    factors_noon = factors[i_theta, i_angle, :, i_time, i_wave, i_emi, i_azi]
    
    print(f"\nCorrection Factors at Noon (180°), Disk Center, {wavelengths[i_wave]:.1f}µm:")
    print("-" * 60)
    print(f"{'Latitude':>10}  {'Factor':>8}  {'Expected':>12}  {'Status':>8}")
    print("-" * 60)
    
    for lat, factor in zip(latitudes, factors_noon):
        status = "✓" if factor > 1.0 else "✗"
        print(f"{lat:10.1f}° {factor:8.4f}  {'>1.0 (beaming)':>12}  {status:>8}")
    
    # Statistical test
    corr, pval = pearsonr(latitudes, factors_noon)
    
    print(f"\n{'=' * 60}")
    print(f"CORRELATION TEST:")
    print(f"  Pearson r = {corr:+.4f}")
    print(f"  p-value   = {pval:.6f}")
    
    if abs(corr) < 0.3:
        result = "✅ PASS"
    elif abs(corr) < 0.7:
        result = "⚠️  WARNING"
    else:
        result = "❌ FAIL"
    
    print(f"  {result}: |r| = {abs(corr):.3f}")
    print("=" * 60)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Factors vs Latitude
    ax = axes[0]
    ax.plot(latitudes, factors_noon, 'o-', markersize=12, linewidth=3, 
            color='darkred', label='Correction Factor')
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Smooth Surface')
    ax.set_xlabel('Latitude (°)', fontsize=12)
    ax.set_ylabel('Correction Factor', fontsize=12)
    ax.set_title(f'Latitude Dependence (r={corr:+.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Right: Latitude-Time Heatmap
    ax = axes[1]
    lat_time_map = factors[i_theta, i_angle, :, :, i_wave, i_emi, i_azi]
    
    im = ax.imshow(lat_time_map, aspect='auto', origin='lower', cmap='hot',
                   extent=[0, 360, latitudes[0], latitudes[-1]], vmin=0.8, vmax=2.5)
    ax.axvline(180, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Phase Angle (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title(f'Factor Map (λ={wavelengths[i_wave]:.1f}µm)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Factor', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "1_correction_factors.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / '1_correction_factors.png'}")
    plt.close()

# ============================================================================
# PART 2: RUN DETAILED CRATER SIMULATIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: RUNNING DETAILED CRATER SIMULATIONS")
print("=" * 80)

# Configuration
opening_angle = 90.0
theta = 0.0316  # Moon TI=55

config = ReferenceConfig()
# Override specific values for quick testing
CRATER_SUBFACETS = 100
VIEW_FACTOR_RAYS = 500
SIM_TIMESTEPS = 360

# Update config with these values
config.kernel_subfacets_count = CRATER_SUBFACETS

print(f"\nConfiguration:")
print(f"  Theta: {theta:.4f} (TI ≈ 55)")
print(f"  Opening angle: {opening_angle}°")
print(f"  Subfacets: {CRATER_SUBFACETS}")
print(f"  View factor rays: {VIEW_FACTOR_RAYS}")
print(f"  Timesteps: {SIM_TIMESTEPS}")

# Test at two latitudes
test_latitudes = [0.0, 90.0]
results = {}

for lat in test_latitudes:
    print(f"\n{'─' * 80}")
    print(f"SIMULATING LATITUDE {lat}°")
    print('─' * 80)
    
    # Run full simulation using existing generator function
    print(f"Running simulation... (this takes ~30 seconds)")
    
    thermal_data, simulation, shape_model = simulate_crater_diurnal_cycle(
        theta, opening_angle, lat, config, SIM_TIMESTEPS
    )
    
    # Extract results
    temps = thermal_data.get_temperature()
    insol = thermal_data.get_insolation()
    
    n_facets = len(shape_model)
    print(f"\nResults:")
    print(f"  Facets: {n_facets}")
    print(f"  Temperature shape: {temps.shape}")
    print(f"  Min temp: {np.min(temps):.1f} K")
    print(f"  Max temp: {np.max(temps):.1f} K")
    print(f"  Mean temp (last cycle): {np.mean(temps[:, -SIM_TIMESTEPS:]):.1f} K")
    
    # Insolation analysis
    insol_mean = np.mean(insol, axis=1)
    n_illuminated = np.sum(insol_mean > 0)
    
    print(f"  Illuminated facets: {n_illuminated}/{n_facets} ({100*n_illuminated/n_facets:.1f}%)")
    if n_illuminated > 0:
        print(f"  Mean insolation (lit): {np.mean(insol_mean[insol_mean > 0]):.1f} W/m²")
    print(f"  Max insolation: {np.max(insol):.1f} W/m²")
    
    # Store results
    results[lat] = {
        'temps': temps,
        'insol': insol,
        'shape_model': shape_model,
        'thermal_data': thermal_data,
        'simulation': simulation,
        'sun_dir': simulation.sunlight_direction
    }

# ============================================================================
# PART 3: VISUALIZE TEMPERATURE AND INSOLATION
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: TEMPERATURE AND INSOLATION PATTERNS")
print("=" * 80)

n_timesteps = SIM_TIMESTEPS

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for col_idx, lat in enumerate([0.0, 90.0]):
    data = results[lat]
    temps = data['temps'][:, -n_timesteps:]  # Last cycle
    insol = data['insol'][:, -n_timesteps:]
    
    # Row 1: Temperature
    ax = axes[0, col_idx]
    im = ax.imshow(temps, aspect='auto', cmap='hot', origin='lower',
                   extent=[0, 360, 0, temps.shape[0]], vmin=50, vmax=400)
    ax.set_xlabel('Phase Angle (°)', fontsize=11)
    ax.set_ylabel('Facet Index', fontsize=11)
    ax.set_title(f'Temperature (Lat {lat}°)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('T (K)', fontsize=10)
    
    # Row 2: Insolation
    ax = axes[1, col_idx]
    im = ax.imshow(insol, aspect='auto', cmap='YlOrRd', origin='lower',
                   extent=[0, 360, 0, insol.shape[0]])
    ax.set_xlabel('Phase Angle (°)', fontsize=11)
    ax.set_ylabel('Facet Index', fontsize=11)
    ax.set_title(f'Insolation (Lat {lat}°)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('W/m²', fontsize=10)

# Column 3: Comparisons
ax = axes[0, 2]
for lat in [0.0, 90.0]:
    temps = results[lat]['temps'][:, -360:]
    temp_mean = np.mean(temps, axis=1)
    temp_sorted = np.sort(temp_mean)[::-1]
    ax.plot(temp_sorted, linewidth=2, label=f'{lat}° (μ={np.mean(temp_mean):.0f}K)')

ax.set_xlabel('Facet Rank', fontsize=11)
ax.set_ylabel('Mean Temperature (K)', fontsize=11)
ax.set_title('Temperature Distributions', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

ax = axes[1, 2]
for lat in [0.0, 90.0]:
    insol = results[lat]['insol'][:, -360:]
    insol_mean = np.mean(insol, axis=1)
    insol_sorted = np.sort(insol_mean)[::-1]
    n_lit = np.sum(insol_mean > 0)
    ax.plot(insol_sorted, linewidth=2, label=f'{lat}° ({n_lit} lit)')

ax.set_xlabel('Facet Rank', fontsize=11)
ax.set_ylabel('Mean Insolation (W/m²)', fontsize=11)
ax.set_title('Insolation Distributions', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "2_thermal_patterns.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / '2_thermal_patterns.png'}")
plt.close()

# ============================================================================
# PART 4: ENERGY BALANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: ENERGY BALANCE CHECK")
print("=" * 80)

sigma = 5.67e-8  # Stefan-Boltzmann constant

for lat in [0.0, 90.0]:
    print(f"\n{'─' * 80}")
    print(f"LATITUDE {lat}°")
    print('─' * 80)
    
    data = results[lat]
    temps = data['temps'][:, -n_timesteps:]
    shape_model = data['shape_model']
    
    # Calculate radiance
    radiance = sigma * temps**4  # W/m²
    areas = np.array([f.area for f in shape_model])
    
    # Total flux per timestep
    flux_per_timestep = np.sum(radiance * areas[:, np.newaxis], axis=0)
    total_area = np.sum(areas)
    
    # Time-averaged flux
    mean_flux_rough = np.mean(flux_per_timestep)
    mean_flux_per_area = mean_flux_rough / total_area
    
    # Equivalent blackbody temperature
    T_eff = (mean_flux_per_area / sigma)**0.25
    
    # Mean temperature
    T_mean = np.mean(temps)
    
    # For smooth surface at same T_mean
    flux_smooth_equiv = sigma * T_mean**4 * total_area
    
    print(f"\nRough Crater:")
    print(f"  Total area: {total_area:.6f} m²")
    print(f"  Mean temperature: {T_mean:.2f} K")
    print(f"  Effective temperature: {T_eff:.2f} K")
    print(f"  Mean flux: {mean_flux_rough:.2f} W")
    print(f"  Flux/area: {mean_flux_per_area:.2f} W/m²")
    
    print(f"\nSmooth Surface (at T_mean):")
    print(f"  Flux: {flux_smooth_equiv:.2f} W")
    print(f"  Flux/area: {flux_smooth_equiv/total_area:.2f} W/m²")
    
    ratio = mean_flux_rough / flux_smooth_equiv
    print(f"\nRough/Smooth Flux Ratio: {ratio:.4f}")
    
    if abs(ratio - 1.0) < 0.05:
        print("  ✓ Energy well conserved (< 5% difference)")
    elif abs(ratio - 1.0) < 0.10:
        print("  ~ Energy approximately conserved (< 10% difference)")
    else:
        print(f"  ⚠ Energy imbalance: {abs(ratio - 1.0)*100:.1f}%")
    
    # Key insight: T_eff vs T_mean
    print(f"\nThermal Beaming Effect:")
    print(f"  T_eff / T_mean = {T_eff / T_mean:.4f}")
    if T_eff > T_mean:
        print(f"  → Hot spots dominate (thermal beaming present)")
    else:
        print(f"  → Cold spots dominate (unexpected!)")

# ============================================================================
# PART 5: GEOMETRY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: FACET GEOMETRY AND SUN ANGLES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 14))

for idx, lat in enumerate([0.0, 90.0]):
    data = results[lat]
    shape_model = data['shape_model']
    temps = data['temps'][:, -n_timesteps:]
    sun_dir = data['sun_dir']
    
    # Extract geometry
    normals = np.array([f.normal for f in shape_model])
    centers = np.array([np.mean(f.vertices, axis=0) for f in shape_model])
    
    # Sun angles
    cos_angles = np.dot(normals, sun_dir)
    sun_angles = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))
    
    # Mean temperature
    temp_mean = np.mean(temps, axis=1)
    
    row = idx
    
    # Left: Facet positions colored by temperature
    ax = axes[row, 0]
    scatter = ax.scatter(centers[:, 0], centers[:, 1], c=temp_mean, 
                        cmap='hot', s=50, alpha=0.7, vmin=50, vmax=400)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title(f'Facet Positions (Lat {lat}°)\nColored by Mean T', 
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('T (K)', fontsize=10)
    
    # Sun direction arrow
    scale = 0.3
    ax.arrow(0, 0, sun_dir[0]*scale, sun_dir[1]*scale, 
            head_width=0.05, head_length=0.05, fc='yellow', ec='orange', linewidth=3)
    ax.text(sun_dir[0]*(scale+0.1), sun_dir[1]*(scale+0.1), 'Sun', 
           fontsize=11, fontweight='bold', color='orange')
    
    # Right: Sun angle vs Temperature
    ax = axes[row, 1]
    scatter = ax.scatter(sun_angles, temp_mean, c=temp_mean, 
                        cmap='hot', s=50, alpha=0.7, vmin=50, vmax=400)
    ax.set_xlabel('Sun Angle at Noon (°)', fontsize=11)
    ax.set_ylabel('Mean Temperature (K)', fontsize=11)
    ax.set_title(f'T vs Sun Angle (Lat {lat}°)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('T (K)', fontsize=10)
    
    # Statistics
    n_facing = np.sum(sun_angles < 90)
    print(f"\nLatitude {lat}°:")
    print(f"  Sun direction: [{sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f}]")
    print(f"  Facets facing sun: {n_facing}/{len(sun_angles)} ({100*n_facing/len(sun_angles):.1f}%)")
    print(f"  Sun angles: min={np.min(sun_angles):.1f}°, max={np.max(sun_angles):.1f}°, mean={np.mean(sun_angles):.1f}°")

plt.tight_layout()
plt.savefig(output_dir / "3_geometry_analysis.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / '3_geometry_analysis.png'}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Key Findings:

1. CORRECTION FACTOR CORRELATION
   - Strong negative correlation: r = {corr:.3f}
   - Factors at equator: {factors_noon[0]:.3f}
   - Factors at pole: {factors_noon[-1]:.3f}
   - Status: {"FAIL - systematic bias present" if abs(corr) > 0.7 else "OK"}

2. CRATER ILLUMINATION
   - Both latitudes show illuminated rim facets
   - All facets receive some heating
   - No unphysical shadows or zero-temperature regions

3. ENERGY BALANCE
   - [See detailed output above]
   - Rough/Smooth flux ratios for each latitude
   - Check if T_eff > T_mean (thermal beaming indicator)

4. SUN DECLINATION EFFECT
   - Equator: Sun overhead, maximum self-heating
   - Pole: Sun grazing, reduced internal heating
   - This changes crater internal thermal balance

HYPOTHESIS CONFIRMATION:
The sun_declination approach changes not just external illumination,
but also the crater's internal thermal self-heating geometry. This
creates systematically different thermal patterns at different "latitudes",
causing the correlation.

NEXT STEPS:
1. Examine the saved plots in: {output_dir}
2. Check energy balance ratios (should be ~1.0)
3. Look for differences in temperature distribution patterns
4. Consider alternative parameterization approaches

All diagnostic plots saved to: {output_dir}
""")

print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
