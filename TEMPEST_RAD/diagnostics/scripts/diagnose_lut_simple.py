#!/usr/bin/env python3
"""
LUT Correlation Diagnostic - Quick Analysis

Analyzes existing LUT to understand the latitude correlation issue.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from scipy.stats import pearsonr
from datetime import datetime
import os

# Setup
root_dir = Path(__file__).parent.resolve()
output_dir = root_dir / "diagnostic_plots"
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("LUT CORRELATION DIAGNOSTIC")
print("=" * 80)
print(f"\nOutput directory: {output_dir}\n")

# Load LUT
LUT_FILE = root_dir / "roughness_lut_moon_test.h5"

print("=" * 80)
print("LOADING LUT DATA")
print("=" * 80)

lut_timestamp = os.path.getmtime(LUT_FILE)
print(f"\nFile: {LUT_FILE.name}")
print(f"Modified: {datetime.fromtimestamp(lut_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Size: {LUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")

with h5py.File(LUT_FILE, 'r') as f:
    latitudes = f['latitude'][...]
    wavelengths = f['wavelength'][...]
    factors = f['lut'][...]  # [theta, angle, lat, time, wave, emission, azimuth]
    
    print(f"\nLUT Shape: {factors.shape}")
    print(f"Dimensions:")
    print(f"  [0] Theta: {factors.shape[0]}")
    print(f"  [1] Opening angles: {factors.shape[1]}")
    print(f"  [2] Latitudes: {factors.shape[2]} values: {latitudes}°")
    print(f"  [3] Time steps: {factors.shape[3]} (phase: 0-360° by 2°)")
    print(f"  [4] Wavelengths: {factors.shape[4]} values: {wavelengths} µm")
    print(f"  [5] Emission angles: {factors.shape[5]} (0-90°)")
    print(f"  [6] Azimuth angles: {factors.shape[6]} (0-360°)")

# ============================================================================
# ANALYSIS 1: FACTORS AT IDEAL GEOMETRY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 1: CORRECTION FACTORS AT NOON, DISK CENTER")
print("=" * 80)

i_theta = 0
i_angle = 0
i_time = 90  # Noon (180° / 2°)
i_wave = np.argmin(np.abs(wavelengths - 15.0))
i_emi = 0
i_azi = 0

factors_noon = factors[i_theta, i_angle, :, i_time, i_wave, i_emi, i_azi]

print(f"\nConditions: Noon (180°), Disk Center (0°), λ={wavelengths[i_wave]:.1f}µm\n")
print(f"{'Latitude':>10}  {'Factor':>8}  {'Expected':>12}  {'Status':>8}")
print("-" * 60)

for lat, factor in zip(latitudes, factors_noon):
    status = "✓" if factor > 1.0 else "✗"
    print(f"{lat:10.1f}° {factor:8.4f}  {'>1.0 (beaming)':>12}  {status:>8}")

corr, pval = pearsonr(latitudes, factors_noon)

print(f"\n{'=' * 60}")
print(f"CORRELATION TEST:")
print(f"  Pearson r = {corr:+.4f}")
print(f"  p-value   = {pval:.6f}")
print(f"  Status: {'❌ FAIL (systematic)' if abs(corr) > 0.7 else '✅ PASS (random)'}")
print("=" * 60)

# ============================================================================
# ANALYSIS 2: VARIATION ACROSS PARAMETER SPACE
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: FACTOR VARIATION ACROSS CONDITIONS")
print("=" * 80)

# Check all wavelengths
print(f"\n At each wavelength (Noon, Disk Center):")
for i_w, wave in enumerate(wavelengths):
    factors_wave = factors[i_theta, i_angle, :, i_time, i_w, i_emi, i_azi]
    r_wave, _ = pearsonr(latitudes, factors_wave)
    print(f"  λ={wave:5.1f}µm:  r={r_wave:+.3f}  range=[{np.min(factors_wave):.3f}, {np.max(factors_wave):.3f}]")

# Check across phase angles (all latitudes averaged)
print(f"\nAveraged over all latitudes vs phase:")
phase_angles = np.arange(0, 360, 2)
factors_vs_phase = np.mean(factors[i_theta, i_angle, :, :, i_wave, i_emi, i_azi], axis=0)
print(f"  Phase range: {np.min(factors_vs_phase):.3f} to {np.max(factors_vs_phase):.3f}")
print(f"  Max at phase: {phase_angles[np.argmax(factors_vs_phase)]}°")
print(f"  Min at phase: {phase_angles[np.argmin(factors_vs_phase)]}°")

# ============================================================================
# ANALYSIS 3: PHYSICAL INTERPRETATION
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 3: PHYSICAL INTERPRETATION")
print("=" * 80)

print(f"\nKey Observations:")
print(f"1. Factors decrease from equator ({factors_noon[0]:.3f}) to pole ({factors_noon[-1]:.3f})")
print(f"2. Decrease is {(factors_noon[0] - factors_noon[-1])/factors_noon[0]*100:.1f}%")
print(f"3. Pole factor = {factors_noon[-1]:.3f}:")
if factors_noon[-1] > 1.2:
    print(f"   → Strong thermal beaming (hot spots dominate)")
elif factors_noon[-1] > 1.0:
    print(f"   → Weak thermal beaming present")
else:
    print(f"   → No beaming! Rough cooler than smooth (unexpected!)")

print(f"\n4. Correlation strength r={corr:.3f} indicates:")
if abs(corr) > 0.9:
    print(f"   → Very strong systematic bias (likely geometry error)")
elif abs(corr) > 0.7:
    print(f"   → Strong systematic trend (not random discretization)")
elif abs(corr) > 0.3:
    print(f"   → Moderate correlation (investigate further)")
else:
    print(f"   → Weak/no correlation (expected random variation)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Figure 1: Basic correlation plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1a: Factors vs Latitude
ax = axes[0]
ax.plot(latitudes, factors_noon, 'o-', markersize=12, linewidth=3, 
        color='darkred', label='Correction Factor')
ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Smooth Surface')
ax.set_xlabel('Latitude (°)', fontsize=12)
ax.set_ylabel('Correction Factor', fontsize=12)
ax.set_title(f'Latitude Dependence\nr = {corr:+.3f} ({"FAIL" if abs(corr) > 0.7 else "OK"})', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# 1b: Latitude-Time Heatmap
ax = axes[1]
lat_time_map = factors[i_theta, i_angle, :, :, i_wave, i_emi, i_azi]

im = ax.imshow(lat_time_map, aspect='auto', origin='lower', cmap='hot',
               extent=[0, 360, latitudes[0], latitudes[-1]], vmin=0.8, vmax=2.5)
ax.axvline(180, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='Noon')
ax.set_xlabel('Phase Angle (°)', fontsize=12)
ax.set_ylabel('Latitude (°)', fontsize=12)
ax.set_title(f'Factor Map (λ={wavelengths[i_wave]:.1f}µm)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Factor', fontsize=11)

# 1c: Phase curve at different latitudes
ax = axes[2]
for i_lat, lat in enumerate([0., 45., 90.]):
    lat_idx = np.argmin(np.abs(latitudes - lat))
    phase_curve = factors[i_theta, i_angle, lat_idx, :, i_wave, i_emi, i_azi]
    ax.plot(phase_angles, phase_curve, linewidth=2, label=f'{lat:.0f}°')

ax.set_xlabel('Phase Angle (°)', fontsize=12)
ax.set_ylabel('Correction Factor', fontsize=12)
ax.set_title('Phase Curves by Latitude', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(title='Latitude', fontsize=10)
ax.axhline(1.0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "1_correlation_analysis.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / '1_correlation_analysis.png'}")
plt.close()

# Figure 2: Wavelength and viewing geometry dependencies
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 2a: Wavelength dependence
ax = axes[0, 0]
for i_lat, lat in enumerate(latitudes):
    factors_wave = factors[i_theta, i_angle, i_lat, i_time, :, i_emi, i_azi]
    ax.plot(wavelengths, factors_wave, 'o-', markersize=8, linewidth=2, label=f'{lat:.0f}°')
ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Wavelength (µm)', fontsize=11)
ax.set_ylabel('Correction Factor', fontsize=11)
ax.set_title('Wavelength Dependence (Noon)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(title='Latitude', fontsize=9)

# 2b: Correlation at each wavelength
ax = axes[0, 1]
correlations = []
for i_w, wave in enumerate(wavelengths):
    factors_wave = factors[i_theta, i_angle, :, i_time, i_w, i_emi, i_azi]
    r_wave, _ = pearsonr(latitudes, factors_wave)
    correlations.append(r_wave)

ax.bar(range(len(wavelengths)), correlations, color=['red' if abs(r) > 0.7 else 'orange' for r in correlations])
ax.set_xticks(range(len(wavelengths)))
ax.set_xticklabels([f'{w:.0f}' for w in wavelengths])
ax.set_xlabel('Wavelength (µm)', fontsize=11)
ax.set_ylabel('Correlation r', fontsize=11)
ax.set_title('Latitude Correlation by Wavelength', fontsize=12, fontweight='bold')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (|r|=0.7)')
ax.axhline(-0.7, color='red', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=9)

# 2c: Emission angle dependence (equator vs pole)
ax = axes[1, 0]
emission_angles = np.linspace(0, 90, factors.shape[5])
factors_eq = factors[i_theta, i_angle, 0, i_time, i_wave, :, i_azi]
factors_pole = factors[i_theta, i_angle, -1, i_time, i_wave, :, i_azi]

ax.plot(emission_angles, factors_eq, 'o-', markersize=8, linewidth=2, 
        color='orange', label=f'Equator (0°)')
ax.plot(emission_angles, factors_pole, 's-', markersize=8, linewidth=2, 
        color='blue', label=f'Pole (90°)')
ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Emission Angle (°)', fontsize=11)
ax.set_ylabel('Correction Factor', fontsize=11)
ax.set_title('Limb Darkening Effect', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# 2d: Difference (Equator - Pole) vs phase
ax = axes[1, 1]
diff_vs_phase = factors[i_theta, i_angle, 0, :, i_wave, i_emi, i_azi] - factors[i_theta, i_angle, -1, :, i_wave, i_emi, i_azi]
ax.plot(phase_angles, diff_vs_phase, linewidth=2, color='purple')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(180, color='red', linestyle='--', alpha=0.5, label='Noon')
ax.set_xlabel('Phase Angle (°)', fontsize=11)
ax.set_ylabel('Factor Difference (Eq - Pole)', fontsize=11)
ax.set_title('Latitude Effect vs Phase', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.fill_between(phase_angles, diff_vs_phase, alpha=0.3, color='purple')

plt.tight_layout()
plt.savefig(output_dir / "2_detailed_analysis.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '2_detailed_analysis.png'}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
KEY FINDINGS:

1. CORRELATION STATUS
   - Pearson r = {corr:+.4f} (p = {pval:.4f})
   - Status: {"❌ FAIL - Strong systematic bias" if abs(corr) > 0.7 else "✅ PASS"}
   - Factor range: {factors_noon[0]:.3f} (equator) → {factors_noon[-1]:.3f} (pole)
   - Decrease: {(factors_noon[0] - factors_noon[-1])/factors_noon[0]*100:.1f}%

2. WAVELENGTH DEPENDENCE
   - All wavelengths show similar correlation pattern
   - Correlations: {[f'{r:.3f}' for r in correlations]}
   - Indicates geometry/physics issue, not wavelength-specific

3. PHASE ANGLE BEHAVIOR
   - Latitude effect persists across all phase angles
   - Maximum difference at phase ≈ {phase_angles[np.argmax(np.abs(diff_vs_phase))]}°
   - Not limited to noon conditions

4. PHYSICAL INTERPRETATION
   - Expected: Factors > 1.0 (thermal beaming) with random variation
   - Observed: Systematic decrease with latitude
   - Hypothesis: Sun declination approach changes crater thermal geometry

DIAGNOSTIC PLOTS SAVED:
  → {output_dir / '1_correlation_analysis.png'}
  → {output_dir / '2_detailed_analysis.png'}

RECOMMENDATION:
The strong systematic correlation (r ≈ {corr:.2f}) indicates the "latitude" 
parameter is affecting not just external illumination but also the crater's
internal thermal self-heating geometry. This violates the assumption that
rough/smooth should have equal energy over a full rotation.
""")

print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
