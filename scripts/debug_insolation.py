#!/usr/bin/env python3
"""Compare insolation between TEMPEST and BTPM to find the source of temperature difference."""
import numpy as np
import sys
from scipy.interpolate import interp1d

sys.path.insert(0, 'TEMPEST_RAD/diagnostics/DGL_KS_comparison/BinaryThermophysicalModel')
from planets import Itokawa as planet

# ====== 1. Solar flux computation comparison ======
print("=" * 60)
print("1. SOLAR FLUX COMPARISON")
print("=" * 60)

# BTPM: solarLum = L/(4π), then flux = solarLum / r_m^2
solarLum_btpm = 3.826e26 / (4.0 * np.pi)
r_m_btpm = planet.rsm  # 1.34e11 m
solarFlux_btpm = solarLum_btpm / r_m_btpm**2
print(f"BTPM:    solarFlux = {solarFlux_btpm:.6f} W/m^2")
print(f"         r = {r_m_btpm:.6e} m  ({r_m_btpm/1.496e11:.6f} AU)")

# TEMPEST: flux = L / (4π r^2)
L = 3.826e26
r_au = 0.896
r_m_tempest = r_au * 1.496e11
solarFlux_tempest = L / (4 * np.pi * r_m_tempest**2)
print(f"TEMPEST: solarFlux = {solarFlux_tempest:.6f} W/m^2")
print(f"         r = {r_m_tempest:.6e} m  ({r_au:.6f} AU)")

print(f"\nFlux ratio (BTPM/TEMPEST): {solarFlux_btpm/solarFlux_tempest:.8f}")
print(f"Flux difference: {solarFlux_btpm - solarFlux_tempest:.4f} W/m^2")

# ====== 2. Shadow/cos(theta) comparison ======
print("\n" + "=" * 60)
print("2. SHADOW & COS(INCIDENCE) COMPARISON")
print("=" * 60)

# BTPM shadow array: (360, 98) - contains cos(incidence) * visibility
shadow = np.load('TEMPEST_RAD/diagnostics/DGL_KS_comparison/BinaryThermophysicalModel/Shadow_Data/itokawa_100_facets_shadows.npy')
print(f"BTPM shadow shape: {shadow.shape} (360 rotation steps, 98 facets)")
print(f"Shadow range: [{shadow.min():.6f}, {shadow.max():.6f}]")

# TEMPEST insolation: already absorbed flux = (1-A) * cos(theta) * shadow * L/(4πr²) + scattering
insol_csv = np.genfromtxt('insolation_data/all_facets_insolation.csv', delimiter=',', skip_header=1)
tempest_insol = insol_csv[:, 1:]  # (n_timesteps, 98)
tempest_angles = insol_csv[:, 0]  # rotation degrees
n_steps = tempest_insol.shape[0]
print(f"TEMPEST insolation shape: {tempest_insol.shape}")

# ====== 3. Reconstruct BTPM absorbed insolation ======
print("\n" + "=" * 60)
print("3. ABSORBED INSOLATION COMPARISON (per facet)")
print("=" * 60)

# Resample BTPM shadows to TEMPEST rotation angles
btpm_angles = np.linspace(0, 360, shadow.shape[0], endpoint=False)
f_interp = interp1d(
    np.append(btpm_angles, 360),
    np.vstack([shadow, shadow[0:1]]),
    axis=0, kind='linear'
)
btpm_shadows_resampled = f_interp(tempest_angles % 360)

# BTPM absorbed insolation: (1-A) * solarFlux * shadow
# shadow already contains cos(incidence) * visibility
btpm_absorbed = (1 - planet.albedo) * solarFlux_btpm * btpm_shadows_resampled

# Load facet matching
matching_data = np.load('TEMPEST_RAD/diagnostics/DGL_KS_comparison_outputs/DGL_KS_facet_matching.npz', allow_pickle=True)
stl_to_btpm = matching_data['matching'].item()  # dict: stl_idx -> btpm_idx

# Per-facet comparison
print(f"\n{'STL':>5} {'BTPM':>5} {'TEMPEST_mean':>14} {'BTPM_mean':>14} {'Ratio':>8} {'Diff':>10} {'Peak_T':>10} {'Peak_B':>10}")
for stl_i in range(98):
    if stl_i not in stl_to_btpm:
        continue
    btpm_i = stl_to_btpm[stl_i]
    t_mean = np.mean(tempest_insol[:, stl_i])
    b_mean = np.mean(btpm_absorbed[:, btpm_i])
    t_peak = np.max(tempest_insol[:, stl_i])
    b_peak = np.max(btpm_absorbed[:, btpm_i])
    ratio = t_mean / b_mean if b_mean > 0 else float('inf')
    if stl_i in [15, 45, 55, 0, 50, 90]:
        print(f"{stl_i:>5} {btpm_i:>5} {t_mean:>14.2f} {b_mean:>14.2f} {ratio:>8.4f} {t_mean-b_mean:>10.2f} {t_peak:>10.2f} {b_peak:>10.2f}")

# Overall stats
all_ratios = []
all_diffs = []
for stl_i in range(98):
    if stl_i not in stl_to_btpm:
        continue
    btpm_i = stl_to_btpm[stl_i]
    t_mean = np.mean(tempest_insol[:, stl_i])
    b_mean = np.mean(btpm_absorbed[:, btpm_i])
    if b_mean > 0:
        all_ratios.append(t_mean / b_mean)
    all_diffs.append(t_mean - b_mean)

print(f"\nOverall stats across all matched facets:")
print(f"  Mean ratio (TEMPEST/BTPM): {np.mean(all_ratios):.6f}")
print(f"  Std ratio:                 {np.std(all_ratios):.6f}")
print(f"  Mean diff (T-B):           {np.mean(all_diffs):.2f} W/m^2")
print(f"  Max diff:                  {np.max(np.abs(all_diffs)):.2f} W/m^2")

# ====== 4. Check if TEMPEST insolation includes scattering ======
print("\n" + "=" * 60)
print("4. DOES TEMPEST INSOLATION INCLUDE SCATTERING?")
print("=" * 60)

# If TEMPEST includes scatter, its nightside values would be non-zero
# where BTPM nightside would be zero
nightside_tempest = []
nightside_btpm = []
for stl_i in range(98):
    if stl_i not in stl_to_btpm:
        continue
    btpm_i = stl_to_btpm[stl_i]
    t_curve = tempest_insol[:, stl_i]
    b_curve = btpm_absorbed[:, btpm_i]
    # Find timesteps where BTPM says nightside (shadow=0)
    night_mask = b_curve == 0
    if night_mask.any():
        nightside_tempest.extend(t_curve[night_mask].tolist())
        nightside_btpm.extend(b_curve[night_mask].tolist())

nightside_tempest = np.array(nightside_tempest)
print(f"Nightside samples: {len(nightside_tempest)}")
print(f"TEMPEST on BTPM-nightside: mean={nightside_tempest.mean():.4f}, max={nightside_tempest.max():.4f}")
print(f"  Non-zero count: {np.sum(nightside_tempest > 0)}")
print(f"  This tells us whether TEMPEST's exported insolation includes scattered light")

# ====== 5. Compare peak insolation timing per facet ======
print("\n" + "=" * 60)
print("5. PEAK INSOLATION TIMING")
print("=" * 60)

for stl_i in [15, 45, 55]:
    if stl_i not in stl_to_btpm:
        continue
    btpm_i = stl_to_btpm[stl_i]
    t_peak_idx = np.argmax(tempest_insol[:, stl_i])
    b_peak_idx = np.argmax(btpm_absorbed[:, btpm_i])
    t_peak_angle = tempest_angles[t_peak_idx]
    b_peak_angle = btpm_angles[np.argmax(shadow[:, btpm_i])] if np.max(shadow[:, btpm_i]) > 0 else -1
    print(f"Facet STL={stl_i} BTPM={btpm_i}:")
    print(f"  TEMPEST peak at angle {t_peak_angle:.1f}° (step {t_peak_idx})")
    print(f"  BTPM peak at angle {b_peak_angle:.1f}° (from shadow LUT)")
    print(f"  Phase offset: {t_peak_angle - b_peak_angle:.1f}°")

# ====== 6. Compare diurnal insolation curves for specific facets ======
print("\n" + "=" * 60)
print("6. DIURNAL INSOLATION CURVE COMPARISON (Facet 45)")
print("=" * 60)

stl_i = 45
btpm_i = stl_to_btpm[stl_i]
print(f"STL facet {stl_i} -> BTPM facet {btpm_i}")

t_curve = tempest_insol[:, stl_i]
b_curve = btpm_absorbed[:, btpm_i]

# Sample at 12 evenly spaced rotation angles
sample_angles = np.linspace(0, 360, 13)[:-1]
sample_indices_t = [np.argmin(np.abs(tempest_angles - a)) for a in sample_angles]

print(f"\n{'Angle':>8} {'TEMPEST':>12} {'BTPM':>12} {'Diff':>10} {'Ratio':>8}")
for sa, si in zip(sample_angles, sample_indices_t):
    tv = t_curve[si]
    bv = b_curve[si]
    d = tv - bv
    r = tv / bv if bv > 0 else float('inf')
    r_str = f"{r:.4f}" if bv > 0 else "inf"
    print(f"{sa:>8.1f} {tv:>12.2f} {bv:>12.2f} {d:>10.2f} {r_str:>8}")

print("\nDone.")
