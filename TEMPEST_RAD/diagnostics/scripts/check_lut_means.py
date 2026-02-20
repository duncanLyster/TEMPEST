#!/usr/bin/env python3
"""
Diagnostic: Verify LUT normalization after per-(t,w) fix.

Checks:
1. cos(e)*sin(e)-weighted mean of R should be ~1.0 at each (t, w)
2. Simple (unweighted) mean should show both >1 and <1 regions
3. 2D grid at specific wavelengths should show limb brightening pattern
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import h5py
import numpy as np

LUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'roughness_lut_spectral_v1.h5')

f = h5py.File(LUT_FILE, 'r')
lut = f['lut'][:]
nf = f['normalization_factors'][:]
lats = f['latitude'][:]
waves = f['wavelength'][:]
emi = f['emission'][:]
azi = f['azimuth'][:]

print(f"LUT shape: {lut.shape}")
print(f"Emission angles: {np.round(emi, 1)}")
print(f"Azimuth angles:  {np.round(azi, 1)}")
print(f"Wavelengths:     {waves}")
print()

# Build cos(e)*sin(e)*d_omega weight grid
em_rad = np.radians(emi)
az_rad = np.radians(azi)

d_em = np.diff(em_rad); d_em = np.append(d_em, d_em[-1])
d_az = np.diff(az_rad); d_az = np.append(d_az, d_az[-1])

cos_e = np.cos(em_rad)
sin_e = np.sin(em_rad)

cse_weights = np.zeros((len(emi), len(azi)))
for ie in range(len(emi)):
    for ia in range(len(azi)):
        cse_weights[ie, ia] = cos_e[ie] * sin_e[ie] * d_em[ie] * d_az[ia] * 2.0

ref_integral = np.sum(cse_weights)
print(f"Reference integral (should be ~pi={np.pi:.4f}): {ref_integral:.4f}")
print()

# =====================================================================
print("=" * 100)
print("1. COS(e)*SIN(e)-WEIGHTED MEAN per latitude (MUST be ~1.0 for energy conservation)")
print("=" * 100)
for i_lat in range(0, len(lats), 2):
    lat = lats[i_lat]
    vals = lut[0, 0, i_lat]  # (Time, Wave, Emi, Azi)
    
    tw_means = []
    for t in range(vals.shape[0]):
        for w in range(vals.shape[1]):
            wm = np.sum(vals[t, w] * cse_weights) / ref_integral
            tw_means.append(wm)
    
    tw_means = np.array(tw_means)
    print(f"  Lat {lat:5.1f}: cos_sin-weighted mean = {np.mean(tw_means):.4f} +/- {np.std(tw_means):.4f}  "
          f"[min={np.min(tw_means):.4f}, max={np.max(tw_means):.4f}]  "
          f"avg_norm={nf[0,0,i_lat]:.4f}")

# =====================================================================
print()
print("=" * 100)
print("2. SIMPLE (UNWEIGHTED) MEAN per latitude")
print("=" * 100)
for i_lat in range(0, len(lats), 4):
    lat = lats[i_lat]
    vals = lut[0, 0, i_lat]
    print(f"  Lat {lat:5.1f}: mean={np.nanmean(vals):.4f}  median={np.nanmedian(vals):.4f}  "
          f"min={np.nanmin(vals):.4f}  max={np.nanmax(vals):.4f}  "
          f"frac>1={np.sum(vals>1)/vals.size*100:.1f}%  frac<1={np.sum(vals<1)/vals.size*100:.1f}%")

# =====================================================================
print()
print("=" * 100)
print("3. PER-EMISSION-ANGLE MEAN at Lat=0 (should show limb brightening)")
print("=" * 100)
for ie, e in enumerate(emi):
    vals = lut[0, 0, 0, :, :, ie, :]  # (Time, Wave, Azi)
    print(f"  e={e:5.1f}: mean={np.nanmean(vals):.4f}  min={np.nanmin(vals):.4f}  max={np.nanmax(vals):.4f}")

# =====================================================================
print()
print("=" * 100)
print("4. 2D GRID: Lat=0, TIME-AVERAGED (Emission x Azimuth) per wavelength")
print("   >1 = rough WARMER, <1 = rough COOLER")
print("=" * 100)
for iw, wv in enumerate(waves):
    print(f"\n  --- lambda = {wv} um ---")
    vals = lut[0, 0, 0, :, iw, :, :]  # (Time, Emi, Azi)
    time_avg = np.nanmean(vals, axis=0)  # (Emi, Azi)
    header = "      Azi:" + "".join(f"{a:7.1f}" for a in azi)
    print(header)
    for ie, e in enumerate(emi):
        row = "".join(f"{time_avg[ie, ia]:7.3f}" for ia in range(len(azi)))
        print(f"  e={e:5.1f}:{row}")

# =====================================================================
print()
print("=" * 100)
print("5. VERIFICATION: Simulated retrieval at specific geometries")
print("   What would a spacecraft see for equatorial facets?")
print("=" * 100)
f_rough = 0.35  # rms_to_fraction(28) ~ 0.35
T_smooth = 300.0

sigma = 5.670374419e-8
c1 = 1.191042e8
c2 = 1.4387752e4

def planck(wave_um, T):
    t_safe = max(T, 1e-5)
    return c1 / (wave_um**5 * (np.exp(c2 / (wave_um * t_safe)) - 1))

test_geometries = [
    (0.0, 0.0, "Nadir, zero phase"),
    (30.0, 0.0, "e=30, az=0 (subsolar limb)"),
    (30.0, 180.0, "e=30, az=180 (antisolar)"),
    (60.0, 0.0, "e=60, az=0 (limb, subsolar)"),
    (60.0, 180.0, "e=60, az=180 (limb, antisolar)"),
]

for e_test, a_test, label in test_geometries:
    spectra_smooth = np.array([planck(w, T_smooth) for w in waves])
    
    ie = np.argmin(np.abs(emi - e_test))
    ia = np.argmin(np.abs(azi - a_test))
    
    R_vals = lut[0, 0, 0, :, :, ie, ia].mean(axis=0)  # (Wave,)
    
    spectra_rough = spectra_smooth * ((1 - f_rough) + f_rough * R_vals)
    
    rad_bol_smooth = np.trapz(spectra_smooth, x=waves)
    rad_bol_rough = np.trapz(spectra_rough, x=waves)
    
    T_bol_smooth = (np.pi * rad_bol_smooth / sigma) ** 0.25
    T_bol_rough = (np.pi * rad_bol_rough / sigma) ** 0.25
    
    delta_T = T_bol_rough - T_bol_smooth
    
    R_str = ", ".join(f"{r:.3f}" for r in R_vals)
    print(f"  {label:35s}: R=[{R_str}]  Ts={T_bol_smooth:.1f}K  Tr={T_bol_rough:.1f}K  dT={delta_T:+.1f}K")

f.close()
print("\nDone.")
