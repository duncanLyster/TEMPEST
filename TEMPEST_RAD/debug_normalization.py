"""
Diagnostic script to track down why the rough surface appears always hotter
than the smooth surface after normalization.

Key question: norm_factor = smooth_total / rough_total ≈ 1.78.
This means rough_total = 0.56 * smooth_total.
By energy conservation, they should be equal (both absorb same solar flux).
"""
import os, sys
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
os.chdir(root_dir)

from src.model.facet import Facet
from src.model.simulation import Simulation, ThermalData
from src.model.solvers import TemperatureSolverFactory
from src.utilities.config import Config
from src.utilities.utils import calculate_rotation_matrix

from src.model.insolation import calculate_insolation
from src.model.view_factors import calculate_all_view_factors, calculate_thermal_view_factors

# Import generator config
from TEMPEST_RAD.generator import (
    ReferenceConfig, planck_function,
    THETA_VALUES, OPENING_ANGLES, WAVELENGTHS_MICRONS,
    EMISSION_ANGLES, AZIMUTH_ANGLES,
    CRATER_SUBFACETS, VIEW_FACTOR_RAYS,
    SIM_TIMESTEPS, LUT_TIMESTEPS,
    simulate_crater_diurnal_cycle
)

config = ReferenceConfig()
theta = THETA_VALUES[0]
opening_angle = OPENING_ANGLES[0]
lat = 0.0

print(f"Running diagnostic for Theta={theta}, Angle={opening_angle}, Lat={lat}")
print("="*80)

# 1. Run simulation
rough_temps_sim, smooth_temps_sim, facet, sun_vectors_sim = simulate_crater_diurnal_cycle(
    theta, opening_angle, lat, config, SIM_TIMESTEPS
)

# Resample
indices = np.linspace(0, SIM_TIMESTEPS-1, LUT_TIMESTEPS, dtype=int)
rough_temps = rough_temps_sim[:, indices]
smooth_temps = smooth_temps_sim[indices]
sun_vectors = sun_vectors_sim[indices]

n_sub = rough_temps.shape[0]
print(f"\n--- TEMPERATURE SUMMARY ---")
print(f"N subfacets: {n_sub}")
print(f"Smooth T: min={smooth_temps.min():.1f}, max={smooth_temps.max():.1f}, mean={smooth_temps.mean():.1f}")
print(f"Rough  T: min={rough_temps.min():.1f}, max={rough_temps.max():.1f}, mean={rough_temps.mean():.1f}")

# Check Stefan-Boltzmann energy
sigma = 5.670374419e-8
mesh = Facet._canonical_subfacet_mesh
areas = np.array([e['area'] for e in mesh])
normals = np.array([e['normal'] for e in mesh])
triangles = np.array([e['vertices'] for e in mesh])
centers = np.array([np.mean(e['vertices'], axis=0) for e in mesh])

print(f"\nTotal crater surface area: {areas.sum():.4f}")
print(f"Aperture area (flat facet): {facet.area:.4f}")

# 2. Check energy balance via Stefan-Boltzmann
# Total emitted by smooth surface (averaged over rotation):
P_smooth_sb = sigma * np.mean(smooth_temps**4) * facet.area
# Total emitted by rough surface (we need view-factor-to-sky for each subfacet):
# Instead, compute σ<T^4>*A for all subfacets
P_rough_total_emission = sigma * np.mean(np.sum(rough_temps**4 * areas[:, np.newaxis], axis=0))

print(f"\n--- ENERGY BALANCE (Stefan-Boltzmann) ---")
print(f"P_smooth (σ<T⁴>*A_flat): {P_smooth_sb:.4f} W")
print(f"P_rough_total (σΣ(T_i⁴*A_i)): {P_rough_total_emission:.4f} W")
print(f"Ratio rough_total/smooth: {P_rough_total_emission/P_smooth_sb:.4f}")
print("  Note: rough_total includes internal re-absorption. P_escape < P_rough_total.")

# 3. Compute the LUT viewing integral directly
facet_normal = facet.normal / np.linalg.norm(facet.normal)

em_rad = np.radians(EMISSION_ANGLES)
az_rad = np.radians(AZIMUTH_ANGLES)
d_em = np.diff(em_rad); d_em = np.append(d_em, d_em[-1])
d_az = np.diff(az_rad); d_az = np.append(d_az, d_az[-1])

waves = np.array(WAVELENGTHS_MICRONS)
wave_weights = np.zeros_like(waves)
if len(waves) > 1:
    wave_weights[0] = (waves[1] - waves[0]) / 2.0
    wave_weights[1:-1] = (waves[2:] - waves[0:-2]) / 2.0
    wave_weights[-1] = (waves[-1] - waves[-2]) / 2.0
else:
    wave_weights[:] = 1.0

# Accumulate with d_omega as currently coded (with extra cos_e)
total_rough_current = 0.0
total_smooth_current = 0.0

# Also accumulate with CORRECTED d_omega (without extra cos_e)
total_rough_correct = 0.0
total_smooth_correct = 0.0

# Also accumulate the DIRECT intensity integral (no E_vis normalization issue)
total_rough_direct = 0.0

# Sample one midday timestep for detailed diagnostics
t_mid = LUT_TIMESTEPS // 4  # Approximately noon

print(f"\n--- DETAILED VIEW AT t_idx={t_mid} (approx noon) ---")
print(f"Smooth T(noon) = {smooth_temps[t_mid]:.1f} K")
print(f"Rough T(noon): min={rough_temps[:, t_mid].min():.1f}, max={rough_temps[:, t_mid].max():.1f}, mean={rough_temps[:, t_mid].mean():.1f}")

for t_idx in range(LUT_TIMESTEPS):
    t_sub = rough_temps[:, t_idx]
    t_smooth = smooth_temps[t_idx]
    
    sun_vec_norm = sun_vectors[t_idx] / np.linalg.norm(sun_vectors[t_idx])
    s_dot_n = np.dot(sun_vec_norm, facet_normal)
    sun_proj = sun_vec_norm - s_dot_n * facet_normal
    if np.linalg.norm(sun_proj) > 1e-6:
        basis_x = sun_proj / np.linalg.norm(sun_proj)
    else:
        up = np.array([0, 0, 1]) if np.abs(np.dot([0,0,1], facet_normal)) < 0.9 else np.array([1, 0, 0])
        basis_x = np.cross(facet_normal, up)
        basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(facet_normal, basis_x)

    for i_e, emi_deg in enumerate(EMISSION_ANGLES):
        emi_rad_val = np.radians(emi_deg)
        sin_e = np.sin(emi_rad_val)
        cos_e = np.cos(emi_rad_val)
        weight_e_current = sin_e * cos_e * d_em[i_e]  # Current (with extra cos)
        weight_e_correct = sin_e * d_em[i_e]           # Correct (just sin*de)

        for i_a, azi_deg in enumerate(AZIMUTH_ANGLES):
            azi_rad_val = np.radians(azi_deg)
            
            v_local_x = sin_e * np.cos(azi_rad_val)
            v_local_y = sin_e * np.sin(azi_rad_val)
            v_local_z = cos_e
            view_vec = v_local_x * basis_x + v_local_y * basis_y + v_local_z * facet_normal
            view_vec /= np.linalg.norm(view_vec)
            
            # Use _process_incident_packet for E_vis (same as generator)
            packet = (1.0, view_vec, 'visible')
            E_vis, _, _, _ = Facet._process_incident_packet(
                packet, facet.world_to_local_rotation, facet.area, 
                normals, areas, triangles, centers, 0.0, 0.0
            )
            
            # Also compute DIRECT intensity (correct physics)
            d_local = facet.world_to_local_rotation.dot(view_vec)
            cos_theta_all = normals.dot(d_local)
            front_facing = cos_theta_all > 0
            
            weight_azi = d_az[i_a] * 2.0
            
            for i_w, wave in enumerate(WAVELENGTHS_MICRONS):
                rad_sub = planck_function(wave, t_sub)
                
                # Method 1: Current code (E_vis based)
                rough_radiance = np.sum(rad_sub * E_vis)
                
                # Method 2: Direct intensity (correct physics)
                # I = Σ_visible B(T_i) * A_i * cos_theta_i
                direct_intensity = np.sum(rad_sub[front_facing] * areas[front_facing] * cos_theta_all[front_facing])
                
                if t_smooth < 5.0:
                    smooth_radiance = 0.0
                else:
                    smooth_radiance = planck_function(wave, t_smooth) * facet.area * cos_e
                
                d_lambda = wave_weights[i_w]
                
                # Current d_omega (with extra cos_e)
                d_omega_current = weight_e_current * weight_azi
                total_rough_current += rough_radiance * d_omega_current * d_lambda
                total_smooth_current += smooth_radiance * d_omega_current * d_lambda
                
                # Correct d_omega (without extra cos_e, for spectral intensity)
                d_omega_correct = weight_e_correct * weight_azi
                total_rough_correct += rough_radiance * d_omega_correct * d_lambda
                total_smooth_correct += smooth_radiance * d_omega_correct * d_lambda
                
                # Direct intensity with correct d_omega
                total_rough_direct += direct_intensity * weight_e_correct * weight_azi * d_lambda
                
                # Print detailed info for midday at selected angles
                if t_idx == t_mid and i_w == 2 and azi_deg == 0:  # λ=15μm, a=0
                    E_vis_sum = np.sum(E_vis)
                    if i_e < 5:  # First few emission angles
                        print(f"  e={emi_deg:5.1f}°, a={azi_deg:5.1f}°, λ={wave}μm: "
                              f"rough_rad={rough_radiance:.4e}, smooth_rad={smooth_radiance:.4e}, "
                              f"ratio={rough_radiance/max(smooth_radiance,1e-20):.4f}, "
                              f"ΣE_vis={E_vis_sum:.4f}, cos_e={cos_e:.4f}, "
                              f"direct_I={direct_intensity:.4e}")

# Smooth total for direct comparison
# For smooth: I_smooth = B * A * cos_e → direct integral = ∫ B * A * cos_e * sin_e * de * dφ * dλ
total_smooth_direct = total_smooth_correct  # Same as correct smooth (it's already I * sin*de*dφ*dλ)

print(f"\n--- NORMALIZATION INTEGRALS ---")
print(f"  Current method (cos²·sin weighting):")
print(f"    total_smooth = {total_smooth_current:.6e}")
print(f"    total_rough  = {total_rough_current:.6e}")
print(f"    norm_factor  = {total_smooth_current/total_rough_current:.4f}")

print(f"\n  Corrected method (cos·sin weighting):")
print(f"    total_smooth = {total_smooth_correct:.6e}")
print(f"    total_rough  = {total_rough_correct:.6e}")
print(f"    norm_factor  = {total_smooth_correct/total_rough_correct:.4f}")

print(f"\n  Direct intensity (bypass E_vis normalization):")
print(f"    total_smooth = {total_smooth_correct:.6e}")
print(f"    total_rough  = {total_rough_direct:.6e}")
print(f"    norm_factor  = {total_smooth_correct/total_rough_direct:.4f}")

print(f"\n--- ANALYSIS ---")
if total_smooth_current > total_rough_current:
    print(f"Rough emits LESS ({total_rough_current/total_smooth_current*100:.1f}%) than smooth in LUT integral.")
    print("This causes norm_factor > 1 → LUT values scaled UP → rough always appears hotter.")
else:
    print(f"Rough emits MORE ({total_rough_current/total_smooth_current*100:.1f}%) than smooth.")
