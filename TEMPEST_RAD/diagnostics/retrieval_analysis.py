
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

# Ensure we can import from TEMPEST_RAD and src
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.utilities.config import Config
from src.utilities.utils import rotate_vector
from TEMPEST_RAD.simulator import (
    load_shape_model, 
    compute_geometry, 
    RoughnessLUT, 
    planck_function, 
    rms_to_fraction, 
    calculate_theta
)

# =============================================================================
# 1. SETUP & RUN TEMPEST
# =============================================================================
# Define paths
CONFIG_PATH = os.path.join(root_dir, "private/data/config/moon/moon_config.yaml")
OUTPUT_DIR = os.path.join(root_dir, "output/retrieval_analysis")
LUT_PATH = os.path.join(root_dir, "roughness_lut_spectral_v1.h5")

print(f"--- Step 1: Running TEMPEST Simulation ---")
print(f"Config: {CONFIG_PATH}")
print(f"Output: {OUTPUT_DIR}")

# Run TEMPEST (Standard Solver) to get smooth temperatures
# tempest.py does not support --out, so we have to find where it saves
cmd = [
    "python", "tempest.py",
    "--config", CONFIG_PATH
]

# Check if output exists to avoid re-running unnecessarily (optional, but good for dev)
# We need to find the output folder dynamically if we run it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find latest run
def get_latest_run_dir():
    base_out = os.path.join(root_dir, "output")
    runs = [os.path.join(base_out, d) for d in os.listdir(base_out) if d.startswith("run_")]
    if not runs: return None
    return max(runs, key=os.path.getmtime)

print("Checking for existing TEMPEST runs...")
latest_run = get_latest_run_dir()

if latest_run and os.path.exists(os.path.join(latest_run, "temperatures.csv")):
    print(f"Using existing run: {latest_run}")
    # Copy to our analysis folder for consistency
    import shutil
    shutil.copy(os.path.join(latest_run, "temperatures.csv"), os.path.join(OUTPUT_DIR, "temperatures.csv"))
else:
    print("Running TEMPEST... (this may take a moment)")
    subprocess.run(cmd, check=True)
    # Get the new latest run
    latest_run = get_latest_run_dir()
    print(f"New run created at: {latest_run}")
    import shutil
    shutil.copy(os.path.join(latest_run, "temperatures.csv"), os.path.join(OUTPUT_DIR, "temperatures.csv"))

# =============================================================================
# 2. LOAD DATA
# =============================================================================
print(f"--- Step 2: Loading Data ---")
config = Config(CONFIG_PATH)
facets, mesh = load_shape_model(config.path_to_shape_model_file)
n_facets = len(facets)

# Load Temperatures
temps_path = os.path.join(OUTPUT_DIR, "temperatures.csv")
try:
    temps_all = np.loadtxt(temps_path, delimiter=',')
except ValueError:
    temps_all = np.loadtxt(temps_path, delimiter=',', skiprows=1)

# Handle Transpose if needed
if temps_all.shape[0] != n_facets and temps_all.shape[1] == n_facets:
    temps_all = temps_all.T

print(f"Loaded temperatures: {temps_all.shape}")

# Load LUT
theta = calculate_theta(config)
print(f"Calculated Theta: {theta:.3f}")
lut = RoughnessLUT(LUT_PATH, target_theta=theta, target_rms=90.0) # Load spectral LUT

# Helper to get temps at a specific time
def get_temps_at_time(time_hours):
    period = getattr(config, 'rotation_period_hours', 24.0)
    n_steps = temps_all.shape[1]
    idx = int((time_hours % period) / period * n_steps)
    idx = np.clip(idx, 0, n_steps - 1)
    return temps_all[:, idx]

# =============================================================================
# 3. ANALYSIS: BRIGHTNESS TEMP MAP (SMOOTH vs ROUGH)
# =============================================================================
print(f"--- Step 3: Generating Brightness Temperature Maps ---")

def calculate_bolometric_tb(time_hours, roughness_rms, phase_angle):
    # Geometry
    sun_vec = np.array(config.sunlight_direction)
    rot_axis = np.array([0, 0, 1]) # Simplified
    
    # Observer Vector
    perp_vec = np.cross(sun_vec, rot_axis)
    if np.linalg.norm(perp_vec) < 1e-6: perp_vec = np.array([0, 1, 0])
    obs_vec = rotate_vector(sun_vec, perp_vec, np.radians(phase_angle))
    
    lats, phases, emis, azis = compute_geometry(facets, sun_vec, obs_vec, rot_axis)
    
    # Temps
    temps_smooth = get_temps_at_time(time_hours)
    
    # Roughness
    f = rms_to_fraction(roughness_rms)
    
    # Wavelength Loop
    wavelengths = lut.axes['wavelength']
    full_spectra = np.zeros((n_facets, len(wavelengths)))
    
    for i, wave in enumerate(wavelengths):
        rad_smooth = planck_function(wave, temps_smooth)
        factors = lut.get_correction_factors(lats, phases, emis, azis, wavelength=wave)
        full_spectra[:, i] = rad_smooth * ((1.0 - f) + f * factors)
        
    # Integrate
    rad_bol = np.trapz(full_spectra, x=wavelengths, axis=1)
    sigma = 5.670374419e-8
    t_eff = (np.pi * rad_bol / sigma) ** 0.25
    
    # Mask hidden
    t_eff[emis > 90] = 0
    
    return t_eff, obs_vec, emis

# Compare at Time=12h, Phase=30
time_tgt = 12.0
phase_tgt = 30.0

tb_smooth, obs_vec, emis = calculate_bolometric_tb(time_tgt, 0.0, phase_tgt)
tb_rough, _, _ = calculate_bolometric_tb(time_tgt, 28.0, phase_tgt)

# Projection for plotting
def project_coords(obs_vec):
    obs_n = obs_vec / np.linalg.norm(obs_vec)
    up = np.array([0, 0, 1])
    if abs(np.dot(up, obs_n)) > 0.9: up = np.array([1, 0, 0])
    u = np.cross(obs_n, up); u /= np.linalg.norm(u)
    v = np.cross(obs_n, u)
    centers = np.array([f.center for f in facets])
    return np.dot(centers, u), np.dot(centers, v)

u, v = project_coords(obs_vec)
mask = emis < 90

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot Smooth
sc1 = axes[0].scatter(u[mask], v[mask], c=tb_smooth[mask], cmap='inferno', s=2)
axes[0].set_title(f"Smooth Model (RMS=0°)\nBolometric Tb")
plt.colorbar(sc1, ax=axes[0], label="K")
axes[0].axis('equal')

# Plot Rough
sc2 = axes[1].scatter(u[mask], v[mask], c=tb_rough[mask], cmap='inferno', s=2)
axes[1].set_title(f"Rough Model (RMS=28°)\nBolometric Tb")
plt.colorbar(sc2, ax=axes[1], label="K")
axes[1].axis('equal')

# Plot Difference
diff = tb_rough - tb_smooth
vmax = np.max(np.abs(diff[mask]))
sc3 = axes[2].scatter(u[mask], v[mask], c=diff[mask], cmap='coolwarm', vmin=-vmax, vmax=vmax, s=2)
axes[2].set_title(f"Difference (Rough - Smooth)")
plt.colorbar(sc3, ax=axes[2], label="Delta K")
axes[2].axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_comparison.png"))
print("Saved map_comparison.png")

# =============================================================================
# 4. ANALYSIS: SPECTRA (BLACKBODY VS ROUGH)
# =============================================================================
print(f"--- Step 4: Spectral Analysis ---")

# Pick a warm facet (e.g., closest to sub-solar point)
sun_vec = np.array(config.sunlight_direction)
centers = np.array([f.center for f in facets])
dots = np.dot(centers / np.linalg.norm(centers, axis=1)[:, None], sun_vec / np.linalg.norm(sun_vec))
target_facet_idx = np.argmax(dots)

print(f"Selected Facet {target_facet_idx} (Sub-solar alignment: {dots[target_facet_idx]:.3f})")

# Calculate Spectra
wavelengths = lut.axes['wavelength']
temps_smooth_val = get_temps_at_time(time_tgt)[target_facet_idx]

# Smooth Spectrum (Planck)
spec_smooth = planck_function(wavelengths, temps_smooth_val)

# Rough Spectrum
# Need geometry for this specific facet
lats, phases, emis, azis = compute_geometry([facets[target_facet_idx]], sun_vec, obs_vec, np.array([0,0,1]))
f = rms_to_fraction(28.0)

spec_rough = np.zeros_like(spec_smooth)
for i, wave in enumerate(wavelengths):
    rad_s = planck_function(wave, temps_smooth_val)
    factor = lut.get_correction_factors(lats, phases, emis, azis, wavelength=wave)[0]
    spec_rough[i] = rad_s * ((1.0 - f) + f * factor)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, spec_smooth, 'o--', label=f'Smooth (T={temps_smooth_val:.1f}K)', color='blue')
plt.plot(wavelengths, spec_rough, 's-', label=f'Rough (RMS=28°)', color='red')

# Fit effective temperature to rough spectrum?
# Just plotting for now
plt.xlabel("Wavelength (microns)")
plt.ylabel("Radiance (W/m2/sr/um)")
plt.title(f"Spectral Radiance Comparison (Facet {target_facet_idx})\nPhase: {phase_tgt}°, Emission: {emis[0]:.1f}°")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "spectrum_comparison.png"))
print("Saved spectrum_comparison.png")

# =============================================================================
# 5. ANALYSIS: PHASE CURVE VARIATION
# =============================================================================
print(f"--- Step 5: Phase Curve Analysis ---")

phases_to_test = [0, 30, 60, 90, 120]
mean_temps_smooth = []
mean_temps_rough = []

for p in phases_to_test:
    # Calculate maps
    tb_s, _, emi_s = calculate_bolometric_tb(time_tgt, 0.0, p)
    tb_r, _, emi_r = calculate_bolometric_tb(time_tgt, 28.0, p)
    
    # Average over visible disk (simple mean for now, ideally area-weighted)
    mask_s = emi_s < 90
    mask_r = emi_r < 90
    
    mean_temps_smooth.append(np.mean(tb_s[mask_s]))
    mean_temps_rough.append(np.mean(tb_r[mask_r]))

plt.figure(figsize=(8, 5))
plt.plot(phases_to_test, mean_temps_smooth, 'o-', label='Smooth')
plt.plot(phases_to_test, mean_temps_rough, 's-', label='Rough (RMS=28°)')
plt.xlabel("Phase Angle (deg)")
plt.ylabel("Mean Disk Brightness Temp (K)")
plt.title("Bolometric Phase Curve (Mean Visible Temp)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "phase_curve.png"))
print("Saved phase_curve.png")

# =============================================================================
# 6. SIMULATED INSTRUMENT OBSERVATION (FOV)
# =============================================================================
print(f"--- Step 6: Simulated Instrument Observation ---")

# Define Instrument
inst_dist = 2000.0 # km
inst_phase = 45.0
fov_deg = 5.0 # Narrow FOV

# Calculate Position
sun_vec = np.array(config.sunlight_direction)
rot_axis = np.array([0, 0, 1])
perp_vec = np.cross(sun_vec, rot_axis)
if np.linalg.norm(perp_vec) < 1e-6: perp_vec = np.array([0, 1, 0])
obs_dir = rotate_vector(sun_vec, perp_vec, np.radians(inst_phase))
obs_dir /= np.linalg.norm(obs_dir)
inst_pos = obs_dir * inst_dist * 1000 # meters

# Boresight (looking at center)
boresight = -obs_dir

# Find facets in FOV
centers = np.array([f.center for f in facets]) # meters
vecs_to_facets = centers - inst_pos # Vector from Inst to Facet
dists = np.linalg.norm(vecs_to_facets, axis=1)
vecs_norm = vecs_to_facets / dists[:, None]

# Angle from boresight
# dot(v, boresight) = cos(theta)
cos_thetas = np.dot(vecs_norm, boresight)
thetas = np.degrees(np.arccos(np.clip(cos_thetas, -1.0, 1.0)))

in_fov = (thetas < (fov_deg / 2.0))

# Check visibility (Emission < 90)
# Need normal vectors
normals = np.array([f.normal for f in facets])
# View vector is -vecs_norm (Facet to Inst)
cos_emis = np.einsum('ij,ij->i', normals, -vecs_norm)
visible = cos_emis > 0

valid_mask = in_fov & visible
valid_indices = np.where(valid_mask)[0]

print(f"Instrument FOV: {fov_deg}° at {inst_dist}km")
print(f"Facets in FOV and Visible: {len(valid_indices)}")

if len(valid_indices) > 0:
    # Integrate Spectrum
    # Flux Density = Sum ( Radiance * Solid_Angle )
    # Solid Angle = Area * cos(emi) / dist^2
    
    areas = np.array([f.area for f in facets])[valid_mask]
    dists_sq = dists[valid_mask]**2
    cos_e = cos_emis[valid_mask]
    solid_angles = areas * cos_e / dists_sq
    
    # Get Rough Radiance for these facets
    # Re-use logic (simplified for this block)
    lats, phases, emis, azis = compute_geometry(
        [facets[i] for i in valid_indices], 
        sun_vec, obs_dir, rot_axis
    )
    
    temps_fov = get_temps_at_time(time_tgt)[valid_mask]
    
    obs_spectrum = np.zeros(len(wavelengths))
    
    for i, wave in enumerate(wavelengths):
        rad_s = planck_function(wave, temps_fov)
        factors = lut.get_correction_factors(lats, phases, emis, azis, wavelength=wave)
        rad_r = rad_s * ((1.0 - f) + f * factors)
        
        # Sum (Radiance * Solid Angle)
        obs_spectrum[i] = np.sum(rad_r * solid_angles)
        
    # Plot Observed Spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, obs_spectrum, 'o-', color='purple')
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Spectral Irradiance at Instrument (W/m2/um)")
    plt.title(f"Simulated Observation (FOV={fov_deg}°)\nPhase: {inst_phase}°, Distance: {inst_dist}km")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "instrument_spectrum.png"))
    print("Saved instrument_spectrum.png")
else:
    print("No facets visible in FOV.")

print("\n--- Analysis Complete ---")
