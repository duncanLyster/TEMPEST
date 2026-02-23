"""
Quick timing test: runs a single latitude through process_single_case
with 2000 subfacets to get a realistic per-case timing estimate.
"""
import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Override mesh resolution for this test BEFORE importing generator
import generator
generator.CRATER_SUBFACETS = 2000
generator.VIEW_FACTOR_RAYS = 20000

from generator import (
    ReferenceConfig, compute_visibility_mask, process_single_case,
    OPENING_ANGLES, Facet, Simulation,
    calculate_rotation_matrix, calculate_all_view_factors,
    calculate_thermal_view_factors, ThermalData,
    LUT_TIMESTEPS, WAVELENGTHS_MICRONS, EMISSION_ANGLES, AZIMUTH_ANGLES,
    THETA_VALUES, LATITUDE_VALUES
)
import numpy as np

print(f"CRATER_SUBFACETS = {generator.CRATER_SUBFACETS}")
print(f"VIEW_FACTOR_RAYS = {generator.VIEW_FACTOR_RAYS}")
print(f"LUT grid: {LUT_TIMESTEPS} time × {len(WAVELENGTHS_MICRONS)} wave × "
      f"{len(EMISSION_ANGLES)} emi × {len(AZIMUTH_ANGLES)} azi")

config = ReferenceConfig()
config.silent_mode = False

# === Step 1: Generate mesh & view factors ===
print("\n--- Step 1: Mesh + View Factors ---")
t0 = time.time()

dummy_normal = np.array([0.0, 0.0, 1.0])
dummy_vs = [np.array([-1,-1,0]), np.array([1,-1,0]), np.array([0,1,0])]
dummy_facet = Facet(dummy_normal, dummy_vs, 1, 1, 1, False)
config.kernel_profile_angle_degrees = OPENING_ANGLES[0]
config.apply_kernel_based_roughness = True
dummy_facet.generate_spherical_depression(config, Simulation(config))

mesh = Facet._canonical_subfacet_mesh
n_facets = len(mesh)
print(f"  Actual subfacets: {n_facets}")

rotation_to_equator = calculate_rotation_matrix(np.array([0.0, 1.0, 0.0]), np.pi/2)
shape_model = []
for entry in mesh:
    rotated_n = np.dot(rotation_to_equator, np.array(entry['normal']))
    rotated_v = np.array([np.dot(rotation_to_equator, v) for v in entry['vertices']])
    shape_model.append(Facet(rotated_n, rotated_v, 1, 1, 1, False))

thermal_data = ThermalData(n_facets, 1, 1, 1, False)
all_indices = np.arange(n_facets)
visible_facets_list = [np.concatenate([all_indices[:i], all_indices[i+1:]]) for i in range(n_facets)]
thermal_data.set_visible_facets(visible_facets_list)

print(f"  Computing view factors...")
calculate_all_view_factors(shape_model, thermal_data, config, generator.VIEW_FACTOR_RAYS)
calculate_thermal_view_factors(shape_model, thermal_data, config)
t_vf = time.time() - t0
print(f"  VF total: {t_vf:.1f}s")

# === Step 2: Build precomputed dict (same as main()) ===
print("\n--- Step 2: Precomputed geometry ---")
canonical_normals = np.array([e['normal'] for e in mesh], dtype=np.float64)
canonical_vertices = np.array([e['vertices'] for e in mesh], dtype=np.float64)
canonical_areas = np.array([e['area'] for e in mesh], dtype=np.float64)
canonical_centers = np.array([np.mean(e['vertices'], axis=0) for e in mesh], dtype=np.float64)

rotated_normals = np.array([np.dot(rotation_to_equator, e['normal']) for e in mesh], dtype=np.float64)
rotated_vertices = np.array([[np.dot(rotation_to_equator, v) for v in e['vertices']] for e in mesh], dtype=np.float64)

precomputed = {
    'canonical_normals': canonical_normals,
    'canonical_vertices': canonical_vertices,
    'canonical_areas': canonical_areas,
    'canonical_centers': canonical_centers,
    'rotated_normals': rotated_normals,
    'rotated_vertices': rotated_vertices,
    'rotation_to_equator': rotation_to_equator,
    'world_to_local': rotation_to_equator.T,
    'aperture_normal': np.array([1.0, 0.0, 0.0]),
    'aperture_area': 1.0,
    'n_facets': n_facets,
}

# JIT warmup
print("  Compiling numba kernel...")
_d = canonical_normals[0] / np.linalg.norm(canonical_normals[0])
_ = compute_visibility_mask(_d, canonical_normals, canonical_centers, canonical_vertices)
print("  Ready.")

# Suppress output during the actual run
config.silent_mode = True

# === Step 3: Benchmark visibility kernel at 2002 facets ===
print("\n--- Step 3: Visibility kernel timing ---")
_d = canonical_normals[0] / np.linalg.norm(canonical_normals[0])
_ = compute_visibility_mask(_d, canonical_normals, canonical_centers, canonical_vertices)
print("  JIT ready.")

n_test = 100
t0 = time.time()
for _ in range(n_test):
    th = np.random.uniform(0, np.pi/3)
    ph = np.random.uniform(0, 2*np.pi)
    d = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    compute_visibility_mask(d, canonical_normals, canonical_centers, canonical_vertices)
t_avg = (time.time() - t0) / n_test
print(f"  {n_facets} subfacets: {t_avg*1000:.1f}ms/call (avg of {n_test})")

# === Step 4: Extrapolate ===
n_lats = len(LATITUDE_VALUES)
n_thetas = len(THETA_VALUES)
n_dirs_per_lat = LUT_TIMESTEPS * len(EMISSION_ANGLES) * len(AZIMUTH_ANGLES)
t_viewing_per_lat = n_dirs_per_lat * t_avg

print(f"\n--- Projected Full Run ({n_facets} subfacets) ---")
print(f"  Visibility calls/latitude: {n_dirs_per_lat:,}")
print(f"  Viewing time/latitude: {t_viewing_per_lat/60:.1f} min")

for label, cores in [("Server 32 cores", 32), ("Local 6 cores", 6)]:
    batches = int(np.ceil(n_lats / cores))
    per_theta = batches * t_viewing_per_lat
    total = n_thetas * per_theta
    print(f"  {label}: {per_theta/60:.0f} min/theta → {total/3600:.1f} hours total (viewing only)")

print(f"\n  (VF precomputation: {t_vf/60:.0f} min one-off at {n_facets} facets with 8 cores)")
print(f"  (Thermal solver adds ~10-30% on top of viewing time)")
