#!/usr/bin/env python3
"""
Benchmark timing for a single LUT case to estimate weekend run feasibility.

Runs a single (theta, latitude) case and reports:
- View factor calculation time (one-off cost)
- Thermal solver time per case
- Viewing geometry computation time per case
- Estimated total time for full LUT

Usage:
    ./venv/bin/python3 TEMPEST_RAD/benchmark_timing.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Setup paths
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

# Import the generator module 
from TEMPEST_RAD.generator import (
    ReferenceConfig, simulate_crater_diurnal_cycle, process_single_case,
    Facet, Simulation, ThermalData,
    calculate_all_view_factors, calculate_thermal_view_factors,
    calculate_rotation_matrix, plot_wireframe,
    THETA_VALUES, OPENING_ANGLES, WAVELENGTHS_MICRONS,
    LATITUDE_VALUES, EMISSION_ANGLES, AZIMUTH_ANGLES,
    LUT_TIMESTEPS, SIM_TIMESTEPS, CRATER_SUBFACETS, VIEW_FACTOR_RAYS
)

def estimate_file_size(n_theta, n_lat, n_time, n_wave, n_emi, n_azi):
    """Calculate file size in MB for one theta slice (float32)."""
    n_cells = n_lat * n_time * n_wave * n_emi * n_azi
    size_bytes = n_cells * 4  # float32
    return size_bytes / (1024**2)

def print_parameter_options():
    """Print file size estimates for various parameter combinations."""
    print("\n" + "="*80)
    print("FILE SIZE ESTIMATES (per theta value, float32)")
    print("="*80)
    
    options = [
        ("Current",     37, 90,  5,  20, 20),
        ("Option A",    37, 180, 10, 30, 30),
        ("Option B",    37, 180, 12, 36, 36),
        ("Option C",    37, 180, 15, 30, 30),
        ("Option D",    37, 360, 10, 30, 30),
        ("Option E",    37, 180, 12, 30, 30),
        ("Option F",    37, 180, 15, 36, 36),
    ]
    
    print(f"{'Name':<12} {'Lat':>4} {'Time':>5} {'Wave':>5} {'Emi':>4} {'Azi':>4} {'Cells':>12} {'Size MB':>10}")
    print("-"*60)
    for name, n_lat, n_time, n_wave, n_emi, n_azi in options:
        size = estimate_file_size(1, n_lat, n_time, n_wave, n_emi, n_azi)
        cells = n_lat * n_time * n_wave * n_emi * n_azi
        marker = " <-- OVER 500MB" if size > 500 else ""
        print(f"{name:<12} {n_lat:>4} {n_time:>5} {n_wave:>5} {n_emi:>4} {n_azi:>4} {cells:>12,} {size:>9.1f}{marker}")

def benchmark_single_case(n_subfacets=None):
    """Benchmark a single case and estimate total LUT time."""
    
    import TEMPEST_RAD.generator as gen
    
    # Optionally override subfacet count for testing
    if n_subfacets is not None:
        original_subfacets = gen.CRATER_SUBFACETS
        gen.CRATER_SUBFACETS = n_subfacets
        print(f"  [Override] CRATER_SUBFACETS: {original_subfacets} -> {n_subfacets}")
    
    config = ReferenceConfig()
    test_theta = 0.5  # mid-range value
    test_lat = 0.0    # equator (typically slowest due to large diurnal signal)
    
    # --- Phase 1: View Factor Setup ---
    print("\n--- Phase 1: View Factor Calculation (one-off) ---")
    
    # Clear cached view factors to force recalculation
    # (comment out if you want to test with cache)
    
    t0 = time.time()
    
    # Generate crater mesh
    dummy_normal = np.array([0.0, 0.0, 1.0])
    dummy_vs = [np.array([-1,-1,0]), np.array([1,-1,0]), np.array([0,1,0])]
    dummy_facet = Facet(dummy_normal, dummy_vs, 1, 1, 1, False)
    config.kernel_profile_angle_degrees = 90.0
    config.apply_kernel_based_roughness = True
    
    sim = Simulation(config)
    dummy_facet.generate_spherical_depression(config, sim)
    mesh = Facet._canonical_subfacet_mesh
    n_facets = len(mesh)
    
    t_mesh = time.time() - t0
    print(f"  Mesh generation: {t_mesh:.1f}s ({n_facets} facets generated)")
    
    # Build shape model for view factor pre-calc
    rotation_to_equator = calculate_rotation_matrix(np.array([0.0, 1.0, 0.0]), np.pi/2)
    shape_model = []
    for entry in mesh:
        canonical_n = np.array(entry['normal'])
        canonical_v = np.array(entry['vertices'])
        rotated_n = np.dot(rotation_to_equator, canonical_n)
        rotated_v = np.array([np.dot(rotation_to_equator, v) for v in canonical_v])
        new_f = Facet(rotated_n, rotated_v, 1, 1, 1, False)
        shape_model.append(new_f)
    
    thermal_data = ThermalData(n_facets, 1, 1, 1, False)
    all_indices = np.arange(n_facets)
    visible_facets_list = [np.concatenate([all_indices[:i], all_indices[i+1:]]) for i in range(n_facets)]
    thermal_data.set_visible_facets(visible_facets_list)
    
    t0_vf = time.time()
    calculate_all_view_factors(shape_model, thermal_data, config, gen.VIEW_FACTOR_RAYS)
    t_vf = time.time() - t0_vf
    print(f"  View factor calculation: {t_vf:.1f}s (or loaded from cache)")
    
    t0_tvf = time.time()
    calculate_thermal_view_factors(shape_model, thermal_data, config)
    t_tvf = time.time() - t0_tvf
    print(f"  Thermal view factors: {t_tvf:.1f}s (or loaded from cache)")
    
    total_setup = t_mesh + t_vf + t_tvf
    print(f"  TOTAL setup (one-off): {total_setup:.1f}s")
    
    # --- Phase 2: Single Case ---
    print(f"\n--- Phase 2: Single Case (theta={test_theta}, lat={test_lat}) ---")
    
    t0_case = time.time()
    result_grid, norm_factor = process_single_case(test_theta, 90.0, test_lat, config)
    t_case = time.time() - t0_case
    
    if np.isnan(norm_factor):
        print(f"  ERROR: Case failed!")
    else:
        print(f"  Case completed: {t_case:.1f}s")
        print(f"  Normalization factor: {norm_factor:.4f}")
        print(f"  Result shape: {result_grid.shape}")
        print(f"  Mean ratio: {np.nanmean(result_grid):.4f}")
        print(f"  NaN fraction: {np.sum(np.isnan(result_grid))/result_grid.size:.4f}")
    
    # --- Phase 3: Timing Estimates ---
    print("\n" + "="*80)
    print("TIMING ESTIMATES")
    print("="*80)
    
    n_lat = len(gen.LATITUDE_VALUES)
    
    print(f"\nMeasured time per case: {t_case:.1f}s ({t_case/60:.1f} min)")
    print(f"Number of latitudes:    {n_lat}")
    print(f"Cases per theta:        {n_lat}")
    print(f"One-off setup cost:     {total_setup:.1f}s")
    
    # Timing for different parallelism levels
    print(f"\nEstimated time PER THETA VALUE:")
    for n_jobs in [1, 4, 8, 16, 32]:
        serial_total = n_lat * t_case
        parallel_total = serial_total / n_jobs
        print(f"  {n_jobs:>2} cores: {parallel_total/60:.1f} min ({parallel_total/3600:.1f} hrs)")
    
    # Weekend budget
    weekend_hours = 60  # conservative: Fri evening to Mon morning
    print(f"\nNumber of THETA values completable in {weekend_hours}h weekend:")
    for n_jobs in [1, 4, 8, 16, 32]:
        time_per_theta = (n_lat * t_case) / n_jobs
        n_theta = int(weekend_hours * 3600 / time_per_theta)
        print(f"  {n_jobs:>2} cores: ~{n_theta} theta values ({time_per_theta/3600:.1f} hrs each)")
    
    # Restore
    if n_subfacets is not None:
        gen.CRATER_SUBFACETS = original_subfacets
    
    return t_case, total_setup, n_facets


def benchmark_subfacet_scaling():
    """Test how timing scales with subfacet count."""
    import TEMPEST_RAD.generator as gen
    
    print("\n" + "="*80)
    print("SUBFACET COUNT SCALING TEST")
    print("="*80)
    print("(Each test runs one case; view factors recalculated each time)\n")
    
    # Clear cached mesh to force regeneration
    subfacet_counts = [200, 500, 1000, 2000]
    results = []
    
    for n_sub in subfacet_counts:
        # Reset canonical mesh
        Facet._canonical_subfacet_mesh = None
        Facet._canonical_normals = None
        Facet._canonical_areas = None
        Facet._canonical_subfacet_triangles = None
        Facet._canonical_subfacet_centers = None
        
        print(f"\nTesting CRATER_SUBFACETS = {n_sub}...")
        gen.CRATER_SUBFACETS = n_sub
        
        config = ReferenceConfig()
        config.config_data['kernel_subfacets_count'] = n_sub
        
        t0 = time.time()
        result_grid, norm_factor = process_single_case(0.5, 90.0, 0.0, config)
        elapsed = time.time() - t0
        
        actual_facets = len(Facet._canonical_subfacet_mesh) if Facet._canonical_subfacet_mesh else 0
        results.append((n_sub, actual_facets, elapsed, norm_factor))
        print(f"  Actual facets: {actual_facets}, Time: {elapsed:.1f}s, Norm: {norm_factor:.4f}")
    
    print(f"\n{'Target':>8} {'Actual':>8} {'Time (s)':>10} {'Norm Factor':>12}")
    print("-"*45)
    for target, actual, elapsed, norm in results:
        print(f"{target:>8} {actual:>8} {elapsed:>10.1f} {norm:>12.4f}")
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("TEMPEST LUT GENERATOR - BENCHMARKING & PLANNING")
    print("="*80)
    
    # Show file size options
    print_parameter_options()
    
    # Ask user what to do
    print("\n" + "="*80)
    print("BENCHMARK OPTIONS:")
    print("  1. Quick: single case with current settings")
    print("  2. Scaling: test multiple subfacet counts (slower)")
    print("  3. Both")
    print("="*80)
    
    choice = input("\nChoice [1/2/3, default=1]: ").strip() or "1"
    
    if choice in ["1", "3"]:
        print(f"\nRunning single-case benchmark...")
        print(f"  Current CRATER_SUBFACETS = {CRATER_SUBFACETS}")
        print(f"  Current SIM_TIMESTEPS = {SIM_TIMESTEPS}")
        t_case, t_setup, n_facets = benchmark_single_case()
    
    if choice in ["2", "3"]:
        benchmark_subfacet_scaling()
    
    print("\nDone!")
