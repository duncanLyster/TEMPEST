#!/usr/bin/env python3
"""
Benchmark script for measuring the time to compute canonical view factor matrices for a given configuration.
"""
import time
from contextlib import redirect_stdout
from io import StringIO
from src.utilities.config import Config
from src.model.simulation import Simulation
from src.model.facet import Facet
from src.model.spherical_cap_mesh import generate_canonical_spherical_cap

if __name__ == "__main__":
    # Load configuration
    cfg = Config("data/config/example_config.yaml")
    sim = Simulation(cfg)
    # Pre-generate canonical meshes
    # Silence prints during mesh generation
    buf = StringIO()
    with redirect_stdout(buf):
        Facet._canonical_subfacet_mesh = generate_canonical_spherical_cap(
            cfg.kernel_subfacets_count,
            cfg.kernel_profile_angle_degrees
        )
        Facet._canonical_dome_mesh = generate_canonical_spherical_cap(
            cfg.kernel_directional_bins,
            90.0
        )
    print(f"Canonical subfacets: {len(Facet._canonical_subfacet_mesh)}, dome facets: {len(Facet._canonical_dome_mesh)}")
    # Benchmark view factor computation
    print("Benchmarking canonical view factor computation...")
    t0 = time.time()
    F_ss, F_sd = Facet._compute_canonical_view_factors(cfg)
    dt = time.time() - t0
    print(f"Computed F_ss {F_ss.shape} and F_sd {F_sd.shape} in {dt:.4f} seconds") 