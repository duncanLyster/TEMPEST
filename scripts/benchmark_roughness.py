#!/usr/bin/env python3
"""
Benchmark script for measuring the time to generate spherical cap meshes for different subfacet counts.
"""
import time
from contextlib import redirect_stdout
from io import StringIO
from src.model.spherical_cap_mesh import generate_canonical_spherical_cap

if __name__ == "__main__":
    counts = [10, 30, 100, 300, 500]
    print("Profiling generate_canonical_spherical_cap for various subfacet counts:")
    for n in counts:
        buf = StringIO()
        t0 = time.time()
        # Silence internal prints and progress bars
        with redirect_stdout(buf):
            generate_canonical_spherical_cap(n, 90.0)
        dt = time.time() - t0
        print(f"n_subfacets={n}: {dt:.4f} seconds") 