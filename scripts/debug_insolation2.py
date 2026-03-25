#!/usr/bin/env python3
"""Verify whether TEMPEST exports absorbed or total insolation."""
import numpy as np

A = 0.016

# Solar flux values
L = 3.826e26
r_au_tempest = 0.896
r_m_tempest = r_au_tempest * 1.496e11
base_flux_tempest = L / (4 * np.pi * r_m_tempest**2)

solarLum_btpm = 3.826e26 / (4 * np.pi)
r_m_btpm = 1.34e11
base_flux_btpm = solarLum_btpm / r_m_btpm**2

absorbed_btpm = (1 - A) * base_flux_btpm

print("=== Theoretical ratios ===")
print(f"base_flux_tempest / absorbed_btpm = {base_flux_tempest / absorbed_btpm:.6f}")
print(f"  (This is the ratio if TEMPEST does NOT apply (1-A))")
print(f"(1-A)*base_flux_tempest / absorbed_btpm = {(1-A)*base_flux_tempest / absorbed_btpm:.6f}")
print(f"  (This is the ratio if TEMPEST DOES apply (1-A))")
print(f"Observed ratio from data: 1.0157")
print()

# The 1.0156 ratio matches "no albedo" almost exactly
# This means TEMPEST exports total incident flux (not absorbed)
# BUT the solver DOES use (1-A) internally. The CSV is just the export format.

# The more important question: does the SOLVER use the right flux?
# Let's check the actual source
print("=== Checking TEMPEST insolation source code ===")
import os
insol_file = 'src/model/insolation.py'
if os.path.exists(insol_file):
    with open(insol_file) as f:
        lines = f.readlines()
    # Find lines with albedo or (1-A) or insolation calculation
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['albedo', '1 - a', '(1-a', 'absorbed', 'csv', 'export', 'save_insol']):
            print(f"  L{i+1}: {line.rstrip()}")

# Also check what the solver actually receives
print()
solver_file = 'src/model/solvers/tempest_standard.py'
if os.path.exists(solver_file):
    with open(solver_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['insolation', 'albedo', 'solar', 'absorbed']):
            print(f"  L{i+1}: {line.rstrip()}")
