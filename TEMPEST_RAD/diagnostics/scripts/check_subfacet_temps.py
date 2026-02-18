#!/usr/bin/env python
"""Check subfacet temperatures and visibility at zero phase."""
import numpy as np
import sys
sys.path.append('.')

from TEMPEST_RAD.generator import simulate_crater_diurnal_cycle, ReferenceConfig

c = ReferenceConfig()
rough_temps, smooth_temps, _, _ = simulate_crater_diurnal_cycle(1.0, 90.0, 0.0, c, 180)

print('Analyzing subfacet temperatures at t=0 (subsolar):')
print(f'  All subfacets: min={rough_temps[:, 0].min():.1f}, max={rough_temps[:, 0].max():.1f}, mean={rough_temps[:, 0].mean():.1f} K')
print(f'  Smooth ref: {smooth_temps[0]:.1f} K')
print()

# Check distribution
bins = [0, 250, 275, 300, 325, 350, 400]
hist, _ = np.histogram(rough_temps[:, 0], bins=bins)
print('Temperature distribution at t=0:')
for i in range(len(bins)-1):
    print(f'  {bins[i]:.0f}-{bins[i+1]:.0f} K: {hist[i]:3d} subfacets ({100*hist[i]/len(rough_temps[:, 0]):.1f}%)')
print()

# The question is: at zero phase, WHICH subfacets are visible from above?
# If we see mostly the HOT ones (300+ K), beaming should work
# If we see a mix including cold shadows (< 250 K), it won't

print('For beaming to work, the VISIBLE subfacets (from above) should be mostly 300+ K')
print('If visibility calculation includes cold inner walls, beaming fails')
