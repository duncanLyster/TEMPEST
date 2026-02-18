#!/usr/bin/env python
"""Debug visibility and beaming calculation."""
import numpy as np
import sys
sys.path.append('.')

from TEMPEST_RAD.generator import process_single_case, ReferenceConfig

c = ReferenceConfig()
print('Testing zero-phase viewing geometry (Theta=1.0, Lat=0°)...\n')

result = process_single_case(1.0, 90.0, 0.0, c)

print(f'\nResult shape: {result.shape}')
print(f'\nCorrection factors at t=0, emission=0°, azimuth=0° (subsolar point):')
for i, wave in enumerate([5.0, 8.0, 15.0, 50.0, 100.0]):
    print(f'  λ={wave:5.1f}μm: {result[0, i, 0, 0]:.4f}')

print(f'\nExpected: Factors > 1.0 at zero phase (beaming)')
