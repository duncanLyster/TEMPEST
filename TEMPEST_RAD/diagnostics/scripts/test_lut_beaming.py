#!/usr/bin/env python
"""Test LUT correction factors at zero phase angle."""

import numpy as np
import sys
sys.path.append('.')
from TEMPEST_RAD.simulator import RoughnessLUT, calculate_theta
from src.utilities.config import Config

# Load config and LUT
config = Config('private/data/config/moon/moon_config.yaml')
theta = calculate_theta(config)
lut = RoughnessLUT('roughness_lut_spectral_v1.h5', target_theta=theta, target_rms=90.0)

# Zero phase angle scenario (thermal beaming expected)
print(f'Testing zero-phase scenario (thermal beaming expected)')
print(f'Theta: {theta:.3f}\n')

wavelengths = lut.axes['wavelength']
print('Correction factors at zero phase (lat=0, phase=0, emis=0, azi=0):')
for wave in wavelengths:
    factor = lut.get_correction_factors(
        lat=np.array([0.0]), 
        phase=np.array([0.0]), 
        emission=np.array([0.0]), 
        azimuth=np.array([0.0]),
        wavelength=wave
    )[0]
    status = "WARM ✓" if factor > 1.0 else "COOL ✗"
    print(f'  λ={wave:.1f}μm: factor={factor:.4f} ({status})')

print('\nExpected: All factors > 1.0 (rough surface beams emission back at observer)')
print('If factors < 1.0, the rough model will appear COOLER (incorrect physics)')
