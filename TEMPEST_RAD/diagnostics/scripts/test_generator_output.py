#!/usr/bin/env python3
"""
Quick test of generator output formatting with reduced cases.
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/duncan/Desktop/DPhil/TEMPEST')

import TEMPEST_RAD.generator as gen

# Temporarily reduce to just 2 Theta × 2 Latitude = 4 cases
gen.THETA_VALUES = np.array([0.5, 1.0])
gen.LATITUDE_VALUES = np.array([0.0, 45.0])

# Run generator
if __name__ == '__main__':
    gen.main()
