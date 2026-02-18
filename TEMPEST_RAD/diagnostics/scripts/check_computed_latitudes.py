#!/usr/bin/env python3
"""
Check what latitudes compute_geometry returns for our geometry
"""
import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from src.utilities.config import Config
from TEMPEST_RAD.simulator import load_shape_model, compute_geometry

CONFIG_PATH = "private/data/config/moon/moon_config.yaml"

config = Config(CONFIG_PATH)
facets, _ = load_shape_model(config.path_to_shape_model_file)

# Zero phase geometry
sun_vec = np.array(config.sunlight_direction)
obs_vec = sun_vec
rot_axis = np.array([0, 0, 1])

lats, phases, emis, azis = compute_geometry(facets, sun_vec, obs_vec, rot_axis)

visible = emis < 90

print("="*70)
print("Latitudes from compute_geometry()")
print("="*70)
print(f"\nVisible facets: {np.sum(visible)}/{len(facets)}")
print(f"\nLatitudes (visible):")
print(f"  Min:  {lats[visible].min():.2f}°")
print(f"  Max:  {lats[visible].max():.2f}°")
print(f"  Mean: {lats[visible].mean():.2f}°")
print(f"  Std:  {lats[visible].std():.2f}°")
print(f"\nLUT latitude axis: [0.0, ..., 85.0] with 4 points")
print(f"\nLatitudes outside LUT range [0, 85]:")
outside = np.sum((lats[visible] < 0) | (lats[visible] > 85))
print(f"  {outside}/{np.sum(visible)} facets ({100*outside/np.sum(visible):.1f}%)")

# Check a few specific latitudes
print(f"\nSample latitudes:")
for i in np.where(visible)[0][:10]:
    print(f"  Facet {i}: lat={lats[i]:.2f}°, emi={emis[i]:.2f}°")

print("="*70)
