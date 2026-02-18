#!/usr/bin/env python3
"""
Check what viewing geometries we're querying for polar facets
"""
import os
import sys
import numpy as np
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.append(str(root_dir))
os.chdir(root_dir)

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
print("Viewing Geometry for Polar vs Equatorial Facets (Zero Phase)")
print("="*70)

# Categorize by latitude
equatorial = (np.abs(lats) < 30) & visible
midlat = (np.abs(lats) >= 30) & (np.abs(lats) < 60) & visible
polar = (np.abs(lats) >= 60) & visible

print(f"\nEquatorial facets (lat < 30°): {np.sum(equatorial)}")
if np.sum(equatorial) > 0:
    print(f"  Latitudes:  min={lats[equatorial].min():.1f}°, max={lats[equatorial].max():.1f}°, mean={lats[equatorial].mean():.1f}°")
    print(f"  Emissions:  min={emis[equatorial].min():.1f}°, max={emis[equatorial].max():.1f}°, mean={emis[equatorial].mean():.1f}°")
    print(f"  Azimuths:   min={azis[equatorial].min():.1f}°, max={azis[equatorial].max():.1f}°, mean={azis[equatorial].mean():.1f}°")

print(f"\nMid-latitude facets (30° ≤ lat < 60°): {np.sum(midlat)}")
if np.sum(midlat) > 0:
    print(f"  Latitudes:  min={lats[midlat].min():.1f}°, max={lats[midlat].max():.1f}°, mean={lats[midlat].mean():.1f}°")
    print(f"  Emissions:  min={emis[midlat].min():.1f}°, max={emis[midlat].max():.1f}°, mean={emis[midlat].mean():.1f}°")
    print(f"  Azimuths:   min={azis[midlat].min():.1f}°, max={azis[midlat].max():.1f}°, mean={azis[midlat].mean():.1f}°")

print(f"\nPolar facets (lat ≥ 60°): {np.sum(polar)}")
if np.sum(polar) > 0:
    print(f"  Latitudes:  min={lats[polar].min():.1f}°, max={lats[polar].max():.1f}°, mean={lats[polar].mean():.1f}°")
    print(f"  Emissions:  min={emis[polar].min():.1f}°, max={emis[polar].max():.1f}°, mean={emis[polar].mean():.1f}°")
    print(f"  Azimuths:   min={azis[polar].min():.1f}°, max={azis[polar].max():.1f}°, mean={azis[polar].mean():.1f}°")

print(f"\n{'='*70}")
print("ISSUE: Are polar facets being viewed at high emission angles?")
print("If so, we're querying LUT where factors are low, making poles cold.")
print("="*70)
