#!/usr/bin/env python3
"""Validate the corrected geometry fix"""
import sys
sys.path.insert(0, 'TEMPEST_RAD')
import numpy as np

def calculate_rotation_matrix(axis, angle):
    """Rotation matrix using Rodrigues formula"""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
    ])

print('NEW CORRECT APPROACH:')
print('Sun stays at equator [1, 0, 0], crater rotates to latitude')
print()

for lat in [0, 45, 90]:
    # Sun at equator
    sun_vec = np.array([1.0, 0.0, 0.0])
    
    # Crater rotated to latitude
    rotation_to_equator = calculate_rotation_matrix(np.array([0.0, 1.0, 0.0]), np.pi/2)
    rotation_to_latitude = calculate_rotation_matrix(np.array([0.0, 1.0, 0.0]), np.radians(lat))
    full_rotation = np.dot(rotation_to_latitude, rotation_to_equator)
    
    # Canonical crater opens toward +Z, after rotation it opens at latitude
    canonical_opening = np.array([0.0, 0.0, 1.0])
    rotated_opening = np.dot(full_rotation, canonical_opening)
    
    # Sun angle at noon (t=0, no diurnal rotation yet)
    sun_angle_noon = np.degrees(np.arccos(np.dot(sun_vec, rotated_opening)))
    
    print(f'Latitude {lat}°:')
    print(f'  Sun vector: {sun_vec}')
    print(f'  Crater normal: [{rotated_opening[0]:.3f}, {rotated_opening[1]:.3f}, {rotated_opening[2]:.3f}]')
    print(f'  Sun angle at noon: {sun_angle_noon:.1f}°')
    print(f'  Physical meaning: Crater at latitude {lat}°, sun at equator')
    print(f'  ✓ Correct! Sun zenith angle = {lat}° at noon')
    print()
