#!/usr/bin/env python3
"""
Inspect the STL mesh to understand its 3D structure and alignment.
"""

import numpy as np
import trimesh
from pathlib import Path

stl_path = '/Users/duncan/Desktop/DPhil/TEMPEST/data/shape_models/Moon/lola_dem_latm4p0_lon336p5_15km_128ppd.stl'

print(f"Loading mesh from {stl_path}")
mesh = trimesh.load(stl_path)

vertices = mesh.vertices
faces = mesh.faces
facet_centers = mesh.triangles_center

print(f"\n=== MESH STRUCTURE ===")
print(f"Vertices: {len(vertices)}")
print(f"Facets: {len(faces)}")
print(f"\n=== VERTEX COORDINATE RANGES ===")
print(f"X: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}] (range: {vertices[:, 0].max() - vertices[:, 0].min():.2f})")
print(f"Y: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}] (range: {vertices[:, 1].max() - vertices[:, 1].min():.2f})")
print(f"Z: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}] (range: {vertices[:, 2].max() - vertices[:, 2].min():.2f})")

print(f"\n=== FACET CENTER COORDINATE RANGES ===")
print(f"X: [{facet_centers[:, 0].min():.2f}, {facet_centers[:, 0].max():.2f}] (range: {facet_centers[:, 0].max() - facet_centers[:, 0].min():.2f})")
print(f"Y: [{facet_centers[:, 1].min():.2f}, {facet_centers[:, 1].max():.2f}] (range: {facet_centers[:, 1].max() - facet_centers[:, 1].min():.2f})")
print(f"Z: [{facet_centers[:, 2].min():.2f}, {facet_centers[:, 2].max():.2f}] (range: {facet_centers[:, 2].max() - facet_centers[:, 2].min():.2f})")

# Check mesh bounds
bounds = mesh.bounds  # [min, max] for each axis
print(f"\n=== MESH BOUNDS ===")
print(f"Bounds (min): {bounds[0]}")
print(f"Bounds (max): {bounds[1]}")
print(f"Extents: {bounds[1] - bounds[0]}")

# Calculate mesh center
center = mesh.centroid
print(f"\n=== MESH CENTER (CENTROID) ===")
print(f"Centroid: {center}")

# Analyze vertex distribution
print(f"\n=== VERTEX STATISTICS ===")
vert_mean = vertices.mean(axis=0)
vert_std = vertices.std(axis=0)
print(f"Mean: {vert_mean}")
print(f"Std Dev: {vert_std}")

# Check if mesh is roughly planar (check Z variation relative to X,Y)
z_range = vertices[:, 2].max() - vertices[:, 2].min()
xy_range = np.sqrt((vertices[:, 0].max() - vertices[:, 0].min())**2 + 
                   (vertices[:, 1].max() - vertices[:, 1].min())**2)
print(f"\n=== PLANARITY CHECK ===")
print(f"XY extent: {xy_range:.2f}")
print(f"Z extent: {z_range:.2f}")
print(f"Z/XY ratio: {z_range/xy_range:.6f} (< 0.01 suggests mostly flat)")

# Check face normal directions
print(f"\n=== FACE NORMAL ANALYSIS ===")
face_normals = mesh.face_normals
print(f"Face normals - X: [{face_normals[:, 0].min():.4f}, {face_normals[:, 0].max():.4f}]")
print(f"Face normals - Y: [{face_normals[:, 1].min():.4f}, {face_normals[:, 1].max():.4f}]")
print(f"Face normals - Z: [{face_normals[:, 2].min():.4f}, {face_normals[:, 2].max():.4f}]")

# Find dominant normal direction
mean_normal = face_normals.mean(axis=0)
print(f"Mean face normal: {mean_normal}")
print(f"Mean normal magnitude: {np.linalg.norm(mean_normal):.4f}")

# Check for patches of similar-oriented faces
if np.abs(mean_normal[2]) > 0.8:
    print("-> Mesh faces point mostly in Z direction (likely flat in XY plane)")
elif np.abs(mean_normal[0]) > 0.8 or np.abs(mean_normal[1]) > 0.8:
    print(f"-> Mesh faces point mostly in {'X' if np.abs(mean_normal[0]) > np.abs(mean_normal[1]) else 'Y'} direction")

# Sample some faces to see their structure
print(f"\n=== SAMPLE FACES ===")
for i in [0, len(faces)//4, len(faces)//2, 3*len(faces)//4, len(faces)-1]:
    face = faces[i]
    v0, v1, v2 = vertices[face]
    print(f"Face {i}: vertices at {v0}, {v1}, {v2}")
