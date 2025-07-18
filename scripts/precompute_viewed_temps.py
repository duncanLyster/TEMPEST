#!/usr/bin/env python3
"""
Precompute viewed (apparent) temperatures for each parent facet using subfacet geometry and temperatures.
"""
import argparse
import numpy as np
import h5py
from stl import mesh
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute apparent facet temperatures from subfacet HDF5 and camera direction'
    )
    parser.add_argument(
        '--h5', default='outputs/subfacet_data.h5',
        help='Path to subfacet HDF5 file (points, faces, temps)'
    )
    parser.add_argument(
        '--stl', required=True,
        help='Path to parent shape model STL file'
    )
    parser.add_argument(
        '--theta', type=float, required=True,
        help='Polar angle (radians) of camera direction (0=+z)'
    )
    parser.add_argument(
        '--phi', type=float, required=True,
        help='Azimuth angle (radians) of camera direction'
    )
    parser.add_argument(
        '--out', default='outputs/viewed_temperatures.npz',
        help='Output NPZ file for apparent temperatures'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Build camera view direction unit vector
    view_dir = np.array([
        np.sin(args.theta) * np.cos(args.phi),
        np.sin(args.theta) * np.sin(args.phi),
        np.cos(args.theta)
    ], dtype=np.float64)

    # Load subfacet HDF5 data
    with h5py.File(args.h5, 'r') as hf:
        pts = hf['points'][:]                # (N_pts, 3)
        faces = hf['faces'][:].astype(int)   # (N_sub, 4)
        temps = hf['temps'][:]               # (N_sub, T)

    # Compute subfacet triangle normals and areas
    idx = faces[:, 1:4]
    p0 = pts[idx[:, 0]]
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    cross = np.cross(e1, e2)
    tri_areas = np.linalg.norm(cross, axis=1) * 0.5
    tri_normals = cross / np.linalg.norm(cross, axis=1)[:, None]

    # Load parent mesh to get facet count
    parent_mesh = mesh.Mesh.from_file(args.stl)
    n_parent = parent_mesh.vectors.shape[0]
    n_sub = temps.shape[0]
    if n_sub % n_parent != 0:
        raise ValueError(f"Cannot group {n_sub} subfacets into {n_parent} parent facets: non-integer M")
    M = n_sub // n_parent

    # Reshape into (n_parent, M, ...)
    T = temps.shape[1]
    temps_grouped = temps.reshape(n_parent, M, T)
    normals_grouped = tri_normals.reshape(n_parent, M, 3)
    areas_grouped = tri_areas.reshape(n_parent, M)

    # Compute weights = max(0, n·view_dir) * area
    cosines = np.einsum('pmk,k->pm', normals_grouped, view_dir)
    weights = np.maximum(0.0, cosines) * areas_grouped
    total_w = weights.sum(axis=1)[:, None] + 1e-12

    # Weighted apparent temperatures
    weighted = weights[:, :, None] * temps_grouped  # shape (n_parent, M, T)
    t_app = weighted.sum(axis=1) / total_w  # shape (n_parent, T)

    # Save results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, viewed_temperatures=t_app, view_dir=view_dir)
    print(f"Saved apparent temperatures ({n_parent} facets x {T} timesteps) to {args.out}")


if __name__ == '__main__':
    main() 