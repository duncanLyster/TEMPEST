#!/usr/bin/env python3
"""
Generate a cratered icosphere STL for TEMPEST validation.

For each facet of the 80-facet icosphere, this script:
  1. Places a 122-subfacet hemispherical crater at the inscribed circle centre
  2. Tessellates the remaining flat "corner" regions around the crater rim
  3. Outputs a combined STL suitable for direct TEMPEST simulation

The crater geometry is identical to what the LUT generator produces
(CRATER_SUBFACETS=100 → 122 actual, opening_angle=90°, power=0.5 ring spacing).

Usage:
    python TEMPEST_RAD/scripts/generate_cratered_stl.py
"""

import numpy as np
import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from stl import mesh as stl_mesh_module
from src.model.spherical_cap_mesh import generate_canonical_spherical_cap


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotation_matrix_z_to(target):
    """Rotation matrix that maps +Z to *target* (unit vector).  Rodrigues."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, target)
    s = np.linalg.norm(v)
    c = np.dot(z, target)

    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        else:
            # 180° flip – rotate around whichever axis is most orthogonal
            return np.diag([1.0, -1.0, -1.0])

    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)


def compute_incircle(v0, v1, v2):
    """Return (incenter, inradius) for triangle (v0, v1, v2)."""
    a = np.linalg.norm(v2 - v1)   # edge opposite v0
    b = np.linalg.norm(v2 - v0)   # edge opposite v1
    c = np.linalg.norm(v1 - v0)   # edge opposite v2
    perimeter = a + b + c
    incenter = (a * v0 + b * v1 + c * v2) / perimeter
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    inradius = area / (perimeter / 2)
    return incenter, inradius


def extract_ordered_rim_points(crater_facets):
    """
    Return the outermost ring of crater vertices (at the opening),
    sorted counter-clockwise when viewed from +Z.
    """
    # Collect unique vertices
    verts_set = set()
    for f in crater_facets:
        for v in f['vertices']:
            verts_set.add(tuple(np.round(v, 10)))
    unique = np.array(sorted(verts_set))

    # Rim vertices sit at the maximum z (≈ 0 for 90° cap)
    z_max = unique[:, 2].max()
    z_range = z_max - unique[:, 2].min()
    tol = 0.02 * z_range
    rim = unique[np.abs(unique[:, 2] - z_max) < tol]

    # Sort by polar angle around the z-axis
    angles = np.arctan2(rim[:, 1], rim[:, 0])
    return rim[np.argsort(angles)]


def _add_oriented_tri(tri_list, va, vb, vc, ref_normal):
    """Append a triangle with winding consistent with *ref_normal*."""
    cross = np.cross(vb - va, vc - va)
    area2 = np.linalg.norm(cross)
    if area2 < 1e-12:
        return  # degenerate
    if np.dot(cross, ref_normal) < 0:
        tri_list.append((va.copy(), vc.copy(), vb.copy(), ref_normal.copy()))
    else:
        tri_list.append((va.copy(), vb.copy(), vc.copy(), ref_normal.copy()))


def tessellate_corners(host_verts, rim_pts, host_normal):
    """
    Fill the flat region between the host triangle edges and the crater rim.

    Strategy: assign each rim point to the nearest host vertex, then for
    each consecutive rim pair create a fan triangle.  At sector boundaries
    (where the assignment changes) an extra triangle fills the gap.
    """
    N = len(rim_pts)
    tris = []

    # Assign each rim point to the closest host vertex
    dists = np.linalg.norm(
        rim_pts[:, np.newaxis, :] - host_verts[np.newaxis, :, :], axis=2
    )
    assign = np.argmin(dists, axis=1)

    for j in range(N):
        jn = (j + 1) % N
        p1 = rim_pts[j]
        p2 = rim_pts[jn]
        a1 = assign[j]
        a2 = assign[jn]

        if a1 == a2:
            # Both rim points belong to the same vertex → single fan tri
            _add_oriented_tri(tris, host_verts[a1], p1, p2, host_normal)
        else:
            # Sector boundary → split the quad into two triangles
            _add_oriented_tri(tris, host_verts[a1], p1, p2, host_normal)
            _add_oriented_tri(tris, host_verts[a1], p2, host_verts[a2], host_normal)

    return tris


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ico_path = ROOT_DIR / 'data' / 'shape_models' / '500m_ico_sphere_80_facets.stl'
    output_path = ROOT_DIR / 'data' / 'shape_models' / '500m_cratered_sphere_80x122.stl'

    # ------------------------------------------------------------------
    # 1. Load icosphere
    # ------------------------------------------------------------------
    ico = stl_mesh_module.Mesh.from_file(str(ico_path))
    n_host = len(ico.normals)
    print(f"Loaded icosphere: {n_host} facets  ({ico_path.name})")

    # ------------------------------------------------------------------
    # 2. Generate canonical hemispherical crater (opening toward +Z)
    # ------------------------------------------------------------------
    print("\nGenerating canonical crater mesh (SUBFACETS=100, angle=90°, power=0.5)…")
    crater_facets = generate_canonical_spherical_cap(100, 90.0)
    n_crater = len(crater_facets)
    print(f"Crater subfacets: {n_crater}")

    # Extract ordered rim points
    rim_pts_canonical = extract_ordered_rim_points(crater_facets)
    n_rim = len(rim_pts_canonical)
    canonical_r = np.sqrt(np.mean(rim_pts_canonical[:, 0]**2 + rim_pts_canonical[:, 1]**2))
    print(f"Rim points: {n_rim}  (canonical opening radius = {canonical_r:.6f})")

    # Verify crater interior area (hemisphere = 2 × opening area for 90° cap)
    crater_area = sum(f['area'] for f in crater_facets)
    print(f"Crater interior area: {crater_area:.6f}  (expected ~2.0 for unit-opening hemisphere)")

    # ------------------------------------------------------------------
    # 3. Build the combined mesh
    # ------------------------------------------------------------------
    all_tris = []   # list of (v0, v1, v2, normal)

    for i in range(n_host):
        v0, v1, v2 = ico.vectors[i]

        # Outward unit normal
        cross = np.cross(v1 - v0, v2 - v0)
        host_n = cross / np.linalg.norm(cross)
        center = (v0 + v1 + v2) / 3.0
        if np.dot(host_n, center) < 0:
            host_n = -host_n
            v0, v1, v2 = v0, v2, v1   # fix winding

        # Incircle → defines crater size & position
        inc_center, inc_radius = compute_incircle(v0, v1, v2)

        # Scale factor:  canonical_r  →  inc_radius  (×0.8 to avoid overlap)
        scale = 0.8 * inc_radius / canonical_r

        # Rotation:  +Z  →  host_n
        R = rotation_matrix_z_to(host_n)

        # --- crater subfacets ---
        for cf in crater_facets:
            verts = cf['vertices'] * scale          # scale
            verts = (R @ verts.T).T                  # rotate
            verts = verts + inc_center               # translate

            n = R @ cf['normal']
            all_tris.append((verts[0].copy(), verts[1].copy(), verts[2].copy(), n.copy()))

        # --- flat corner triangles ---
        rim_3d = rim_pts_canonical * scale
        rim_3d = (R @ rim_3d.T).T + inc_center

        corners = tessellate_corners(np.array([v0, v1, v2]), rim_3d, host_n)
        all_tris.extend(corners)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Host facet {i+1:3d}/{n_host}  "
                  f"inc_r={inc_radius:.2f} m  "
                  f"crater subfacets={n_crater}  corners={len(corners)}")

    # ------------------------------------------------------------------
    # 4. Write STL
    # ------------------------------------------------------------------
    total = len(all_tris)
    n_corners_avg = (total - n_host * n_crater) / n_host
    print(f"\nTotal facets: {total}  "
          f"({n_host} × {n_crater} crater + {n_corners_avg:.0f} corner = "
          f"{n_host * n_crater} + {total - n_host * n_crater})")

    combined = stl_mesh_module.Mesh(np.zeros(total, dtype=stl_mesh_module.Mesh.dtype))
    n_flipped = 0
    for j, (va, vb, vc, n) in enumerate(all_tris):
        # Ensure vertex winding matches the stored normal (STL viewers use winding)
        winding_normal = np.cross(vb - va, vc - va)
        if np.dot(winding_normal, n) < 0:
            # Swap two vertices to reverse winding
            vb, vc = vc, vb
            n_flipped += 1
        combined.vectors[j] = [va, vb, vc]
        combined.normals[j] = n
    print(f"Fixed winding on {n_flipped}/{total} facets")
    # NOTE: do NOT call combined.update_normals() — we set them explicitly

    combined.save(str(output_path))
    print(f"\nSaved → {output_path}")

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    areas = 0.5 * np.linalg.norm(
        np.cross(combined.vectors[:, 1] - combined.vectors[:, 0],
                 combined.vectors[:, 2] - combined.vectors[:, 0]), axis=1)

    ico_areas = 0.5 * np.linalg.norm(
        np.cross(ico.vectors[:, 1] - ico.vectors[:, 0],
                 ico.vectors[:, 2] - ico.vectors[:, 0]), axis=1)

    print(f"\n{'='*60}")
    print(f"Original smooth sphere area : {ico_areas.sum():.1f} m²")
    print(f"Cratered model total area   : {areas.sum():.1f} m²")
    print(f"  crater subfacets area     : {areas[:n_host*n_crater].sum():.1f} m²")
    print(f"  flat corner area          : {areas[n_host*n_crater:].sum():.1f} m²")
    print(f"Facet area range            : [{areas.min():.4f}, {areas.max():.2f}] m²")
    print(f"Total facets                : {total}")

    # Crater opening area fraction per host facet
    opening_areas = np.pi * np.array([
        compute_incircle(ico.vectors[i, 0], ico.vectors[i, 1], ico.vectors[i, 2])[1]**2
        for i in range(n_host)
    ])
    frac = opening_areas / ico_areas
    print(f"\nCrater opening / host facet area : "
          f"mean={frac.mean():.3f}  range=[{frac.min():.3f}, {frac.max():.3f}]")
    print(f"  → equivalent roughness fraction f ≈ {frac.mean():.3f}")

    # Expected area increase: hemisphere interior = 2× opening area
    # Extra area = (2-1) × opening area = opening area  (the curved interior
    # replaces the flat disc, adding πr² extra)
    expected_area = ico_areas.sum() + opening_areas.sum()
    print(f"\nExpected area (smooth + Σ opening) : {expected_area:.1f} m²")

    print(f"\n{'='*60}")
    print("Done.  Use this STL as the TEMPEST shape model for validation.")
    print(f"It has {n_host} craters × {n_crater} subfacets each + {total - n_host*n_crater} flat corner facets.")


if __name__ == '__main__':
    main()
