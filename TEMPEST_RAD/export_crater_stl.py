"""Export spherical crater meshes as STL for use with TEMPEST.

The canonical mesh normals point outward (away from sphere center) — correct
for the thermal sim of a concave bowl illuminated from above. For STL viewing
in Preview/MeshLab, normals need to point toward the viewer (inward toward
the sphere center), so we flip them and reverse vertex winding.
"""
import sys, struct, argparse
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model.spherical_cap_mesh import generate_canonical_spherical_cap

PROFILE_ANGLE = 90.0  # hemisphere

parser = argparse.ArgumentParser(description="Export crater STL")
parser.add_argument("-n", type=int, default=10000, help="Target subfacet count")
args = parser.parse_args()
TARGET_SUBFACETS = args.n

print(f"Generating canonical spherical cap: {TARGET_SUBFACETS} target subfacets, "
      f"opening angle {PROFILE_ANGLE}°...")
mesh = generate_canonical_spherical_cap(TARGET_SUBFACETS, PROFILE_ANGLE)
n = len(mesh)
print(f"  Actual subfacet count: {n}")

# Write binary STL with normals flipped for correct external viewing
out_path = Path(__file__).resolve().parent / f"crater_{TARGET_SUBFACETS}.stl"
with open(out_path, 'wb') as f:
    # 80-byte header
    header = f"TEMPEST crater mesh: {n} facets, {PROFILE_ANGLE}deg".encode('ascii')
    f.write(header.ljust(80, b'\0'))
    # Number of triangles
    f.write(struct.pack('<I', n))
    for entry in mesh:
        # Flip normal (outward→inward) for correct STL display
        normal = (-entry['normal']).astype(np.float32)
        # Reverse vertex winding to match flipped normal
        verts = entry['vertices'][::-1].astype(np.float32)
        f.write(struct.pack('<3f', *normal))
        for v in verts:
            f.write(struct.pack('<3f', *v))
        f.write(struct.pack('<H', 0))  # attribute byte count

print(f"  Saved: {out_path}")
print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")
