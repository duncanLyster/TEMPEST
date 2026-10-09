#!/usr/bin/env python3
"""
Convert a region of a projected GeoTIFF DEM into a TEMPEST-ready STL mesh.

TEMPEST rotates the shape model about an axis through the origin, so a terrain
patch must sit at its true position on the body (radius R + h in the direction of
its latitude and longitude) to go through a diurnal cycle. A flat patch at z = 0
would just spin in place.

Frames
------
Output vertices are in the TEMPEST body frame, metres:
    +Z  = rotation axis (north)
    +X  = sub-solar direction at t=0
    +Y  = completes the right-handed set
The ROI centre longitude is rotated onto +X, so the patch centre starts at
local noon. Latitude is preserved, which is what sets the solar elevation.

Placement modes
---------------
sphere   (default) Each post is unprojected to (lat, lon) and placed at
                   (R + h) * u(lat, lon). Exact; carries body curvature.
tangent            The patch is built on a local East/North/Up plane and then
                   rigidly moved to the same place on the sphere. Identical to
                   `sphere` except that curvature is removed, so differencing
                   the two isolates the curvature contribution.

Elevation convention
--------------------
For a shape-from-shading DEM only short-wavelength topography is constrained, so
h is placed on a sphere: the sphere gives the long wavelengths and the DEM the
short ones. `--detrend-km` high-pass filters h to remove any remaining regional trend.

Sidecar
-------
Alongside the STL a `<name>.facets.npz` is written whose row i describes STL
facet i, in the same order TEMPEST reads them. That is what lets per-facet
property maps (albedo, thermal inertia, ...) be sampled from any co-registered
raster and stay aligned with the mesh.

Examples
--------
  # 50x50 posts starting at row 1700, col 900, native resolution
  python scripts/dem_to_stl.py DEM.tif out.stl --window 1700 900 50 50

  # a lon/lat box, 2x2 binned, mildly smoothed
  python scripts/dem_to_stl.py DEM.tif out.stl \
      --bbox-lonlat 172 176 -61 -59 --bin 2 --smooth-px 1.0
"""

import argparse
import json
import os
import sys

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import xy as raster_xy


# --------------------------------------------------------------------------- #
# DEM reading and conditioning
# --------------------------------------------------------------------------- #

def read_dem_window(path, row0, col0, nrows, ncols):
    """Read a pixel window, returning elevation with nodata as NaN plus context."""
    with rasterio.open(path) as src:
        full_h, full_w = src.height, src.width
        row0 = max(0, min(int(row0), full_h - 1))
        col0 = max(0, min(int(col0), full_w - 1))
        nrows = int(min(nrows, full_h - row0))
        ncols = int(min(ncols, full_w - col0))
        if nrows < 2 or ncols < 2:
            raise ValueError(f"Window {nrows}x{ncols} is too small; need at least 2x2 posts.")

        z = src.read(1, window=((row0, row0 + nrows), (col0, col0 + ncols))).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            z[z == nodata] = np.nan
        # Guard against float32 sentinels that survive an exact-equality test.
        z[~np.isfinite(z)] = np.nan
        z[np.abs(z) > 1e30] = np.nan

        return {
            "z": z,
            "row0": row0,
            "col0": col0,
            "transform": src.transform,
            "crs": src.crs,
            "pixel_size": abs(src.transform.a),
            "source": os.path.abspath(path),
            "full_shape": (full_h, full_w),
        }


def bbox_lonlat_to_window(path, lon_min, lon_max, lat_min, lat_max):
    """Translate a lon/lat box into a pixel window on the DEM grid."""
    with rasterio.open(path) as src:
        body_r = _body_radius(src.crs)
        geog = CRS.from_user_input(f"+proj=longlat +R={body_r} +no_defs")
        fwd = Transformer.from_crs(geog, src.crs, always_xy=True)
        lons = np.array([lon_min, lon_max, lon_min, lon_max], dtype=float)
        lats = np.array([lat_min, lat_min, lat_max, lat_max], dtype=float)
        xs, ys = fwd.transform(lons, lats)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        row0, row1 = int(np.min(rows)), int(np.max(rows))
        col0, col1 = int(np.min(cols)), int(np.max(cols))
    return row0, col0, row1 - row0 + 1, col1 - col0 + 1


def bin_block(z, factor):
    """Average z in factor x factor blocks, ignoring NaN, trimming the remainder."""
    if factor <= 1:
        return z
    h = (z.shape[0] // factor) * factor
    w = (z.shape[1] // factor) * factor
    if h < factor or w < factor:
        raise ValueError(f"Window too small to bin by {factor}.")
    blocks = z[:h, :w].reshape(h // factor, factor, w // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3))


def smooth_nan(z, sigma_px):
    """Gaussian smooth while respecting NaN (normalised convolution)."""
    if sigma_px <= 0:
        return z
    from scipy.ndimage import gaussian_filter

    valid = np.isfinite(z)
    filled = np.where(valid, z, 0.0)
    num = gaussian_filter(filled, sigma_px, mode="nearest")
    den = gaussian_filter(valid.astype(float), sigma_px, mode="nearest")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-12), np.nan)
    out[~valid] = np.nan
    return out


def detrend(z, scale_px):
    """High-pass: subtract a heavily smoothed copy, keeping short wavelengths."""
    if scale_px <= 0:
        return z
    return z - smooth_nan(z, scale_px)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _body_radius(crs):
    """Body radius in metres from a projected CRS, defaulting to Europa."""
    try:
        ell = CRS.from_user_input(crs).ellipsoid
        if ell is not None and ell.semi_major_metre:
            return float(ell.semi_major_metre)
    except Exception:
        pass
    return 1560800.0


def posts_to_lonlat(meta, z_shape, bin_factor):
    """Lon/lat of every post centre in the (possibly binned) window."""
    nrows, ncols = z_shape
    step = max(1, bin_factor)
    # Centre of each (binned) cell, in original pixel coordinates.
    rr = meta["row0"] + (np.arange(nrows) * step + (step - 1) / 2.0)
    cc = meta["col0"] + (np.arange(ncols) * step + (step - 1) / 2.0)
    CC, RR = np.meshgrid(cc, rr)
    xs, ys = raster_xy(meta["transform"], RR, CC, offset="center")
    xs = np.asarray(xs, dtype=float).reshape(nrows, ncols)
    ys = np.asarray(ys, dtype=float).reshape(nrows, ncols)

    body_r = _body_radius(meta["crs"])
    geog = CRS.from_user_input(f"+proj=longlat +R={body_r} +no_defs")
    inv = Transformer.from_crs(meta["crs"], geog, always_xy=True)
    lon, lat = inv.transform(xs, ys)
    return np.asarray(lon), np.asarray(lat), xs, ys, body_r


def place_on_sphere(lon, lat, z, body_r, lon0, mode):
    """Map posts to TEMPEST body-frame XYZ (metres)."""
    # Wrap into (-180, 180] so a ROI straddling the antimeridian has no 360 deg jump
    # (tangent mode uses lam linearly).
    lam = np.radians((lon - lon0 + 180.0) % 360.0 - 180.0)
    phi = np.radians(lat)
    h = np.where(np.isfinite(z), z, 0.0)
    r = body_r + h

    if mode == "sphere":
        X = r * np.cos(phi) * np.cos(lam)
        Y = r * np.cos(phi) * np.sin(lam)
        Z = r * np.sin(phi)
        return np.stack([X, Y, Z], axis=-1)

    if mode == "tangent":
        # Local ENU offsets about the ROI centre, then lifted onto the sphere.
        phi0 = np.radians(np.nanmean(lat))
        east = body_r * np.cos(phi0) * lam
        north = body_r * (phi - phi0)
        up = h
        # ENU basis at (phi0, lon0) with lon0 already rotated onto +X.
        e_hat = np.array([0.0, 1.0, 0.0])
        n_hat = np.array([-np.sin(phi0), 0.0, np.cos(phi0)])
        u_hat = np.array([np.cos(phi0), 0.0, np.sin(phi0)])
        origin = body_r * u_hat
        return (origin
                + east[..., None] * e_hat
                + north[..., None] * n_hat
                + up[..., None] * u_hat)

    raise ValueError(f"Unknown placement mode: {mode}")


def build_faces(valid):
    """Two triangles per cell whose four corners are all valid.

    Returns face vertex indices into the flattened post grid, and the (row, col)
    of the cell each triangle came from.
    """
    nrows, ncols = valid.shape
    j, i = np.mgrid[0 : nrows - 1, 0 : ncols - 1]
    j = j.ravel()
    i = i.ravel()

    v00 = j * ncols + i
    v01 = j * ncols + (i + 1)
    v10 = (j + 1) * ncols + i
    v11 = (j + 1) * ncols + (i + 1)

    flat_valid = valid.ravel()
    cell_ok = (flat_valid[v00] & flat_valid[v01] & flat_valid[v10] & flat_valid[v11])

    v00, v01, v10, v11 = v00[cell_ok], v01[cell_ok], v10[cell_ok], v11[cell_ok]
    jj, ii = j[cell_ok], i[cell_ok]

    # Interleave so the two triangles of a cell stay adjacent in the file.
    tri_a = np.stack([v00, v01, v10], axis=1)
    tri_b = np.stack([v01, v11, v10], axis=1)
    faces = np.empty((2 * len(tri_a), 3), dtype=np.int64)
    faces[0::2] = tri_a
    faces[1::2] = tri_b

    cell_rc = np.empty((2 * len(jj), 2), dtype=np.int64)
    cell_rc[0::2, 0] = jj
    cell_rc[0::2, 1] = ii
    cell_rc[1::2, 0] = jj
    cell_rc[1::2, 1] = ii
    return faces, cell_rc


def face_geometry(verts, faces):
    """Outward normals (away from body centre), centroids and areas."""
    tri = verts[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area

    normals = np.zeros_like(cross)
    good = twice_area > 0
    normals[good] = cross[good] / twice_area[good, None]

    centroids = tri.mean(axis=1)
    # Flip any normal pointing back into the body.
    radial = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    flip = np.einsum("ij,ij->i", normals, radial) < 0
    normals[flip] *= -1.0
    faces = faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return normals, centroids, areas, faces


# --------------------------------------------------------------------------- #
# STL output
# --------------------------------------------------------------------------- #

def write_stl_binary(path, verts, faces, normals):
    tri = verts[faces].astype(np.float32)
    n = normals.astype(np.float32)
    rec = np.zeros(len(faces), dtype=np.dtype([
        ("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")
    ]))
    rec["normal"] = n
    rec["v"] = tri
    with open(path, "wb") as f:
        header = b"TEMPEST terrain mesh (metres, body frame)"
        f.write(header.ljust(80, b"\0"))
        f.write(np.uint32(len(faces)).tobytes())
        f.write(rec.tobytes())


def write_stl_ascii(path, verts, faces, normals):
    tri = verts[faces]
    with open(path, "w") as f:
        f.write("solid terrain\n")
        for k in range(len(faces)):
            nx, ny, nz = normals[k]
            f.write(f"facet normal {nx:.9e} {ny:.9e} {nz:.9e}\n")
            f.write("  outer loop\n")
            for v in tri[k]:
                f.write(f"    vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
            f.write("  endloop\n")
            f.write("endfacet\n")
        f.write("endsolid terrain\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def convert(args):
    if args.bbox_lonlat:
        row0, col0, nrows, ncols = bbox_lonlat_to_window(args.dem, *args.bbox_lonlat)
        print(f"lon/lat box -> window row0={row0} col0={col0} {nrows}x{ncols}")
    else:
        row0, col0, nrows, ncols = args.window

    meta = read_dem_window(args.dem, row0, col0, nrows, ncols)
    z_raw = meta["z"]
    n_valid_raw = int(np.isfinite(z_raw).sum())
    print(f"read {z_raw.shape[0]}x{z_raw.shape[1]} posts @ {meta['pixel_size']:.2f} m/px "
          f"({100 * n_valid_raw / z_raw.size:.1f}% valid)")
    if n_valid_raw == 0:
        raise SystemExit("Window contains no valid DEM data.")

    z = bin_block(z_raw, args.bin)
    if args.bin > 1:
        print(f"binned {args.bin}x -> {z.shape[0]}x{z.shape[1]} posts "
              f"@ {meta['pixel_size'] * args.bin:.2f} m/px")
    z = smooth_nan(z, args.smooth_px)
    if args.detrend_km > 0:
        eff_px = meta["pixel_size"] * max(1, args.bin)
        scale_px = args.detrend_km * 1000.0 / eff_px
        z = detrend(z, scale_px)
        print(f"detrended: removed wavelengths > ~{args.detrend_km} km ({scale_px:.1f} px)")

    lon, lat, proj_x, proj_y, body_r = posts_to_lonlat(meta, z.shape, args.bin)
    # Circular mean, in case the ROI straddles the antimeridian
    if args.lon0 is None:
        _lr = np.radians(lon[np.isfinite(lon)])
        lon0 = float(np.degrees(np.arctan2(np.nanmean(np.sin(_lr)),
                                           np.nanmean(np.cos(_lr))))) % 360.0
    else:
        lon0 = args.lon0
    verts = place_on_sphere(lon, lat, z, body_r, lon0, args.mode)
    verts = verts.reshape(-1, 3)

    faces, cell_rc = build_faces(np.isfinite(z))
    if len(faces) == 0:
        raise SystemExit("No cell had four valid corners; nothing to mesh.")
    normals, centroids, areas, faces = face_geometry(verts, faces)

    # Drop degenerate slivers.
    keep = areas > 0
    if not keep.all():
        print(f"dropped {int((~keep).sum())} degenerate facets")
        faces, normals, centroids, areas = faces[keep], normals[keep], centroids[keep], areas[keep]
        cell_rc = cell_rc[keep]

    # Slope relative to the local radial direction.
    radial = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    cos_slope = np.clip(np.einsum("ij,ij->i", normals, radial), -1.0, 1.0)
    slope_deg = np.degrees(np.arccos(cos_slope))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if args.ascii:
        write_stl_ascii(args.out, verts, faces, normals)
    else:
        write_stl_binary(args.out, verts, faces, normals)

    # Per-facet source pixel in ORIGINAL DEM coordinates, for property sampling.
    step = max(1, args.bin)
    src_row = meta["row0"] + (cell_rc[:, 0] * step + (step - 1) / 2.0) + step / 2.0
    src_col = meta["col0"] + (cell_rc[:, 1] * step + (step - 1) / 2.0) + step / 2.0

    lon_f = lon.ravel()[faces].mean(axis=1)
    lat_f = lat.ravel()[faces].mean(axis=1)

    sidecar = os.path.splitext(args.out)[0] + ".facets.npz"
    provenance = {
        "dem": meta["source"],
        "window": [int(meta["row0"]), int(meta["col0"]), int(z_raw.shape[0]), int(z_raw.shape[1])],
        "bin": int(args.bin),
        "smooth_px": float(args.smooth_px),
        "detrend_km": float(args.detrend_km),
        "mode": args.mode,
        "lon0_deg": float(lon0),
        "body_radius_m": float(body_r),
        "pixel_size_m": float(meta["pixel_size"]),
        "effective_post_spacing_m": float(meta["pixel_size"] * step),
        "crs": str(meta["crs"]),
        "n_facets": int(len(faces)),
        "ascii": bool(args.ascii),
    }
    np.savez_compressed(
        sidecar,
        src_row=src_row, src_col=src_col,
        lon=lon_f, lat=lat_f,
        centroid=centroids, normal=normals, area=areas, slope_deg=slope_deg,
        provenance=json.dumps(provenance),
    )

    span_km = (np.nanmax(proj_x) - np.nanmin(proj_x)) / 1000.0, \
              (np.nanmax(proj_y) - np.nanmin(proj_y)) / 1000.0
    print()
    print(f"  facets              {len(faces):,}")
    print(f"  patch span          {span_km[0]:.1f} x {span_km[1]:.1f} km")
    print(f"  centre lon/lat      {lon0:.3f} E, {np.nanmean(lat):.3f}  (lon is a circular mean)")
    print(f"  elevation range     {np.nanmin(z):.1f} to {np.nanmax(z):.1f} m")
    print(f"  facet area          mean {areas.mean() / 1e6:.4f} km^2")
    print(f"  slope from radial   mean {slope_deg.mean():.2f} deg, "
          f"p95 {np.percentile(slope_deg, 95):.2f}, max {slope_deg.max():.2f}")
    if not args.ascii:
        ulp = np.spacing(np.float32(body_r))
        print(f"  float32 STL precision at r={body_r/1e3:.0f} km: {ulp:.3f} m "
              f"(~{np.degrees(np.arctan(ulp * 1.41 / (meta['pixel_size'] * step))):.3f} deg slope noise)")
    print()
    print(f"  wrote {args.out}")
    print(f"  wrote {sidecar}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a projected GeoTIFF DEM region into a TEMPEST-ready STL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[-1],
    )
    p.add_argument("dem", help="input DEM GeoTIFF (elevation in metres)")
    p.add_argument("out", help="output .stl path")

    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--window", nargs=4, type=int, metavar=("ROW0", "COL0", "NROWS", "NCOLS"),
                     help="pixel window on the DEM grid")
    sel.add_argument("--bbox-lonlat", nargs=4, type=float,
                     metavar=("LONMIN", "LONMAX", "LATMIN", "LATMAX"),
                     help="geographic box (degrees east, degrees north)")

    p.add_argument("--bin", type=int, default=1, help="average NxN pixel blocks (default 1)")
    p.add_argument("--smooth-px", type=float, default=0.0,
                   help="Gaussian sigma in (binned) pixels, NaN-aware (default 0)")
    p.add_argument("--detrend-km", type=float, default=0.0,
                   help="high-pass: remove wavelengths longer than this (default 0 = off)")
    p.add_argument("--mode", choices=["sphere", "tangent"], default="sphere",
                   help="placement on the body (default sphere)")
    p.add_argument("--lon0", type=float, default=None,
                   help="longitude rotated onto +X, i.e. local noon at t=0 (default: ROI centre)")
    p.add_argument("--ascii", action="store_true",
                   help="write ASCII STL (float64 precision, ~6x larger, slower to load)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(convert(parse_args()))
