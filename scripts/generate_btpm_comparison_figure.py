#!/usr/bin/env python3
"""
Generate publication-quality comparison figure: TEMPEST (explicit & implicit) vs Sorli BTPM.

Layout (combined figure):
  Row 0:  [Explicit temp map]  [Implicit temp map]  [BTPM temp map]
  Row 1:  [Explicit−BTPM diff] [Implicit−BTPM diff] [Explicit−Implicit diff]
  Row 2:  Diurnal curves for N selected facets (all three models overlaid)
  Row 3:  Residuals vs BTPM

Selected facets are highlighted on the 3D maps by drawing their triangle edges
in the colour used for that facet's diurnal curve below.

Usage examples:
    # Explicit paths
    python scripts/generate_btpm_comparison_figure.py \
        --tempest-explicit data/output/remote_outputs/tempest_explicit_selfheating/animation_params.npz \
        --tempest-implicit data/output/remote_outputs/tempest_implicit_selfheating/animation_params.npz \
        --facets 10 45 82 --combined --timestep -1

    # Auto-detect (uses well-known folder names)
    python scripts/generate_btpm_comparison_figure.py --facets 10 45 82 --combined

Requirements:
    pip install pyvista numpy matplotlib numpy-stl scipy
"""

import argparse
import json
import os
import sys
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import numpy as np
from scipy.interpolate import interp1d
from stl import mesh as stlmesh

# ---- Global font setup: serif throughout ----
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
})

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

STL_PATH = os.path.join(PROJECT_ROOT, 'data', 'shape_models', 'Itokawa_low_poly.stl')
OBJ_PATH = os.path.join(PROJECT_ROOT, 'TEMPEST_RAD', 'diagnostics',
                         'DGL_KS_comparison', 'itokawa.obj')
BTPM_TEMP_PATH = os.path.join(PROJECT_ROOT, 'TEMPEST_RAD', 'diagnostics',
                               'DGL_KS_comparison', 'BinaryThermophysicalModel',
                               'Output_Data', 'itokawa_98_facets_T.npy')
BTPM_LT_PATH = os.path.join(PROJECT_ROOT, 'TEMPEST_RAD', 'diagnostics',
                              'DGL_KS_comparison', 'BinaryThermophysicalModel',
                              'Output_Data', 'itokawa_98_facets_lt.npy')
MATCHING_PATH = os.path.join(PROJECT_ROOT, 'TEMPEST_RAD', 'diagnostics',
                              'DGL_KS_comparison_outputs', 'DGL_KS_facet_matching.npz')

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'paper_figures', 'btpm_comparison')

# Colours for plot lines (original tab10 shades)
FACET_COLORS_HEX = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
# Lighter versions for 3D highlight edges (easier to see on dark maps)
FACET_3D_COLORS_HEX = ['#74b9e8', '#ff7f0e', '#7dcd7d', '#d62728', '#9467bd', '#8c564b']


# ============================================================================
# Data loading helpers
# ============================================================================

def load_stl(path):
    """Return (pv_mesh, stl_mesh_obj, centroids, normals, areas)."""
    import pyvista as pv
    shape = stlmesh.Mesh.from_file(path)
    n = shape.vectors.shape[0]
    vertices = shape.points.reshape(-1, 3)
    faces = [[3, 3 * i, 3 * i + 1, 3 * i + 2] for i in range(n)]
    pv_mesh = pv.PolyData(vertices, faces)

    centroids = np.zeros((n, 3))
    normals = np.zeros((n, 3))
    areas = np.zeros(n)
    for i in range(n):
        v1, v2, v3 = shape.vectors[i]
        centroids[i] = (v1 + v2 + v3) / 3.0
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)
        areas[i] = 0.5 * norm
        normals[i] = normal / norm if norm > 0 else np.array([0, 0, 1])
    return pv_mesh, shape, centroids, normals, areas


def load_btpm_temperatures(temp_path, lt_path, n_target_timesteps):
    """Load BTPM surface temps and resample to *n_target_timesteps*.
    Returns array (n_facets, n_target_timesteps).
    """
    T = np.load(temp_path)   # (n_timesteps, n_layers, n_facets)
    lt = np.load(lt_path)    # (n_timesteps, 3)

    surface = T[:, 0, :]
    lt_hours = lt[:, 0]
    if lt_hours.max() > 24.0:
        lt_hours = lt_hours / 3600.0

    idx = np.argsort(lt_hours)
    lt_sorted = lt_hours[idx]
    surf_sorted = surface[idx, :]

    target_times = np.linspace(0, 24, n_target_timesteps)
    n_facets = surf_sorted.shape[1]
    resampled = np.zeros((n_facets, n_target_timesteps))
    for fi in range(n_facets):
        f_interp = interp1d(lt_sorted, surf_sorted[:, fi], kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        resampled[fi, :] = f_interp(target_times)
    return resampled


def load_tempest_temperatures(path):
    """Load TEMPEST final-day surface temperatures.
    Returns array (n_facets, n_timesteps).
    """
    if path.endswith('.npz'):
        data = np.load(path, allow_pickle=True)
        for key in ('plotted_variable_array', 'final_day_temperatures', 'temperatures'):
            if key in data:
                arr = data[key]
                return arr if arr.ndim == 2 else arr.reshape(-1, 1)
        arr = data[list(data.keys())[0]]
        return arr if arr.ndim == 2 else arr.reshape(-1, 1)
    elif path.endswith('.h5') or path.endswith('.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            for loc in ('animation_io/plotted_variable_array',
                        'final_day_temperatures', 'temperatures',
                        'plotted_variable_array'):
                if loc in f:
                    return f[loc][:]
            return f[list(f.keys())[0]][:]
    else:
        raise ValueError(f"Unsupported TEMPEST output format: {path}")


def load_facet_matching(path):
    """Return dict {stl_idx: btpm_idx}."""
    data = np.load(path, allow_pickle=True)
    return {int(k): int(v) for k, v in data['matching'].item().items()}


def find_tempest_output(remote_dir, name_hint=None):
    """Find a TEMPEST output folder. If name_hint given, look for that folder first."""
    if not os.path.isdir(remote_dir):
        return None
    if name_hint:
        candidate = os.path.join(remote_dir, name_hint, 'animation_params.npz')
        if os.path.exists(candidate):
            return candidate
    # Fall back to latest animation_outputs_* or tempest_*
    anim_dirs = sorted(d for d in os.listdir(remote_dir)
                       if d.startswith('animation_outputs_') or d.startswith('tempest_'))
    for d in reversed(anim_dirs):
        npz = os.path.join(remote_dir, d, 'animation_params.npz')
        if os.path.exists(npz):
            return npz
    return None


# ============================================================================
# 3D rendering helpers  (off-screen with PyVista)
# ============================================================================

def _extract_facet_edges(stl_shape, facet_indices):
    """Return list of edge-pair lists for each facet.
    Each facet → 3 edges → each edge is (point_a, point_b).
    """
    edges_per_facet = []
    for fi in facet_indices:
        v1, v2, v3 = stl_shape.vectors[fi]
        edges_per_facet.append([(v1, v2), (v2, v3), (v3, v1)])
    return edges_per_facet


def render_3d_temperature_map(pv_mesh, stl_shape, temperatures, title, cmap, clim,
                               camera_position, window_size=(800, 800),
                               highlight_facets=None, facet_colors=None):
    """Render off-screen 3D frame with optional coloured-edge facet highlights.

    Parameters
    ----------
    highlight_facets : list of int
        Facet indices whose triangle edges should be drawn.
    facet_colors : list of str
        One colour (hex or named) per facet in *highlight_facets*.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('white')

    pv_mesh_copy = pv_mesh.copy()
    pv_mesh_copy.cell_data['Temperature (K)'] = temperatures
    pl.add_mesh(pv_mesh_copy, scalars='Temperature (K)', cmap=cmap,
                clim=clim, show_edges=False,
                show_scalar_bar=False)

    # Draw coloured edges around selected facets
    if highlight_facets is not None and facet_colors is not None:
        extent = np.ptp(pv_mesh_copy.points, axis=0).max()
        edge_width = min(6.0, extent * 0.1)

        # Compute camera screen-space axes so we can offset the label anchor
        # down-left to counteract add_point_labels' inherent up-right text offset.
        cam_p    = np.array(camera_position[0])
        focal_p  = np.array(camera_position[1])
        viewup_v = np.array(camera_position[2])
        view_dir = focal_p - cam_p
        view_dir /= np.linalg.norm(view_dir)
        screen_right = np.cross(view_dir, viewup_v)
        screen_right /= np.linalg.norm(screen_right)
        screen_up = np.cross(screen_right, view_dir)
        screen_up /= np.linalg.norm(screen_up)
        # Shift anchor down-left: text will then appear centred on the facet
        label_shift = -screen_right * extent * 0.02 - screen_up * extent * 0.025

        edges_list = _extract_facet_edges(stl_shape, highlight_facets)
        for fi_idx, (fi, edges) in enumerate(zip(highlight_facets, edges_list)):
            color = facet_colors[fi_idx % len(facet_colors)]
            for (p1, p2) in edges:
                line = pv.Line(p1, p2)
                pl.add_mesh(line, color=color, line_width=edge_width,
                            render_lines_as_tubes=True)
            # Numeric label: anchor shifted to place text visually at centroid
            centroid = np.mean(stl_shape.vectors[fi], axis=0)
            label_pos = centroid + label_shift
            pl.add_point_labels(
                [label_pos], [str(fi_idx + 1)],
                font_size=60, text_color=color,
                point_size=0, show_points=False, shape=None,
                render_points_as_spheres=False,
                always_visible=True, bold=True,
            )

    pl.camera_position = camera_position
    pl.camera.zoom(1.3)

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ============================================================================
# Local-time helpers
# ============================================================================

def _compute_noon_indices(normals, rotation_axis, sun_dir, n_timesteps):
    """Return per-facet index of local noon (max illumination).

    For each facet, rotate its normal through one full rotation and find
    the timestep where cos(incidence angle) is maximised.
    """
    from scipy.spatial.transform import Rotation as R
    n_facets = normals.shape[0]
    angles = np.linspace(0, 2 * np.pi, n_timesteps, endpoint=False)
    ax = rotation_axis / np.linalg.norm(rotation_axis)
    sun = sun_dir / np.linalg.norm(sun_dir)

    noon_idx = np.zeros(n_facets, dtype=int)
    for fi in range(n_facets):
        cos_inc = np.empty(n_timesteps)
        for ti, theta in enumerate(angles):
            rot = R.from_rotvec(ax * theta)
            n_rot = rot.apply(normals[fi])
            cos_inc[ti] = np.dot(n_rot, sun)
        noon_idx[fi] = int(np.argmax(cos_inc))
    return noon_idx


def _load_rotation_and_sun(path):
    """Return (rotation_axis, sun_dir) from NPZ metadata or sibling config.yaml.

    Some recent TEMPEST runs save only temperatures in NPZ, so we fall back to the
    run folder's copied config file.
    """
    if path.endswith('.npz'):
        npz = np.load(path, allow_pickle=True)
        if 'rotation_axis' in npz and 'sunlight_direction' in npz:
            return npz['rotation_axis'].astype(float), npz['sunlight_direction'].astype(float)

    # Fallback: parse config.yaml in same run directory
    cfg_path = os.path.join(os.path.dirname(path), 'config.yaml')
    if not os.path.exists(cfg_path):
        raise KeyError(
            f"Could not find rotation metadata in {path} and no sibling config.yaml present"
        )

    with open(cfg_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    sun_match = re.search(r'^\s*sunlight_direction\s*:\s*\[([^\]]+)\]', txt, flags=re.MULTILINE)
    ra_match = re.search(r'^\s*ra_degrees\s*:\s*([-+]?\d*\.?\d+)', txt, flags=re.MULTILINE)
    dec_match = re.search(r'^\s*dec_degrees\s*:\s*([-+]?\d*\.?\d+)', txt, flags=re.MULTILINE)

    if sun_match is None or ra_match is None or dec_match is None:
        raise KeyError(
            f"config.yaml at {cfg_path} is missing sunlight_direction and/or pole coordinates"
        )

    sun_vals = [float(x.strip()) for x in sun_match.group(1).split(',')]
    sun_dir = np.array(sun_vals, dtype=float)

    ra_deg = float(ra_match.group(1))
    dec_deg = float(dec_match.group(1))
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # Equatorial-to-Cartesian unit vector from (RA, Dec)
    rotation_axis = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ], dtype=float)

    return rotation_axis, sun_dir


def _roll_to_local_time(curve, noon_idx, n_timesteps):
    """Roll a 1-D array so that *noon_idx* maps to the midpoint (local noon = 12 h)."""
    mid = n_timesteps // 2
    shift = mid - noon_idx
    return np.roll(curve, shift)


# ============================================================================
# Diurnal plotting
# ============================================================================

def plot_diurnal_comparison(local_time_hours, explicit_curves, implicit_curves,
                            btpm_curves, facet_labels, facet_colors, output_path):
    """Diurnal temperature curves + residuals for all three models."""
    n = len(facet_labels)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
    if n == 1:
        axes = axes.reshape(-1, 1)

    fig.align_ylabels()

    for i, label in enumerate(facet_labels):
        ax_top = axes[0, i]
        ax_bot = axes[1, i]
        c = facet_colors[i % len(facet_colors)]

        ex_curve = explicit_curves[label]
        im_curve = implicit_curves[label]
        bt_curve = btpm_curves[label]

        ax_top.plot(local_time_hours, ex_curve, '-', color=c, linewidth=1.5,
                    label='TEMPEST (explicit)')
        ax_top.plot(local_time_hours, im_curve, '-.', color=c, linewidth=1.5,
                    alpha=0.85, label='TEMPEST (implicit)')
        ax_top.plot(local_time_hours, bt_curve, '--', color='k', linewidth=1.5,
                    label='Sorli BTPM')
        if i == 0:
            ax_top.set_ylabel('Temperature (K)')
        ax_top.set_title(f'Point {i + 1} ({label})', fontsize=11)
        ax_top.set_xlim(0, 24)
        ax_top.set_xticks([0, 6, 12, 18, 24])
        if i == 0:
            ax_top.legend(fontsize=12)
        ax_top.grid(True, alpha=0.3)

        res_ex = ex_curve - bt_curve
        res_im = im_curve - bt_curve
        ax_bot.plot(local_time_hours, res_ex, '-', color=c, linewidth=1.2,
                    label='Explicit \u2212 BTPM')
        ax_bot.plot(local_time_hours, res_im, '-.', color=c, linewidth=1.2,
                    alpha=0.85, label='Implicit \u2212 BTPM')
        ax_bot.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        if i == 0:
            ax_bot.set_ylabel('Residual (K)')
        ax_bot.set_xlabel('Local time (hours)')
        ax_bot.set_xlim(0, 24)
        ax_bot.set_xticks([0, 6, 12, 18, 24])
        ax_bot.grid(True, alpha=0.3)

        if i == 0:
            ax_bot.legend(fontsize=11, loc='lower left')

    fig.subplots_adjust(hspace=0.08, wspace=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved diurnal comparison \u2192 {output_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TEMPEST (explicit + implicit) vs Sorli BTPM comparison figure.')
    parser.add_argument('--tempest-explicit', type=str, default=None,
                        help='Path to explicit-solver animation_params.npz.')
    parser.add_argument('--tempest-implicit', type=str, default=None,
                        help='Path to implicit-solver animation_params.npz.')
    parser.add_argument('--tempest-output', type=str, default=None,
                        help='(Legacy) single TEMPEST output path.')
    parser.add_argument('--facets', type=int, nargs='+', default=[15, 45, 57],
                        help='STL facet indices for diurnal curves (default: 15 45 55).')
    parser.add_argument('--timestep', type=int, default=-1,
                        help='Timestep for 3D snapshots (0=first, -1=peak mean temp, default: -1).')
    parser.add_argument('--cmap', type=str, default='magma')
    parser.add_argument('--diff-cmap', type=str, default='viridis')
    parser.add_argument('--camera', type=float, nargs=9, default=None,
                        help='Camera: posX posY posZ focalX focalY focalZ upX upY upZ')
    parser.add_argument('--pick-camera', action='store_true')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--no-combined', action='store_true',
                        help='Skip generating the combined figure.')
    parser.add_argument('--window-size', type=int, nargs=2, default=[1400, 1400])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    remote_dir = os.path.join(PROJECT_ROOT, 'data', 'output', 'remote_outputs')

    # ------------------------------------------------------------------
    # 1. Load shape model
    # ------------------------------------------------------------------
    print("Loading shape model …")
    pv_mesh, stl_shape, stl_centroids, stl_normals, stl_areas = load_stl(STL_PATH)
    n_facets = len(stl_areas)
    print(f"  {n_facets} facets loaded from {os.path.basename(STL_PATH)}")

    # ------------------------------------------------------------------
    # 2. Load TEMPEST outputs (both solvers)
    # ------------------------------------------------------------------
    explicit_path = args.tempest_explicit
    implicit_path = args.tempest_implicit

    if explicit_path is None:
        explicit_path = find_tempest_output(remote_dir, 'tempest_explicit_selfheating')
    if implicit_path is None:
        implicit_path = find_tempest_output(remote_dir, 'tempest_implicit_selfheating')

    if explicit_path is None and implicit_path is None and args.tempest_output:
        explicit_path = args.tempest_output
        implicit_path = args.tempest_output
        print("WARNING: using same output for both solvers (legacy mode)")

    if explicit_path is None or implicit_path is None:
        print("ERROR: Could not locate both solver outputs.")
        print("  Provide --tempest-explicit and --tempest-implicit, or place outputs in")
        print("  data/output/remote_outputs/tempest_explicit_selfheating/ and")
        print("  data/output/remote_outputs/tempest_implicit_selfheating/")
        sys.exit(1)

    print(f"Loading TEMPEST explicit: {explicit_path}")
    explicit_temps = load_tempest_temperatures(explicit_path)
    print(f"  Shape: {explicit_temps.shape}")

    print(f"Loading TEMPEST implicit: {implicit_path}")
    implicit_temps = load_tempest_temperatures(implicit_path)
    print(f"  Shape: {implicit_temps.shape}")

    n_timesteps = explicit_temps.shape[1]
    if implicit_temps.shape[1] != n_timesteps:
        print(f"  Resampling implicit ({implicit_temps.shape[1]}) → ({n_timesteps}) timesteps")
        old_t = np.linspace(0, 24, implicit_temps.shape[1])
        new_t = np.linspace(0, 24, n_timesteps)
        resampled = np.zeros((n_facets, n_timesteps))
        for fi in range(n_facets):
            resampled[fi, :] = np.interp(new_t, old_t, implicit_temps[fi, :])
        implicit_temps = resampled

    # ------------------------------------------------------------------
    # 3. Load BTPM temperatures
    # ------------------------------------------------------------------
    print("Loading BTPM temperatures …")
    btpm_temps_raw = load_btpm_temperatures(BTPM_TEMP_PATH, BTPM_LT_PATH, n_timesteps)
    print(f"  BTPM resampled shape: {btpm_temps_raw.shape}")

    # ------------------------------------------------------------------
    # 4. Facet matching
    # ------------------------------------------------------------------
    print("Loading facet matching …")
    matching = load_facet_matching(MATCHING_PATH)
    btpm_temps = np.full_like(explicit_temps, np.nan)
    for stl_idx, btpm_idx in matching.items():
        if stl_idx < n_facets and btpm_idx < btpm_temps_raw.shape[0]:
            btpm_temps[stl_idx, :] = btpm_temps_raw[btpm_idx, :]
    n_matched = np.sum(~np.isnan(btpm_temps[:, 0]))
    print(f"  Matched {n_matched}/{n_facets} facets")

    # ------------------------------------------------------------------
    # 5. Timestep for 3D snapshots
    # ------------------------------------------------------------------
    if args.timestep == -1:
        mean_temps = np.nanmean(explicit_temps, axis=0)
        ts = int(np.argmax(mean_temps))
    else:
        ts = args.timestep % n_timesteps
    print(f"Using timestep {ts}/{n_timesteps} for 3D snapshots")

    ex_snap = explicit_temps[:, ts]
    im_snap = implicit_temps[:, ts]
    bt_snap = btpm_temps[:, ts]

    # Difference maps are RMS over all timesteps (per facet), not single-snapshot deltas.
    diff_ex_bt_rms = np.sqrt(np.nanmean((explicit_temps - btpm_temps) ** 2, axis=1))
    diff_im_bt_rms = np.sqrt(np.nanmean((implicit_temps - btpm_temps) ** 2, axis=1))
    diff_ex_im_rms = np.sqrt(np.nanmean((explicit_temps - implicit_temps) ** 2, axis=1))

    # ------------------------------------------------------------------
    # 6. Select facets for diurnal curves
    # ------------------------------------------------------------------
    if args.facets is not None:
        selected_facets = args.facets
    else:
        mean_t = np.nanmean(explicit_temps, axis=1)
        valid = ~np.isnan(btpm_temps[:, 0])
        valid_idx = np.where(valid)[0]
        means_valid = mean_t[valid_idx]
        order = np.argsort(means_valid)
        selected_facets = [
            int(valid_idx[order[-1]]),
            int(valid_idx[order[len(order) // 2]]),
            int(valid_idx[order[0]]),
        ]
        print(f"Auto-selected facets: {selected_facets}")

    for fi in selected_facets:
        if fi >= n_facets:
            print(f"ERROR: Facet {fi} out of range (0–{n_facets - 1})")
            sys.exit(1)
        if np.isnan(btpm_temps[fi, 0]):
            print(f"WARNING: Facet {fi} has no BTPM match")

    facet_labels = [f'Facet {fi}' for fi in selected_facets]
    facet_colors = [FACET_COLORS_HEX[i % len(FACET_COLORS_HEX)]
                    for i in range(len(selected_facets))]
    facet_3d_colors = [FACET_3D_COLORS_HEX[i % len(FACET_3D_COLORS_HEX)]
                       for i in range(len(selected_facets))]
    print(f"Diurnal comparison facets: {selected_facets}")

    # ------------------------------------------------------------------
    # 7. Camera
    # ------------------------------------------------------------------
    if args.pick_camera:
        import pyvista as pv
        print("Opening interactive window …")
        pl = pv.Plotter()
        pl.add_mesh(pv_mesh, color='lightgrey', show_edges=True)
        pl.show()
        cam_pos = pl.camera_position
        print(f"Camera: {cam_pos}")
    elif args.camera is not None:
        c = args.camera
        cam_pos = [(c[0], c[1], c[2]), (c[3], c[4], c[5]), (c[6], c[7], c[8])]
    else:
        extent = np.ptp(pv_mesh.points, axis=0).max()
        cam_pos = [(extent * 0.8, extent * 2.5, extent * 1.0),
                    (0, 0, 0),
                    (0, 0, 1)]

    wsize = tuple(args.window_size)

    # ------------------------------------------------------------------
    # 8. Colour limits
    # ------------------------------------------------------------------
    valid_mask = ~np.isnan(bt_snap)
    all_temps = np.concatenate([ex_snap[valid_mask], im_snap[valid_mask],
                                 bt_snap[valid_mask]])
    vmin_t = float(np.floor(all_temps.min()))
    vmax_t = float(np.ceil(all_temps.max()))
    temp_clim = (vmin_t, vmax_t)

    max_rms_bt = float(max(np.nanmax(diff_ex_bt_rms),
                           np.nanmax(diff_im_bt_rms)))
    max_rms_solver = float(np.nanmax(diff_ex_im_rms))
    if max_rms_solver < 0.01:
        max_rms_solver = 0.01
    # Common scale across all RMS difference panels
    max_rms_all = max(max_rms_bt, max_rms_solver)
    diff_clim = (0.0, max_rms_all)

    print(f"Temperature range: {vmin_t:.0f}–{vmax_t:.0f} K")
    print(f"Max RMS(Explicit−BTPM): {max_rms_bt:.2f} K")
    print(f"Max RMS(Explicit−Implicit): {max_rms_solver:.2f} K")
    print(f"Common RMS diff scale: 0–{max_rms_all:.2f} K")

    # ------------------------------------------------------------------
    # 9. Render 3D panels
    # ------------------------------------------------------------------
    render_kw = dict(stl_shape=stl_shape, camera_position=cam_pos,
                     window_size=wsize, highlight_facets=selected_facets,
                     facet_colors=facet_3d_colors)

    print("Rendering TEMPEST explicit temperature map …")
    img_explicit = render_3d_temperature_map(
        pv_mesh, temperatures=ex_snap, title='TEMPEST (Explicit)',
        cmap=args.cmap, clim=temp_clim, **render_kw)

    print("Rendering TEMPEST implicit temperature map …")
    img_implicit = render_3d_temperature_map(
        pv_mesh, temperatures=im_snap, title='TEMPEST (Implicit)',
        cmap=args.cmap, clim=temp_clim, **render_kw)

    print("Rendering BTPM temperature map …")
    img_btpm = render_3d_temperature_map(
        pv_mesh, temperatures=bt_snap, title='Sorli BTPM',
        cmap=args.cmap, clim=temp_clim, **render_kw)

    print("Rendering RMS(Explicit−BTPM) difference …")
    img_diff_ex_bt = render_3d_temperature_map(
        pv_mesh, temperatures=np.where(np.isnan(diff_ex_bt_rms), 0, diff_ex_bt_rms),
        title='RMS(Explicit − BTPM)', cmap=args.diff_cmap, clim=diff_clim,
        **render_kw)

    print("Rendering RMS(Implicit−BTPM) difference …")
    img_diff_im_bt = render_3d_temperature_map(
        pv_mesh, temperatures=np.where(np.isnan(diff_im_bt_rms), 0, diff_im_bt_rms),
        title='RMS(Implicit − BTPM)', cmap=args.diff_cmap, clim=diff_clim,
        **render_kw)

    print("Rendering RMS(Explicit−Implicit) difference …")
    img_diff_ex_im = render_3d_temperature_map(
        pv_mesh, temperatures=diff_ex_im_rms,
        title='RMS(Explicit − Implicit)', cmap=args.diff_cmap, clim=diff_clim,
        **render_kw)

    # Save individual panels
    for img, name in [
        (img_explicit,    'tempest_explicit_map'),
        (img_implicit,    'tempest_implicit_map'),
        (img_btpm,        'btpm_temperature_map'),
        (img_diff_ex_bt,  'diff_explicit_btpm'),
        (img_diff_im_bt,  'diff_implicit_btpm'),
        (img_diff_ex_im,  'diff_explicit_implicit'),
    ]:
        path = os.path.join(args.output_dir, f'{name}.png')
        plt.imsave(path, img)
        print(f"  Saved → {path}")

    # ------------------------------------------------------------------
    # 10. Compute per-facet local time & diurnal comparison plot
    # ------------------------------------------------------------------
    print("Computing per-facet local solar time …")
    rotation_axis, sun_dir = _load_rotation_and_sun(explicit_path)
    noon_indices = _compute_noon_indices(stl_normals, rotation_axis, sun_dir, n_timesteps)

    # Build a common local-time axis: 0 → 24 h with noon at midpoint
    local_time_hours = np.linspace(0, 24, n_timesteps, endpoint=False)

    # Roll each selected facet's curves so peak-insolation maps to 12 h
    ex_curves = {}
    im_curves = {}
    bt_curves = {}
    for fi, l in zip(selected_facets, facet_labels):
        ni = noon_indices[fi]
        ex_curves[l] = _roll_to_local_time(explicit_temps[fi, :], ni, n_timesteps)
        im_curves[l] = _roll_to_local_time(implicit_temps[fi, :], ni, n_timesteps)
        bt_curves[l] = _roll_to_local_time(btpm_temps[fi, :],     ni, n_timesteps)

    diurnal_path = os.path.join(args.output_dir, 'diurnal_comparison.png')
    plot_diurnal_comparison(local_time_hours, ex_curves, im_curves, bt_curves,
                            facet_labels, facet_colors, diurnal_path)

    # ------------------------------------------------------------------
    # 11. Combined figure (default — skip with --no-combined)
    # ------------------------------------------------------------------
    if not args.no_combined:
        print("Assembling combined figure …")

        # Helper: crop whitespace from off-screen PyVista renders
        def _crop_3d(img, frac=0.27):
            """Trim *frac* of the height from top and bottom."""
            h = img.shape[0]
            t = int(h * frac)
            b = h - t
            return img[t:b, :, :]

        # Layout: outer GridSpec has 8 rows; the last row holds a nested
        # GridSpec for the plots so they can have their own independent wspace.
        #   Row 0: temperature maps
        #   Row 1: temp colourbar
        #   Row 2: spacer
        #   Row 3: difference maps
        #   Row 4: spacer
        #   Row 5: diff colourbar
        #   Row 6: spacer (before graphs)
        #   Row 7: plot section (nested: diurnal | spacer | residuals)
        fig = plt.figure(figsize=(18, 17))
        gs = gridspec.GridSpec(8, 3,
                               height_ratios=[2.0, 0.13, 0.55, 2.0, 0.05, 0.13, 0.70, 4.0],
                               hspace=0.0, wspace=0.05)

        # Nested GridSpec for the plot section — own wspace independent of 3D rows
        gs_plots = gridspec.GridSpecFromSubplotSpec(
            3, 3,
            subplot_spec=gs[7, :],
            height_ratios=[2.5, 0.20, 0.9],
            hspace=0.0, wspace=0.10)

        # ---- Row 0: temperature maps ----
        row0_axes = []
        for col, (img, title) in enumerate([
            (img_explicit, 'TEMPEST (Explicit)'),
            (img_implicit, 'TEMPEST (Implicit)'),
            (img_btpm,     'Sorli BTPM'),
        ]):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(_crop_3d(img))
            ax.set_title(title)
            ax.axis('off')
            row0_axes.append(ax)

        # Shared temp colourbar (row 1, centre column only)
        cbar_ax0 = fig.add_subplot(gs[1, 1])
        norm_t = mcolors.Normalize(vmin=temp_clim[0], vmax=temp_clim[1])
        sm_t = plt.cm.ScalarMappable(cmap=args.cmap, norm=norm_t)
        sm_t.set_array([])
        cb0 = fig.colorbar(sm_t, cax=cbar_ax0, orientation='horizontal')
        cb0.set_label('Temperature (K)')

        # ---- Row 3: difference maps ----
        for col, (img, title) in enumerate([
            (img_diff_ex_bt, 'RMS(Explicit \u2212 BTPM)'),
            (img_diff_im_bt, 'RMS(Implicit \u2212 BTPM)'),
            (img_diff_ex_im, 'RMS(Explicit \u2212 Implicit)'),
        ]):
            ax = fig.add_subplot(gs[3, col])
            ax.imshow(_crop_3d(img))
            ax.set_title(title)
            ax.axis('off')

        # Shared diff colourbar (row 5, centre column only)
        cbar_ax1 = fig.add_subplot(gs[5, 1])
        norm_d = mcolors.Normalize(vmin=diff_clim[0], vmax=diff_clim[1])
        sm_d = plt.cm.ScalarMappable(cmap=args.diff_cmap, norm=norm_d)
        sm_d.set_array([])
        cb1 = fig.colorbar(sm_d, cax=cbar_ax1, orientation='horizontal')
        cb1.set_label('RMS temperature difference (K)')

        # ---- Plots row 0 (gs_plots row 0): diurnal curves ----
        for i, (fi, label) in enumerate(zip(selected_facets, facet_labels)):
            ax = fig.add_subplot(gs_plots[0, i])
            c = facet_colors[i]
            ax.plot(local_time_hours, ex_curves[label], '-', color=c,
                    linewidth=1.5, label='Explicit')
            ax.plot(local_time_hours, im_curves[label], '-.', color=c,
                    linewidth=1.5, alpha=0.85, label='Implicit')
            ax.plot(local_time_hours, bt_curves[label], '--', color='k',
                    linewidth=1.5, label='Sorli BTPM')
            if i == 0:
                ax.set_ylabel('Temperature (K)')
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            if i == 0:
                ax.legend(fontsize=12, loc='best')
            ax.set_title(f'Point {i + 1} ({label})')
            ax.grid(True, alpha=0.3)

        # ---- Plots row 2 (gs_plots row 2): residuals ----
        for i, (fi, label) in enumerate(zip(selected_facets, facet_labels)):
            ax = fig.add_subplot(gs_plots[2, i])
            c = facet_colors[i]
            res_ex = ex_curves[label] - bt_curves[label]
            res_im = im_curves[label] - bt_curves[label]
            ax.plot(local_time_hours, res_ex, '-', color=c, linewidth=1.2, label='Expl\u2212BTPM')
            ax.plot(local_time_hours, res_im, '-.', color=c, linewidth=1.2,
                    alpha=0.85, label='Impl\u2212BTPM')
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            if i == 0:
                ax.set_ylabel('Residual (K)')
            ax.set_xlabel('Local time (hours)')
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=11, loc='lower left')

        combined_path = os.path.join(args.output_dir, 'combined_comparison_figure.png')
        fig.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"  Saved combined figure → {combined_path}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done!  Output in:", args.output_dir)
    print("=" * 60)
    for name in ['tempest_explicit_map', 'tempest_implicit_map', 'btpm_temperature_map',
                  'diff_explicit_btpm', 'diff_implicit_btpm', 'diff_explicit_implicit',
                  'diurnal_comparison']:
        print(f"  {name}.png")
    if not args.no_combined:
        print("  combined_comparison_figure.png")
    print(f"\nFacets: {selected_facets}")
    print(f"Timestep: {ts}")
    print(f"Temp range: {temp_clim[0]:.0f}–{temp_clim[1]:.0f} K")
    print(f"RMS(Explicit−BTPM) max: {max_rms_bt:.2f} K")
    print(f"RMS(Explicit−Implicit) max: {max_rms_solver:.2f} K")


if __name__ == '__main__':
    main()
