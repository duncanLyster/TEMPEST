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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.interpolate import interp1d
from stl import mesh as stlmesh

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

STL_PATH = os.path.join(PROJECT_ROOT, 'data', 'shape_models', 'Itokawa_low_poly.stl')
OBJ_PATH = os.path.join(PROJECT_ROOT, 'TEMPEST_RAD', 'diagnostics',
                         'DGL_KS_comparison', 'itokawa.obj')
BTPM_TEMP_PATH = os.path.join(PROJECT_ROOT, 'TEMPEST_RAD', 'diagnostics',
                               'DGL_KS_comparison', 'BinaryThermophysicalModel',
                               'Output_Data', 'itokawa_100_facets_T.npy')
BTPM_LT_PATH = os.path.join(PROJECT_ROOT, 'TEMPEST_RAD', 'diagnostics',
                              'DGL_KS_comparison', 'BinaryThermophysicalModel',
                              'Output_Data', 'itokawa_100_facets_lt.npy')
MATCHING_PATH = os.path.join(PROJECT_ROOT, 'TEMPEST_RAD', 'diagnostics',
                              'DGL_KS_comparison_outputs', 'DGL_KS_facet_matching.npz')

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'paper_figures', 'btpm_comparison')

# Consistent colours for selected facets (tab10 first 6)
FACET_COLORS_HEX = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


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
                scalar_bar_args={'title': 'Temperature (K)',
                                 'title_font_size': 14,
                                 'label_font_size': 12,
                                 'color': 'black',
                                 'width': 0.4,
                                 'position_x': 0.3})

    # Draw coloured edges around selected facets
    if highlight_facets is not None and facet_colors is not None:
        extent = np.ptp(pv_mesh_copy.points, axis=0).max()
        edge_width = max(5.0, extent * 0.08)
        edges_list = _extract_facet_edges(stl_shape, highlight_facets)
        for fi_idx, (fi, edges) in enumerate(zip(highlight_facets, edges_list)):
            color = facet_colors[fi_idx % len(facet_colors)]
            for (p1, p2) in edges:
                line = pv.Line(p1, p2)
                pl.add_mesh(line, color=color, line_width=edge_width,
                            render_lines_as_tubes=True)
            # Numeric label above centroid
            centroid = np.mean(stl_shape.vectors[fi], axis=0)
            normal = np.cross(
                stl_shape.vectors[fi][1] - stl_shape.vectors[fi][0],
                stl_shape.vectors[fi][2] - stl_shape.vectors[fi][0])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            label_pos = centroid + normal * extent * 0.03
            pl.add_point_labels(
                [label_pos], [str(fi_idx + 1)],
                font_size=20, text_color=color,
                point_size=0, shape=None,
                render_points_as_spheres=False,
                always_visible=True, bold=True,
            )

    pl.add_text(title, position='upper_edge', font_size=14, color='black')
    pl.camera_position = camera_position
    pl.camera.zoom(1.0)

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ============================================================================
# Diurnal plotting
# ============================================================================

def plot_diurnal_comparison(time_hours, explicit_curves, implicit_curves,
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

        ax_top.plot(time_hours, ex_curve, '-', color=c, linewidth=1.5,
                    label='TEMPEST (explicit)')
        ax_top.plot(time_hours, im_curve, '-.', color=c, linewidth=1.5,
                    alpha=0.85, label='TEMPEST (implicit)')
        ax_top.plot(time_hours, bt_curve, '--', color='k', linewidth=1.5,
                    label='Sorli BTPM')
        ax_top.set_ylabel('Temperature (K)')
        ax_top.set_title(f'Point {i + 1} ({label})', fontsize=11)
        if i == 0:
            ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.3)

        res_ex = ex_curve - bt_curve
        res_im = im_curve - bt_curve
        ax_bot.plot(time_hours, res_ex, '-', color=c, linewidth=1.2,
                    label='Explicit − BTPM')
        ax_bot.plot(time_hours, res_im, '-.', color=c, linewidth=1.2,
                    alpha=0.85, label='Implicit − BTPM')
        ax_bot.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax_bot.set_ylabel('Residual (K)')
        ax_bot.set_xlabel('Local time (hours)')
        ax_bot.grid(True, alpha=0.3)

        rms_ex = np.sqrt(np.nanmean(res_ex ** 2))
        rms_im = np.sqrt(np.nanmean(res_im ** 2))
        ax_bot.text(0.97, 0.92,
                    f'RMS ex={rms_ex:.2f} K\nRMS im={rms_im:.2f} K',
                    transform=ax_bot.transAxes, fontsize=8,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        if i == 0:
            ax_bot.legend(fontsize=8, loc='lower left')

    fig.subplots_adjust(hspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved diurnal comparison → {output_path}")
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
    parser.add_argument('--facets', type=int, nargs='+', default=None,
                        help='STL facet indices for diurnal curves (e.g. --facets 10 45 82).')
    parser.add_argument('--timestep', type=int, default=0,
                        help='Timestep for 3D snapshots (0=first, -1=peak mean temp).')
    parser.add_argument('--cmap', type=str, default='magma')
    parser.add_argument('--diff-cmap', type=str, default='coolwarm')
    parser.add_argument('--camera', type=float, nargs=9, default=None,
                        help='Camera: posX posY posZ focalX focalY focalZ upX upY upZ')
    parser.add_argument('--pick-camera', action='store_true')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--combined', action='store_true',
                        help='Save a single combined figure with all panels.')
    parser.add_argument('--window-size', type=int, nargs=2, default=[1000, 1000])
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
    diff_ex_bt = ex_snap - bt_snap
    diff_im_bt = im_snap - bt_snap
    diff_ex_im = ex_snap - im_snap

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
                    (0, 0, -1)]

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

    abs_diff_max_bt = float(max(np.nanmax(np.abs(diff_ex_bt)),
                                np.nanmax(np.abs(diff_im_bt))))
    diff_clim_bt = (-abs_diff_max_bt, abs_diff_max_bt)

    abs_diff_max_solver = float(np.nanmax(np.abs(diff_ex_im)))
    # Use a symmetric limit but ensure it's not zero
    if abs_diff_max_solver < 0.01:
        abs_diff_max_solver = 0.01
    diff_clim_solver = (-abs_diff_max_solver, abs_diff_max_solver)

    print(f"Temperature range: {vmin_t:.0f}–{vmax_t:.0f} K")
    print(f"Max |Explicit−BTPM|: {abs_diff_max_bt:.1f} K")
    print(f"Max |Explicit−Implicit|: {abs_diff_max_solver:.2f} K")

    # ------------------------------------------------------------------
    # 9. Render 3D panels
    # ------------------------------------------------------------------
    render_kw = dict(stl_shape=stl_shape, camera_position=cam_pos,
                     window_size=wsize, highlight_facets=selected_facets,
                     facet_colors=facet_colors)

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

    print("Rendering Explicit−BTPM difference …")
    img_diff_ex_bt = render_3d_temperature_map(
        pv_mesh, temperatures=np.where(np.isnan(diff_ex_bt), 0, diff_ex_bt),
        title='Explicit − BTPM', cmap=args.diff_cmap, clim=diff_clim_bt,
        **render_kw)

    print("Rendering Implicit−BTPM difference …")
    img_diff_im_bt = render_3d_temperature_map(
        pv_mesh, temperatures=np.where(np.isnan(diff_im_bt), 0, diff_im_bt),
        title='Implicit − BTPM', cmap=args.diff_cmap, clim=diff_clim_bt,
        **render_kw)

    print("Rendering Explicit−Implicit difference …")
    img_diff_ex_im = render_3d_temperature_map(
        pv_mesh, temperatures=diff_ex_im,
        title='Explicit − Implicit', cmap=args.diff_cmap, clim=diff_clim_solver,
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
    # 10. Diurnal comparison plot
    # ------------------------------------------------------------------
    time_hours = np.linspace(0, 24, n_timesteps)
    ex_curves = {l: explicit_temps[fi, :] for fi, l in zip(selected_facets, facet_labels)}
    im_curves = {l: implicit_temps[fi, :] for fi, l in zip(selected_facets, facet_labels)}
    bt_curves = {l: btpm_temps[fi, :]     for fi, l in zip(selected_facets, facet_labels)}

    diurnal_path = os.path.join(args.output_dir, 'diurnal_comparison.png')
    plot_diurnal_comparison(time_hours, ex_curves, im_curves, bt_curves,
                            facet_labels, facet_colors, diurnal_path)

    # ------------------------------------------------------------------
    # 11. Combined figure
    # ------------------------------------------------------------------
    if args.combined:
        print("Assembling combined figure …")
        fig = plt.figure(figsize=(20, 22))
        gs = gridspec.GridSpec(4, 3,
                               height_ratios=[4, 4, 3, 1],
                               hspace=0.22, wspace=0.12)

        # Row 0: temperature maps
        for col, (img, title) in enumerate([
            (img_explicit, 'TEMPEST (Explicit)'),
            (img_implicit, 'TEMPEST (Implicit)'),
            (img_btpm,     'Sorli BTPM'),
        ]):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis('off')

        # Row 1: difference maps
        for col, (img, title) in enumerate([
            (img_diff_ex_bt, f'Explicit − BTPM (±{abs_diff_max_bt:.1f} K)'),
            (img_diff_im_bt, f'Implicit − BTPM (±{abs_diff_max_bt:.1f} K)'),
            (img_diff_ex_im, f'Explicit − Implicit (±{abs_diff_max_solver:.2f} K)'),
        ]):
            ax = fig.add_subplot(gs[1, col])
            ax.imshow(img)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

        # Row 2: diurnal curves
        for i, (fi, label) in enumerate(zip(selected_facets, facet_labels)):
            ax = fig.add_subplot(gs[2, i])
            c = facet_colors[i]
            ax.plot(time_hours, explicit_temps[fi, :], '-', color=c,
                    linewidth=1.5, label='Explicit')
            ax.plot(time_hours, implicit_temps[fi, :], '-.', color=c,
                    linewidth=1.5, alpha=0.85, label='Implicit')
            ax.plot(time_hours, btpm_temps[fi, :], '--', color='k',
                    linewidth=1.5, label='Sorli BTPM')
            ax.set_ylabel('Temperature (K)')
            if i == 0:
                ax.legend(fontsize=9)
            ax.set_title(f'Point {i + 1} ({label})', fontsize=11)
            ax.grid(True, alpha=0.3)

        # Row 3: residuals
        for i, (fi, label) in enumerate(zip(selected_facets, facet_labels)):
            ax = fig.add_subplot(gs[3, i])
            c = facet_colors[i]
            res_ex = explicit_temps[fi, :] - btpm_temps[fi, :]
            res_im = implicit_temps[fi, :] - btpm_temps[fi, :]
            ax.plot(time_hours, res_ex, '-', color=c, linewidth=1.2, label='Expl−BTPM')
            ax.plot(time_hours, res_im, '-.', color=c, linewidth=1.2,
                    alpha=0.85, label='Impl−BTPM')
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ax.set_ylabel('Residual (K)')
            ax.set_xlabel('Local time (hours)')
            ax.grid(True, alpha=0.3)
            rms_ex = np.sqrt(np.nanmean(res_ex ** 2))
            rms_im = np.sqrt(np.nanmean(res_im ** 2))
            ax.text(0.97, 0.92,
                    f'RMS ex={rms_ex:.2f} K\nRMS im={rms_im:.2f} K',
                    transform=ax.transAxes, fontsize=8,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            if i == 0:
                ax.legend(fontsize=8, loc='lower left')

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
    if args.combined:
        print("  combined_comparison_figure.png")
    print(f"\nFacets: {selected_facets}")
    print(f"Timestep: {ts}")
    print(f"Temp range: {temp_clim[0]:.0f}–{temp_clim[1]:.0f} K")
    print(f"|Explicit−BTPM| max: {abs_diff_max_bt:.1f} K")
    print(f"|Explicit−Implicit| max: {abs_diff_max_solver:.2f} K")


if __name__ == '__main__':
    main()
