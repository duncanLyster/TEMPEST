#!/usr/bin/env python3
"""
Script to sweep kernel_profile_angle_degrees in TEMPEST and plot min/max and diurnal curves for a chosen facet.
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is on PYTHONPATH
root = Path(__file__).resolve().parent
sys.path.append(str(root))

from tempest import read_shape_model
from src.utilities.config import Config
from src.model.simulation import Simulation, ThermalData
from src.model.facet import Facet
from src.model.view_factors import calculate_and_cache_visible_facets, calculate_view_factors
from src.model.insolation import calculate_insolation
from src.model.solvers import TemperatureSolverFactory
from numba.typed import List
from src.utilities.utils import calculate_rotation_matrix

def run_for_angle(angle, config_path, facet_index=37):
    # Load and override config
    base_cfg = yaml.safe_load(open(config_path, 'r'))
    base_cfg['kernel_profile_angle_degrees'] = angle
    # Force local mode, leave silent off so main prints
    base_cfg['silent_mode'] = False
    base_cfg['remote'] = False
    # Write temporary config file
    tmp_cfg = root / f"temp_config_{angle}.yaml"
    yaml.safe_dump(base_cfg, open(tmp_cfg, 'w'))

    # Initialize config and simulation
    config = Config(str(tmp_cfg))
    # Override roughness and output settings
    config.apply_kernel_based_roughness = True
    config.silent_mode = True
    config.remote = False
    simulation = Simulation(config)

    # Read shape model
    shape_model = read_shape_model(
        config.path_to_shape_model_file,
        simulation.timesteps_per_day,
        simulation.n_layers,
        simulation.max_days,
        config.calculate_energy_terms
    )

    # Clear cached canonical subfacet data so view factors regenerate correctly for this angle
    if hasattr(Facet, '_canonical_subfacet_mesh'):
        Facet._canonical_subfacet_mesh = None
    if hasattr(Facet, '_canonical_mesh_params'):
        Facet._canonical_mesh_params = None
    if hasattr(Facet, '_canonical_F_ss'):
        delattr(Facet, '_canonical_F_ss')
    if hasattr(Facet, '_canonical_F_sd'):
        delattr(Facet, '_canonical_F_sd')

    # Prepare thermal data
    thermal_data = ThermalData(
        len(shape_model),
        simulation.timesteps_per_day,
        simulation.n_layers,
        simulation.max_days,
        config.calculate_energy_terms
    )

    # Apply spherical-cap roughness
    for facet in shape_model:
        facet.generate_spherical_depression(config, simulation)
    # Precompute dome-to-parent view factors
    import math
    parent_vertices = np.array([f.vertices for f in shape_model])
    parent_areas = np.array([f.area for f in shape_model])
    for facet in shape_model:
        dome = facet.dome_facets
        M = len(dome)
        subj_vertices = np.zeros((M, 3, 3), dtype=np.float64)
        subj_normals  = np.zeros((M, 3), dtype=np.float64)
        subj_areas    = np.zeros(M, dtype=np.float64)
        parent_radius = math.sqrt(facet.area / math.pi)
        for j, entry in enumerate(dome):
            local_scaled = entry['vertices'] * (config.kernel_dome_radius_factor * parent_radius)
            world_tris   = (facet.dome_rotation.dot(local_scaled.T)).T + facet.position
            subj_vertices[j] = world_tris
            subj_normals[j]  = facet.dome_rotation.dot(entry['normal'])
            subj_areas[j]    = entry['area'] * (config.kernel_dome_radius_factor * parent_radius)**2
        F_dp = np.zeros((M, len(shape_model)), dtype=np.float64)
        for j in range(M):
            F_dp[j, :] = calculate_view_factors(
                subj_vertices[j], subj_normals[j], subj_areas[j],
                parent_vertices, parent_areas, config.vf_rays
            )
        facet.depression_global_F_dp = F_dp

    # Visible facets
    positions = np.array([f.position for f in shape_model])
    normals   = np.array([f.normal   for f in shape_model])
    vertices  = np.array([f.vertices for f in shape_model])
    visible_indices = calculate_and_cache_visible_facets(
        config.silent_mode, shape_model, positions, normals, vertices, config
    )
    thermal_data.set_visible_facets(visible_indices)
    for i, facet in enumerate(shape_model):
        facet.visible_facets = visible_indices[i]

    # Insolation
    thermal_data = calculate_insolation(thermal_data, shape_model, simulation, config)
    # Kernel-based roughness coupling (populate sub-facet insolation)
    parent_areas = np.array([f.area for f in shape_model], dtype=np.float64)
    for t in range(simulation.timesteps_per_day):
        base_ins = thermal_data.insolation[:, t].copy()
        angle_t = (2 * np.pi / simulation.timesteps_per_day) * t
        R = calculate_rotation_matrix(simulation.rotation_axis, angle_t)
        world_dir = R.T.dot(simulation.sunlight_direction)
        for i, facet in enumerate(shape_model):
            flux_net = base_ins[i]
            cos_parent = np.dot(facet.normal, world_dir)
            if cos_parent <= 0:
                continue
            flux0 = flux_net / ((1 - simulation.albedo) * cos_parent)
            facet.parent_incident_energy_packets.append((flux0, world_dir, 'solar'))
            facet.process_intra_depression_energetics(config, simulation)
            # Record sub-facet insolation for this timestep
            facet.depression_thermal_data.insolation[:, t] = facet._last_absorbed_solar
            # Directional coupling back to parent insolation
            out_vis = np.array([facet.depression_outgoing_flux_distribution['scattered_visible'].get(j, 0.0) for j in range(facet.depression_global_F_dp.shape[0])], dtype=np.float64)
            inc_vis = facet.depression_global_F_dp.T.dot(out_vis)
            thermal_data.insolation[:, t] += inc_vis / parent_areas
            out_th = np.array([facet.depression_outgoing_flux_distribution['thermal'].get(j, 0.0) for j in range(facet.depression_global_F_dp.shape[0])], dtype=np.float64)
            inc_th = facet.depression_global_F_dp.T.dot(out_th)
            thermal_data.insolation[:, t] += inc_th / parent_areas

    # Initialize and run solver
    solver = TemperatureSolverFactory.create(config.temp_solver)
    thermal_data = solver.initialize_temperatures(thermal_data, simulation, config)
    result = solver.solve(thermal_data, shape_model, simulation, config)

    # Sub-facet solves and aggregate final-day temperatures
    sub_solver = TemperatureSolverFactory.create(config.temp_solver)
    # Suppress prints during sub-facet solves
    old_silent = config.silent_mode
    config.silent_mode = True
    for facet in shape_model:
        sub_solver.initialize_temperatures(facet.depression_thermal_data, simulation, config)
        facet.depression_temperature_result = sub_solver.solve(
            facet.depression_thermal_data,
            facet.sub_facets,
            simulation,
            config
        )
    # Restore print settings
    config.silent_mode = old_silent
    # Aggregate
    tpd = simulation.timesteps_per_day
    n_parents = len(shape_model)
    parent_t_final = np.zeros((n_parents, tpd), dtype=np.float64)
    mesh = Facet._canonical_subfacet_mesh
    for i, facet in enumerate(shape_model):
        sub_temps = facet.depression_temperature_result['final_day_temperatures']
        area_vec  = np.array([entry['area'] * facet.area for entry in mesh], dtype=np.float64)
        parent_t_final[i, :] = np.power((area_vec[:, None] * np.power(sub_temps, 4)).sum(axis=0) / area_vec.sum(), 0.25)

    # Return diurnal curve for the target facet
    os.remove(tmp_cfg)
    return parent_t_final[facet_index]


def main():
    # Configuration
    config_path = root / 'data' / 'config' / 'example_config.yaml'
    facet_index = 3
    # Sweep kernel profile angle from 0° to 90° in 10° increments
    angles = list(range(0, 91, 10))

    # Data storage
    summary = []
    curves = {}

    for angle in angles:
        print(f"Running for profile angle = {angle}°...")
        curve = run_for_angle(angle, str(config_path), facet_index)
        curves[angle] = curve
        summary.append((angle, float(np.min(curve)), float(np.max(curve))) )

    # Save summary
    df_sum = pd.DataFrame(summary, columns=['angle_deg','min_temp_K','max_temp_K'])
    df_sum.to_csv('facet37_min_max_vs_angle.csv', index=False)

    # Save diurnal curves
    tpd = len(curve)
    degrees = np.linspace(0, 360, tpd, endpoint=False)
    df_curves = pd.DataFrame({'rotation_deg': degrees})
    for angle in angles:
        df_curves[f'{angle}deg'] = curves[angle]
    df_curves.to_csv('facet37_diurnal_curves.csv', index=False)

    # Plot min/max vs angle
    plt.figure()
    plt.plot(df_sum['angle_deg'], df_sum['min_temp_K'], marker='o', label='Min')
    plt.plot(df_sum['angle_deg'], df_sum['max_temp_K'], marker='o', label='Max')
    plt.xlabel('Kernel profile angle (°)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Facet {facet_index} Min/Max Temp vs Profile Angle')
    plt.legend()
    plt.grid(True)
    plt.savefig('facet37_min_max_vs_angle.png')

    # Plot diurnal curves
    plt.figure()
    for angle in angles:
        plt.plot(df_curves['rotation_deg'], df_curves[f'{angle}deg'], label=f'{angle}°')
    plt.xlabel('Rotation angle (°)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Facet {facet_index} Diurnal Curves vs Profile Angle')
    plt.legend()
    plt.grid(True)
    plt.savefig('facet37_diurnal_curves.png')

if __name__ == '__main__':
    main() 