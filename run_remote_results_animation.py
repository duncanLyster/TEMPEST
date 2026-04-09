#!/usr/bin/env python3
"""
Run animation for remote results directly without needing pre-processed animation parameters.
Usage: python run_remote_results_animation.py <path_to_remote_results>
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utilities.plotting.animate_model import animate_model


def run_remote_animation(results_dir):
    """
    Load remote results and run animation.
    
    Args:
        results_dir (str): Path to remote results directory containing config.yaml and temperatures.npy
    """
    results_dir = Path(results_dir)
    
    # Load config
    config_file = results_dir / 'config.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load temperatures
    temp_file = results_dir / 'temperatures.npy'
    if not temp_file.exists():
        raise FileNotFoundError(f"Temperature file not found: {temp_file}")
    
    temperatures = np.load(temp_file)
    print(f"Loaded temperatures: shape {temperatures.shape}")
    
    # Use full temperature time series for animation
    plotted_variable_array = temperatures
    print(f"Using full temperature time series: shape {plotted_variable_array.shape}")
    
    # Get shape model file
    shape_model_file = config.get('shape_model_file')
    shape_model_dir = config.get('shape_model_directory')
    if not shape_model_file or not shape_model_dir:
        raise ValueError("shape_model_file or shape_model_directory not found in config")
    
    # Construct full path to shape model
    shape_model_path = os.path.join(os.path.dirname(__file__), shape_model_dir, shape_model_file)
    if not os.path.exists(shape_model_path):
        raise FileNotFoundError(f"Shape model not found: {shape_model_path}")
    
    print(f"Using shape model: {shape_model_path}")
    
    # Extract animation parameters from config
    timesteps_per_day = config.get('timesteps_per_day', 240)
    solar_distance_au = config.get('solar_distance_au', 1.0)
    rotation_period_hr = config.get('rotation_period_hours', 24.0)
    emissivity = config.get('emissivity', 0.9)
    
    # Rotation axis (spin pole)
    ra = np.radians(config.get('ra_degrees', 0))
    dec = np.radians(config.get('dec_degrees', 0))
    rotation_axis = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])
    
    # Sunlight direction
    sunlight_direction = np.array(config.get('sunlight_direction', [1, 0, 0]))
    sunlight_direction = sunlight_direction / np.linalg.norm(sunlight_direction)
    
    print(f"Rotation axis: {rotation_axis}")
    print(f"Sunlight direction: {sunlight_direction}")
    print(f"Emissivity: {emissivity}")
    
    # Animation parameters
    plot_title = f"Temperature Distribution - {results_dir.name}"
    axis_label = "Temperature (K)"
    animation_frames = 200
    save_animation = False
    background_colour = 'black'
    colour_map = 'coolwarm'
    
    # Get optional parameters
    apply_kernel_roughness = config.get('apply_kernel_based_roughness', False)
    dome_radius_factor = config.get('kernel_dome_radius_factor', 1.0)
    
    print("\n" + "="*60)
    print("Running animation...")
    print("="*60)
    
    # Call animate_model
    animate_model(
        shape_model_path,
        plotted_variable_array,
        rotation_axis,
        sunlight_direction,
        timesteps_per_day,
        solar_distance_au,
        rotation_period_hr,
        emissivity,
        plot_title,
        axis_label,
        animation_frames,
        save_animation,
        f"{results_dir.name}_animation.gif",
        background_colour,
        dome_radius_factor=dome_radius_factor,
        colour_map=colour_map,
        apply_kernel_based_roughness=apply_kernel_roughness,
        animation_debug_mode=True,
        output_dir=str(results_dir),
        center_mesh_at_origin=False,
        rotate_mesh=True
    )
    
    print("Animation complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_remote_results_animation.py <path_to_remote_results>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    run_remote_animation(results_dir)
