#!/usr/bin/env python3
"""
Script to visualize the fracture heating temperature distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from stl import mesh as stl_mesh_module
import sys
import os
import pandas as pd

# Add src to path
sys.path.append('src')

from src.model.fracture_heating import FractureHeating
from src.utilities.config import Config

def visualize_fracture_heating(config_path, output_dir="plots"):
    """Visualize the fracture heating temperature distribution"""
    
    # Load configuration
    config = Config(config_path)
    
    # Load shape model
    shape_mesh = stl_mesh_module.Mesh.from_file(config.path_to_shape_model_file)
    
    # Calculate facet positions (centroids)
    facet_positions = []
    for i in range(len(shape_mesh.vectors)):
        vertices = shape_mesh.vectors[i]
        centroid = np.mean(vertices, axis=0)
        facet_positions.append(centroid)
    facet_positions = np.array(facet_positions)
    
    # Initialize fracture heating
    fracture_heating = FractureHeating(config)
    
    print(f"Fracture heating configuration:")
    print(f"  Enabled: {fracture_heating.enabled}")
    print(f"  Direction: {fracture_heating.fracture_direction}")
    print(f"  Position: {fracture_heating.fracture_position} m")
    print(f"  Peak temp: {fracture_heating.peak_temperature} K")
    print(f"  Background temp: {fracture_heating.background_temperature} K")
    print(f"  Characteristic distance: {fracture_heating.characteristic_distance} m")
    print(f"  Profile: {fracture_heating.temperature_profile}")
    print(f"  Coordinate scale: {fracture_heating.coordinate_scale}")
    print(f"  Rotation: {fracture_heating.rotation_angle_degrees}°")
    print(f"  X offset: {fracture_heating.fracture_offset_x} m")
    print(f"  Y offset: {fracture_heating.fracture_offset_y} m")
    
    if not fracture_heating.enabled:
        print("Fracture heating is not enabled in the configuration!")
        return
    
    # Calculate distances from fracture
    distances = fracture_heating.calculate_distance_from_fracture(facet_positions)
    
    # Calculate temperatures
    temperatures = np.array([fracture_heating.calculate_base_temperature(d) for d in distances])
    
    # Also get the transformed coordinates for plotting
    scaled_positions = facet_positions * fracture_heating.coordinate_scale
    
    # Apply lateral offsets
    offset_positions = scaled_positions.copy()
    offset_positions[:, 0] -= fracture_heating.fracture_offset_x
    offset_positions[:, 1] -= fracture_heating.fracture_offset_y
    
    # Apply rotation
    if fracture_heating.rotation_angle_degrees != 0.0:
        angle_rad = np.radians(fracture_heating.rotation_angle_degrees)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        x_rotated = offset_positions[:, 0] * cos_angle + offset_positions[:, 1] * sin_angle
        y_rotated = -offset_positions[:, 0] * sin_angle + offset_positions[:, 1] * cos_angle
        z_rotated = offset_positions[:, 2]
        
        transformed_positions = np.column_stack((x_rotated, y_rotated, z_rotated))
    else:
        transformed_positions = offset_positions
    
    # Print statistics
    print(f"\nShape model statistics (original coordinates):")
    print(f"  Number of facets: {len(facet_positions)}")
    print(f"  X range: {np.min(facet_positions[:, 0]):.3f} to {np.max(facet_positions[:, 0]):.3f} km")
    print(f"  Y range: {np.min(facet_positions[:, 1]):.3f} to {np.max(facet_positions[:, 1]):.3f} km")
    print(f"  Z range: {np.min(facet_positions[:, 2]):.3f} to {np.max(facet_positions[:, 2]):.3f} km")
    
    print(f"\nTransformed coordinates (scaled and rotated):")
    print(f"  X range: {np.min(transformed_positions[:, 0]):.1f} to {np.max(transformed_positions[:, 0]):.1f} m")
    print(f"  Y range: {np.min(transformed_positions[:, 1]):.1f} to {np.max(transformed_positions[:, 1]):.1f} m")
    print(f"  Z range: {np.min(transformed_positions[:, 2]):.1f} to {np.max(transformed_positions[:, 2]):.1f} m")
    
    print(f"\nDistance statistics:")
    print(f"  Distance range: {np.min(distances):.1f} to {np.max(distances):.1f} m")
    print(f"  Mean distance: {np.mean(distances):.1f} m")
    
    print(f"\nTemperature statistics:")
    print(f"  Temperature range: {np.min(temperatures):.1f} to {np.max(temperatures):.1f} K")
    print(f"  Mean temperature: {np.mean(temperatures):.1f} K")
    
    # Create plots
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Distance vs Temperature
    sorted_indices = np.argsort(distances)
    axes[0, 0].scatter(distances[sorted_indices], temperatures[sorted_indices], alpha=0.6, s=1)
    
    # Overlay theoretical curve
    dist_range = np.linspace(0, np.max(distances), 100)
    theoretical_temps = [fracture_heating.calculate_base_temperature(d) for d in dist_range]
    axes[0, 0].plot(dist_range, theoretical_temps, 'r-', linewidth=2, label='Theoretical')
    
    axes[0, 0].set_xlabel('Distance from fracture (m)')
    axes[0, 0].set_ylabel('Temperature (K)')
    axes[0, 0].set_title('Temperature vs Distance from Fracture')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spatial distribution (transformed coordinates, colored by temperature)
    scatter = axes[0, 1].scatter(transformed_positions[:, 0], transformed_positions[:, 1], 
                                c=temperatures, cmap='hot', s=1)
    axes[0, 1].axvline(x=fracture_heating.fracture_position, color='blue', 
                      linestyle='--', linewidth=2, label=f'Fracture (x={fracture_heating.fracture_position}m)')
    axes[0, 1].set_xlabel('X position (m, transformed)')
    axes[0, 1].set_ylabel('Y position (m, transformed)')
    axes[0, 1].set_title(f'Temperature Distribution (rotated {fracture_heating.rotation_angle_degrees}°)')
    axes[0, 1].legend()
    plt.colorbar(scatter, ax=axes[0, 1], label='Temperature (K)')
    
    # Plot 3: Distance distribution histogram
    axes[1, 0].hist(distances, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=fracture_heating.characteristic_distance, color='red', 
                      linestyle='--', linewidth=2, label=f'Char. distance ({fracture_heating.characteristic_distance}m)')
    axes[1, 0].set_xlabel('Distance from fracture (m)')
    axes[1, 0].set_ylabel('Number of facets')
    axes[1, 0].set_title('Distribution of Distances from Fracture')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Temperature distribution histogram
    axes[1, 1].hist(temperatures, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=fracture_heating.peak_temperature, color='red', 
                      linestyle='--', linewidth=2, label=f'Peak temp ({fracture_heating.peak_temperature}K)')
    axes[1, 1].axvline(x=fracture_heating.background_temperature, color='blue', 
                      linestyle='--', linewidth=2, label=f'Background temp ({fracture_heating.background_temperature}K)')
    axes[1, 1].set_xlabel('Temperature (K)')
    axes[1, 1].set_ylabel('Number of facets')
    axes[1, 1].set_title('Distribution of Base Temperatures')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'fracture_heating_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: {plot_path}")
    
    # Save data to CSV
    data = pd.DataFrame({
        'facet_index': range(len(facet_positions)),
        'x_original_km': facet_positions[:, 0],
        'y_original_km': facet_positions[:, 1],
        'z_original_km': facet_positions[:, 2],
        'x_transformed_m': transformed_positions[:, 0],
        'y_transformed_m': transformed_positions[:, 1],
        'z_transformed_m': transformed_positions[:, 2],
        'distance_from_fracture_m': distances,
        'base_temperature_K': temperatures
    })
    
    csv_path = os.path.join(output_dir, 'fracture_heating_data.csv')
    data.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")

if __name__ == "__main__":
    config_path = "data/config/test_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    visualize_fracture_heating(config_path)
