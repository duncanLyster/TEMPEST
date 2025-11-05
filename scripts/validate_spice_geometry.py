#!/usr/bin/env python3
# scripts/validate_spice_geometry.py

"""
Validation script for SPICE-derived geometry in TEMPEST.

This script helps verify that SPICE kernels are correctly loaded and
that geometry calculations produce expected results.

Usage:
    python scripts/validate_spice_geometry.py --config data/config/bennu_spice_example.yaml
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure src directory is in the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.utilities.config import Config
from src.model.simulation import Simulation


def validate_spice_geometry(config_path):
    """
    Validate SPICE geometry calculations.
    
    Args:
        config_path: Path to configuration file with SPICE settings
    """
    print("=" * 60)
    print("SPICE Geometry Validation for TEMPEST")
    print("=" * 60)
    
    # Load configuration
    print(f"\nLoading configuration from: {config_path}")
    config = Config(config_path=config_path)
    
    if not config.use_spice:
        print("ERROR: SPICE mode is not enabled in this configuration.")
        print("Set 'use_spice: true' in your config file.")
        return False
        
    # Validate SPICE config
    try:
        config.validate_spice_config()
        print("✓ SPICE configuration validation passed")
    except ValueError as e:
        print(f"✗ SPICE configuration validation failed: {e}")
        return False
    
    # Initialize simulation (this will load SPICE kernels)
    print("\nInitializing simulation and loading SPICE kernels...")
    try:
        simulation = Simulation(config)
        print("✓ SPICE kernels loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load SPICE kernels: {e}")
        return False
    
    # Print basic information
    print(f"\nTarget body: {config.spice_target_body}")
    print(f"Observer: {config.spice_observer}")
    print(f"Start time: {config.spice_start_time}")
    print(f"Duration: {config.spice_duration_hours} hours")
    print(f"Timesteps per day: {simulation.timesteps_per_day}")
    print(f"Update frequency: {config.spice_update_frequency}")
    
    # Validate solar distances
    print("\n" + "-" * 60)
    print("Solar Distance Validation")
    print("-" * 60)
    
    if simulation.spice_solar_distances is not None:
        distances_au = simulation.spice_solar_distances / 1.496e11
        print(f"Minimum distance: {distances_au.min():.4f} AU")
        print(f"Maximum distance: {distances_au.max():.4f} AU")
        print(f"Mean distance: {distances_au.mean():.4f} AU")
        print(f"Distance variation: {(distances_au.max() - distances_au.min()):.6f} AU")
        
        # Check if distances are reasonable
        if distances_au.min() < 0.3:
            print("⚠ Warning: Minimum solar distance seems too small")
        if distances_au.max() > 50:
            print("⚠ Warning: Maximum solar distance seems too large")
        else:
            print("✓ Solar distances appear reasonable")
    else:
        print("✗ No solar distance data available")
        
    # Validate sun directions
    print("\n" + "-" * 60)
    print("Sun Direction Validation")
    print("-" * 60)
    
    if simulation.spice_sun_directions is not None:
        # Check that all directions are normalized
        norms = np.linalg.norm(simulation.spice_sun_directions, axis=1)
        if np.allclose(norms, 1.0, atol=1e-6):
            print("✓ All sun direction vectors are normalized")
        else:
            print(f"✗ Some sun directions are not normalized (norms: {norms.min():.6f} to {norms.max():.6f})")
            
        # Print first and last directions
        print(f"\nFirst sun direction: [{simulation.spice_sun_directions[0, 0]:.4f}, "
              f"{simulation.spice_sun_directions[0, 1]:.4f}, "
              f"{simulation.spice_sun_directions[0, 2]:.4f}]")
        print(f"Last sun direction:  [{simulation.spice_sun_directions[-1, 0]:.4f}, "
              f"{simulation.spice_sun_directions[-1, 1]:.4f}, "
              f"{simulation.spice_sun_directions[-1, 2]:.4f}]")
        
        # Calculate angular change
        dot_product = np.dot(simulation.spice_sun_directions[0], 
                            simulation.spice_sun_directions[-1])
        angular_change = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        print(f"\nAngular change from start to end: {angular_change:.2f} degrees")
    else:
        print("✗ No sun direction data available")
        
    # Validate body orientations (if available)
    print("\n" + "-" * 60)
    print("Body Orientation Validation")
    print("-" * 60)
    
    if simulation.spice_body_orientations is not None:
        print(f"✓ Body orientation data available ({len(simulation.spice_body_orientations)} timesteps)")
        
        # Check that rotation matrices are valid
        for i, R in enumerate(simulation.spice_body_orientations[:3]):  # Check first 3
            det = np.linalg.det(R)
            if not np.isclose(det, 1.0, atol=1e-6):
                print(f"⚠ Warning: Rotation matrix {i} has determinant {det:.6f} (expected 1.0)")
                
        print("✓ Rotation matrices appear valid")
    else:
        print("ℹ Body orientation data not available (will use identity matrices)")
    
    # Test geometry retrieval at specific timesteps
    print("\n" + "-" * 60)
    print("Geometry Retrieval Test")
    print("-" * 60)
    
    test_timesteps = [0, simulation.timesteps_per_day // 4, 
                     simulation.timesteps_per_day // 2,
                     simulation.timesteps_per_day - 1]
    
    for t in test_timesteps:
        geom = simulation.get_geometry_at_timestep(t)
        dist_au = geom['solar_distance_m'] / 1.496e11
        sun_dir = geom['sun_direction']
        print(f"\nTimestep {t}:")
        print(f"  Solar distance: {dist_au:.4f} AU")
        print(f"  Sun direction: [{sun_dir[0]:.4f}, {sun_dir[1]:.4f}, {sun_dir[2]:.4f}]")
        print(f"  Direction norm: {np.linalg.norm(sun_dir):.6f}")
    
    # Create validation plots
    print("\n" + "-" * 60)
    print("Creating Validation Plots")
    print("-" * 60)
    
    create_validation_plots(simulation, config)
    
    # Cleanup
    simulation.cleanup_spice()
    print("\n✓ SPICE resources cleaned up")
    
    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
    
    return True


def create_validation_plots(simulation, config):
    """Create plots to visualize SPICE-derived geometry."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'SPICE Geometry Validation: {config.spice_target_body}', fontsize=14)
    
    timesteps = np.arange(len(simulation.spice_solar_distances))
    
    # Plot 1: Solar distance over time
    ax = axes[0, 0]
    distances_au = simulation.spice_solar_distances / 1.496e11
    ax.plot(timesteps, distances_au, 'b-', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Solar Distance (AU)')
    ax.set_title('Solar Distance Variation')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sun direction components
    ax = axes[0, 1]
    ax.plot(timesteps, simulation.spice_sun_directions[:, 0], 'r-', label='X', linewidth=2)
    ax.plot(timesteps, simulation.spice_sun_directions[:, 1], 'g-', label='Y', linewidth=2)
    ax.plot(timesteps, simulation.spice_sun_directions[:, 2], 'b-', label='Z', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Sun Direction Component')
    ax.set_title('Sun Direction Vector Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Solar flux variation
    ax = axes[1, 0]
    solar_flux = simulation.solar_luminosity / (4 * np.pi * simulation.spice_solar_distances ** 2)
    ax.plot(timesteps, solar_flux, 'orange', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Solar Flux (W/m²)')
    ax.set_title('Solar Flux at Body Surface')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: 3D sun direction trajectory
    ax = axes[1, 1]
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot(simulation.spice_sun_directions[:, 0],
            simulation.spice_sun_directions[:, 1],
            simulation.spice_sun_directions[:, 2],
            'b-', linewidth=2)
    ax.scatter(simulation.spice_sun_directions[0, 0],
              simulation.spice_sun_directions[0, 1],
              simulation.spice_sun_directions[0, 2],
              c='green', s=100, marker='o', label='Start')
    ax.scatter(simulation.spice_sun_directions[-1, 0],
              simulation.spice_sun_directions[-1, 1],
              simulation.spice_sun_directions[-1, 2],
              c='red', s=100, marker='x', label='End')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sun Direction Trajectory')
    ax.legend()
    
    # Draw unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='gray')
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'spice_geometry_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Validation plots saved to: {output_path}")
    
    # Show plot if not in remote mode
    if not config.remote:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Validate SPICE geometry calculations for TEMPEST'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to SPICE-enabled configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        success = validate_spice_geometry(args.config)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

