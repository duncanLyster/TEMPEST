
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(os.getcwd()).resolve()
sys.path.append(str(current_dir))

from src.model.facet import Facet
from src.model.simulation import Simulation
from src.utilities.config import Config

def verify_facet_geo():
    print("--- Verifying Facet Geometry ---")
    
    # Setup Dummy Config
    class DummyConfig:
        kernel_subfacets_count = 100
        kernel_profile_angle_degrees = 90.0 # Hemisphere
        kernel_directional_bins = 36
        apply_kernel_based_roughness = True
        temp_solver = 'crank_nicolson'
        calculate_energy_terms = False
        config_data = {
            'thermal_inertia': 200,
            'solar_distance_au': 1.0, 
            'solar_luminosity': 3.828e26,
            'emissivity': 0.95,
            'albedo': 0.12,
            'rotation_period_hours': 10,
            'n_layers': 40,
            'density': 2000,
            'specific_heat_capacity': 1000,
            'ra_degrees': 0.0,
            'dec_degrees': 90.0,
            'convergence_target': 1,
            'min_days': 1
        }
        
    config = DummyConfig()
    
    # Setup Dummy Simulation
    sim = Simulation(config)
    sim.timesteps_per_day = 10
    sim.max_days = 1
    sim.n_layers = 1
    
    # Create Facet Pointing Up (Z)
    normal = np.array([0.0, 0.0, 1.0])
    vertices = [
        np.array([-0.5, -0.5, 0.0]),
        np.array([0.5, -0.5, 0.0]),
        np.array([0.0, 0.5, 0.0])
    ] # Area = 0.5 * 1 * 1 = 0.5
    
    facet = Facet(normal, vertices, 10, 1, 1, False)
    facet.generate_spherical_depression(config, sim)
    
    print(f"Facet Area: {facet.area}")
    print(f"Subfacets: {len(facet.sub_facets)}")
    
    # Test Angles
    angles = [0, 30, 60, 80, 85]
    
    for angle_deg in angles:
        rad = np.radians(angle_deg)
        # Vector from Zenith, tilted by angle
        # Direction TO Source (or Observer)
        view_vec = np.array([np.sin(rad), 0.0, np.cos(rad)]) 
        
        # Expected Projected Area for Flat Surface
        proj_flat = facet.area * np.cos(rad)
        
        # Calculate Effective Projected Area from Subfacets
        # E_vis = Sum( Area_sub * cos(theta_sub) * Visibility )
        # This is returned by _process_incident_packet
        
        # Flux=1.0. view_vec is direction TO source.
        # But wait. process_incident_packet expects 'dir_world'.
        # If it interprets it as source vector, then d_local points UP.
        
        packet = (1.0, view_vec, 'visible')
        
        # Manually extract canonical arrays
        if not hasattr(Facet, '_canonical_subfacet_mesh') or Facet._canonical_subfacet_mesh is None:
            raise ValueError("Canonical mesh not generated!")
            
        mesh = Facet._canonical_subfacet_mesh
        normals = np.array([entry['normal'] for entry in mesh])
        areas = np.array([entry['area'] for entry in mesh])
        triangles = np.array([entry['vertices'] for entry in mesh])
        centers = np.array([np.mean(entry['vertices'], axis=0) for entry in mesh])
        
        E_vis_local, _, _, _ = Facet._process_incident_packet(
            packet, facet.dome_rotation, facet.area, 
            normals, areas, 
            triangles, centers, 
            0.0, 0.0
        )
        
        proj_rough = np.sum(E_vis_local)
        
        ratio = proj_rough / proj_flat if proj_flat > 1e-9 else 1.0
        
        print(f"Angle {angle_deg:2d} | Flat: {proj_flat:.4f} | Rough: {proj_rough:.4f} | Ratio: {ratio:.4f}")
        
        if ratio < 0.95:
            print(f"  WARNING: Rough Projected Area is significantly less ({ratio:.2f}x) than Flat.")

if __name__ == "__main__":
    verify_facet_geo()
