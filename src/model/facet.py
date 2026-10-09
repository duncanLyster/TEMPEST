# src/model/facet.py

import numpy as np
from src.model.sub_facet import SubFacet
from src.model.spherical_cap_mesh import generate_canonical_spherical_cap
import math
from src.utilities.utils import rotate_vector, calculate_rotation_matrix

class Facet:
    def __init__(self, normal, vertices, timesteps_per_day, max_days, n_layers, calculate_energy_terms):
        self.normal = np.array(normal)
        self.vertices = np.array(vertices)
        self.area = self.calculate_area(vertices)
        self.position = np.mean(vertices, axis=0)
        self.visible_facets = []
        
        self.sub_facets = []  # SubFacet objects of the roughness kernel

    def set_dynamic_arrays(self, length):
        self.secondary_radiation_view_factors = np.zeros(length)    

    @staticmethod
    def calculate_area(vertices):
        # Implement area calculation based on vertices
        v0, v1, v2 = vertices
        return np.linalg.norm(np.cross(v1-v0, v2-v0)) / 2

    def generate_spherical_depression(self, config, simulation):
        if not config.apply_kernel_based_roughness:
            return

        # Only generate the canonical mesh once and cache it as a class attribute
        if not hasattr(Facet, "_canonical_subfacet_mesh") or Facet._canonical_subfacet_mesh is None:
            # Generate canonical sub-facet mesh for this kernel
            Facet._canonical_subfacet_mesh = generate_canonical_spherical_cap(
                config.kernel_subfacets_count,
                config.kernel_profile_angle_degrees
            )
            Facet._canonical_mesh_params = (config.kernel_subfacets_count, config.kernel_profile_angle_degrees)
        else:
            # Regenerate if kernel params changed
            current_params = (config.kernel_subfacets_count, config.kernel_profile_angle_degrees)
            if hasattr(Facet, "_canonical_mesh_params") and Facet._canonical_mesh_params != current_params:
                Facet._canonical_subfacet_mesh = generate_canonical_spherical_cap(
                    config.kernel_subfacets_count,
                    config.kernel_profile_angle_degrees
                )
                Facet._canonical_mesh_params = current_params

        self.sub_facets = []
        if Facet._canonical_subfacet_mesh:
            for i, _entry in enumerate(Facet._canonical_subfacet_mesh):
                subfacet = SubFacet(parent_id=id(self), local_id=i)
                self.sub_facets.append(subfacet)

        # Compute world->local rotation matrix for mapping directions into facet space
        up = np.array([0.0, 0.0, 1.0])
        n = self.normal / np.linalg.norm(self.normal)
        axis = np.cross(up, n)
        if np.linalg.norm(axis) < 1e-8:
            # Degenerate: facet normal aligned or opposite to up
            if np.dot(up, n) > 0:
                R_l2w = np.eye(3)
            else:
                R_l2w = calculate_rotation_matrix(np.array([1.0, 0.0, 0.0]), math.pi)
        else:
            axis_norm = axis / np.linalg.norm(axis)
            angle = math.acos(np.clip(np.dot(up, n), -1.0, 1.0))
            R_l2w = calculate_rotation_matrix(axis_norm, angle)
        # Store world->local conversion
        self.world_to_local_rotation = R_l2w.T  # inverse of local->world

        # Skip thermal data initialization if simulation lacks required attributes (e.g., in tests)
        if not hasattr(simulation, 'timesteps_per_day'):
            return
        # Initialize per-depression ThermalData for sub-facet conduction and self-heating
        from src.model.simulation import ThermalData
        N = len(self.sub_facets)
        self.depression_thermal_data = ThermalData(
            n_facets=N,
            timesteps_per_day=simulation.timesteps_per_day,
            n_layers=simulation.n_layers,
            max_days=simulation.max_days,
            calculate_energy_terms=config.calculate_energy_terms
        )
        # Precompute and cache canonical subfacet triangles and centers for shadow tests
        if not hasattr(Facet, '_canonical_subfacet_triangles'):
            mesh = Facet._canonical_subfacet_mesh
            Facet._canonical_subfacet_triangles = np.array([entry['vertices'] for entry in mesh])
            Facet._canonical_subfacet_centers  = np.mean(Facet._canonical_subfacet_triangles, axis=1)
