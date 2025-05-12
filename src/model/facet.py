# src/model/facet.py

import numpy as np
from src.model.sub_facet import SubFacet
from src.model.spherical_cap_mesh import generate_canonical_spherical_cap

class Facet:
    def __init__(self, normal, vertices, timesteps_per_day, max_days, n_layers, calculate_energy_terms):
        self.normal = np.array(normal)
        self.vertices = np.array(vertices)
        self.area = self.calculate_area(vertices)
        self.position = np.mean(vertices, axis=0)
        self.visible_facets = []
        
        # New attributes for parent facet functionality
        self.sub_facets = []  # List of SubFacet objects
        self.parent_incident_energy_packets = []  # List of (flux_amount, direction_vector, type_flag) tuples
        self.depression_total_absorbed_solar_flux = 0.0
        self.depression_total_absorbed_scattered_flux = 0.0
        self.depression_total_absorbed_thermal_flux = 0.0
        self.depression_outgoing_flux_distribution = {
            'scattered_visible': {},  # Maps directional bins to flux amounts
            'thermal': {}            # Maps directional bins to flux amounts
        }

    def set_dynamic_arrays(self, length):
        self.secondary_radiation_view_factors = np.zeros(length)    

    @staticmethod
    def calculate_area(vertices):
        # Implement area calculation based on vertices
        v0, v1, v2 = vertices
        return np.linalg.norm(np.cross(v1-v0, v2-v0)) / 2

    def generate_spherical_depression(self, config, simulation):
        if not config.apply_spherical_depression_roughness:
            return

        # Only generate the canonical mesh once and cache it as a class attribute
        if not hasattr(Facet, "_canonical_subfacet_mesh") or Facet._canonical_subfacet_mesh is None:
            Facet._canonical_subfacet_mesh = generate_canonical_spherical_cap(
                config.depression_subfacets_count,
                config.depression_profile_angle_degrees
            )
            # Optional: Store config parameters used to generate the mesh to detect if regeneration is needed
            Facet._canonical_mesh_params = (config.depression_subfacets_count, config.depression_profile_angle_degrees)
        else:
            # Optional: Check if config changed, requiring regeneration
            current_params = (config.depression_subfacets_count, config.depression_profile_angle_degrees)
            if hasattr(Facet, "_canonical_mesh_params") and Facet._canonical_mesh_params != current_params:
                Facet._canonical_subfacet_mesh = generate_canonical_spherical_cap(
                    config.depression_subfacets_count,
                    config.depression_profile_angle_degrees
                )
                Facet._canonical_mesh_params = current_params


        self.sub_facets = []
        if Facet._canonical_subfacet_mesh: # Check if mesh generation was successful
            for i, _ in enumerate(Facet._canonical_subfacet_mesh): # Iterate using index
                # SubFacet no longer takes vertices/normal directly
                subfacet = SubFacet(
                    parent_id=id(self), 
                    local_id=i # Use 'i' as the local_id and canonical_index
                )
                # subfacet.canonical_index = i # Already set by local_id in SubFacet's __init__
                self.sub_facets.append(subfacet)

    def process_intra_depression_energetics(self, config, simulation):
        """
        Process energy exchange within the depression.
        This is a placeholder - we'll implement the actual MCRT in Step 4.
        """
        if not config.apply_spherical_depression_roughness:
            # If roughness is disabled, just pass through the parent facet's energy
            return
            
        # For now, just pass through the energy to the single sub-facet
        # We'll implement proper MCRT in Step 4
        if self.sub_facets:
            self.sub_facets[0].incident_energy = sum(packet[0] for packet in self.parent_incident_energy_packets)
            self.sub_facets[0].absorbed_energy = self.sub_facets[0].incident_energy * (1 - simulation.albedo)
            self.sub_facets[0].radiated_energy = self.sub_facets[0].incident_energy * simulation.albedo
            
            # Update parent facet's absorbed energy
            self.depression_total_absorbed_solar_flux = self.sub_facets[0].absorbed_energy
            
            # Clear the incident energy packets
            self.parent_incident_energy_packets = []