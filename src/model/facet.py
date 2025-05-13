# src/model/facet.py

import numpy as np
from src.model.sub_facet import SubFacet
from src.model.spherical_cap_mesh import generate_canonical_spherical_cap
import math
from src.utilities.utils import rotate_vector, calculate_rotation_matrix
from src.model.insolation import calculate_shadowing
from src.model.view_factors import calculate_view_factors

class Facet:
    def __init__(self, normal, vertices, timesteps_per_day, max_days, n_layers, calculate_energy_terms):
        self.normal = np.array(normal)
        self.vertices = np.array(vertices)
        self.area = self.calculate_area(vertices)
        self.position = np.mean(vertices, axis=0)
        self.visible_facets = []
        
        # New attributes for parent facet functionality
        self.sub_facets = []  # List of SubFacet objects
        self.dome_facets = []  # List of dome directional bins (dicts with normal, area)
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

        # Build canonical dome mesh once and cache it (in facet-local space)
        if not hasattr(Facet, "_canonical_dome_mesh") or Facet._canonical_dome_mesh is None:
            Facet._canonical_dome_mesh = generate_canonical_spherical_cap(
                config.kernel_directional_bins,
                90.0
            )
            Facet._canonical_dome_params = config.kernel_directional_bins

        # Store dome facets in facet-local coordinates
        self.dome_facets = Facet._canonical_dome_mesh

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
        self.dome_rotation = R_l2w.T  # inverse of local->world

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
        # Ensure canonical view factors are computed
        if not hasattr(Facet, '_canonical_F_ss'):
            F_ss, F_sd = Facet._compute_canonical_view_factors(config)
            Facet._canonical_F_ss = F_ss
            Facet._canonical_F_sd = F_sd
        else:
            F_ss = Facet._canonical_F_ss
        # Set sub-facet visible indices and secondary radiation view factors
        self.depression_thermal_data.visible_facets = [np.arange(N, dtype=np.int64) for _ in range(N)]
        self.depression_thermal_data.secondary_radiation_view_factors = [F_ss[i].copy() for i in range(N)]

    @staticmethod
    def _compute_canonical_view_factors(config):
        """
        Compute the canonical view-factor matrices between sub-facets and dome facets.
        Uses Monte-Carlo ray tracing via calculate_view_factors.
        Returns:
            F_ss: np.ndarray shape (N, N)
            F_sd: np.ndarray shape (N, M)
        """

        sub = Facet._canonical_subfacet_mesh
        dome = Facet._canonical_dome_mesh
        N = len(sub)
        M = len(dome)
        # prepare test sets: arrays of triangle vertices and their areas
        test_vertices_ss = np.array([entry['vertices'] for entry in sub])  # shape (N,3,3)
        test_areas_ss    = np.array([entry['area']     for entry in sub])  # shape (N,)
        test_vertices_sd = np.array([entry['vertices'] for entry in dome]) # shape (M,3,3)
        test_areas_sd    = np.array([entry['area']     for entry in dome]) # shape (M,)
        # allocate view-factor matrices
        F_ss = np.zeros((N, N), dtype=np.float64)
        F_sd = np.zeros((N, M), dtype=np.float64)
        # compute subfacet-to-subfacet view factors
        for i in range(N):
            subj = sub[i]
            F_ss[i, :] = calculate_view_factors(
                subj['vertices'], subj['normal'], subj['area'],
                test_vertices_ss, test_areas_ss, config.vf_rays
            )
        # compute subfacet-to-dome view factors
        for i in range(N):
            subj = sub[i]
            F_sd[i, :] = calculate_view_factors(
                subj['vertices'], subj['normal'], subj['area'],
                test_vertices_sd, test_areas_sd, config.vf_rays
            )

        return F_ss, F_sd

    def process_intra_depression_energetics(self, config, simulation):
        """
        Process energy exchange within the depression.

        This function processes the intra-facet energy exchange within the depression.
        It calculates the radiosity of the depression and projects it onto the dome.
        The radiosity is then used to calculate the outgoing flux distribution.
        The outgoing flux distribution is then used to calculate the total absorbed flux.
        The total absorbed flux is then used to calculate the total absorbed flux.
        The total absorbed flux is then used to calculate the total absorbed flux.
        """
        if not config.apply_kernel_based_roughness:
            # If roughness is disabled, just pass through the parent facet's energy
            return
            
        # Build and solve the radiosity problem in facet-local space
        N = len(self.sub_facets)
        if N == 0:
            self.parent_incident_energy_packets = []
            return

        # Ensure canonical F matrices exist
        if not hasattr(Facet, '_canonical_F_ss') or not hasattr(Facet, '_canonical_F_sd'):
            F_ss, F_sd = Facet._compute_canonical_view_factors(config)
            Facet._canonical_F_ss = F_ss
            Facet._canonical_F_sd = F_sd
        else:
            F_ss = Facet._canonical_F_ss
            F_sd = Facet._canonical_F_sd

        # Prepare emissive power vectors
        E0_vis = np.zeros(N, dtype=np.float64) # Emissive power vector for visible scattering, this is the total incident visible energy from the parent facet
        E0_th  = np.zeros(N, dtype=np.float64) # Emissive power vector for thermal scattering, this is the total incident thermal energy from the parent facet

        # Gather geometry
        mesh = Facet._canonical_subfacet_mesh
        # Precompute sub-facet triangles and centers for shadow tests
        sub_triangles = np.array([entry['vertices'] for entry in mesh])  # shape (N,3,3)
        centers = np.mean(sub_triangles, axis=1)                         # shape (N,3)

        # Accumulate incident energy into E0 vectors, accounting for shadowing
        for flux, dir_world, type_flag in self.parent_incident_energy_packets:
            # Map direction into facet-local space
            d_local = self.dome_rotation.dot(np.array(dir_world))
            # Shadow test each subfacet against all *other* triangles to avoid self-shadowing
            for i in range(N):
                # Build list of other triangles (exclude self) for shadow test
                other_idx = np.arange(N) != i
                triangles_other = sub_triangles[other_idx]
                idxs_other = np.arange(triangles_other.shape[0])
                # Shadowing check: 1 if lit, 0 if shadowed
                lit = calculate_shadowing(
                    np.array([centers[i]]),  # subject position
                    np.array([d_local]),     # ray direction
                    triangles_other,         # other triangle vertices
                    idxs_other               # their indices
                )
                if lit == 0:
                    continue
                entry = mesh[i]
                n_i = entry['normal']  # local normal
                area_i = entry['area'] * self.area  # scale canonical area
                cos_theta = np.dot(n_i, d_local)
                if cos_theta <= 0:
                    continue
                # Compute incident energy using raw flux and subfacet orientation
                incident = flux * area_i * cos_theta
                if type_flag in ('solar', 'scattered_visible'):
                    # visible incident
                    E0_vis[i] += incident
                    self.depression_total_absorbed_solar_flux += incident * (1 - simulation.albedo)
                elif type_flag == 'thermal':
                    # thermal incident
                    E0_th[i] += incident
                    self.depression_total_absorbed_thermal_flux += incident * (1 - simulation.emissivity)

        # Convert absorbed power E0_vis (W) into absorbed flux per subfacet area (W/m^2)
        area_vector = np.array([entry['area'] * self.area for entry in mesh], dtype=np.float64)
        absorbed = (1 - simulation.albedo) * E0_vis
        # Avoid division by zero
        self._last_absorbed_solar = np.where(area_vector > 0, absorbed / area_vector, 0.0)

        # Iterative scattering within depression (visible)
        rho = simulation.albedo
        B_vis = np.zeros(N, dtype=np.float64)
        curr = E0_vis.copy()
        for _ in range(config.intra_facet_scatters):
            B_vis += curr
            curr = rho * F_ss.dot(curr)

        # Thermal emission and single-bounce self-heating (no scattering)
        eps = simulation.emissivity
        # Direct thermal emission
        B_th_direct = eps * E0_th
        # Self-heating: energy from other sub-facets (one bounce)
        B_th_self = F_ss.T.dot(B_th_direct)
        # Total thermal radiosity
        B_th = B_th_direct + B_th_self

        # Project onto dome: outgoing[M] = F_sd^T dot B
        out_vis = F_sd.T.dot(B_vis)
        out_th  = F_sd.T.dot(B_th)

        # Store directional distributions as dicts
        self.depression_outgoing_flux_distribution['scattered_visible'] = {j: out_vis[j] for j in range(out_vis.size)}
        self.depression_outgoing_flux_distribution['thermal']          = {j: out_th[j]  for j in range(out_th.size)}

        # Clear incident packets
        self.parent_incident_energy_packets = []