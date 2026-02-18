# src/model/facet.py

import numpy as np
from src.model.sub_facet import SubFacet
from src.model.spherical_cap_mesh import generate_canonical_spherical_cap
import math
from src.utilities.utils import rotate_vector, calculate_rotation_matrix
from src.model.insolation import calculate_shadowing
from src.model.view_factors import calculate_view_factors
from joblib import Parallel, delayed

class Facet:
    def __init__(self, normal, vertices, timesteps_per_day, max_days, n_layers, calculate_energy_terms):
        self.normal = np.array(normal)
        self.vertices = np.array(vertices)
        self.area = self.calculate_area(vertices)
        self.position = np.mean(vertices, axis=0)
        self.visible_facets = []
        
        # New attributes for parent facet functionality
        self.sub_facets = []  # List of SubFacet objects
        self.dome_facets = []  # DEPRECATED: Only for old radiosity approach, not used by LUT generator
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

        # NOTE: Old dome mesh generation removed - only needed for deprecated radiosity calculations
        # The LUT generator and new thermal EPF approach don't use dome meshes
        
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
        # Ensure canonical view factors are computed (only needed for radiosity calculations)
        # Skip if dome mesh wasn't generated (e.g., for LUT generator using thermal EPF approach)
        if (hasattr(Facet, '_canonical_dome_mesh') and 
            Facet._canonical_dome_mesh is not None and 
            not hasattr(Facet, '_canonical_F_ss')):
            F_ss, F_sd = Facet._compute_canonical_view_factors(config)
            Facet._canonical_F_ss = F_ss
            Facet._canonical_F_sd = F_sd
        
        # Only set up thermal view factors if we computed them (needed for radiosity-based self-heating)
        if hasattr(Facet, '_canonical_F_ss'):
            F_ss = Facet._canonical_F_ss
            # Set sub-facet visible indices and thermal view factors for self-heating
            self.depression_thermal_data.visible_facets = [np.arange(N, dtype=np.int64) for _ in range(N)]
            self.depression_thermal_data.thermal_view_factors = [F_ss[i].copy() for i in range(N)]

        # Precompute and cache canonical subfacet triangles and centers for shadow tests
        if not hasattr(Facet, '_canonical_subfacet_triangles'):
            mesh = Facet._canonical_subfacet_mesh
            Facet._canonical_subfacet_triangles = np.array([entry['vertices'] for entry in mesh])
            Facet._canonical_subfacet_centers  = np.mean(Facet._canonical_subfacet_triangles, axis=1)

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

        #########################################################
        # Monte-Carlo view-factor computation
        #########################################################

        # # parallel compute canonical view factors
        # # compute subfacet-to-subfacet view factors
        # ss_results = Parallel(n_jobs=config.n_jobs)(
        #     delayed(calculate_view_factors)(
        #         sub[i]['vertices'], sub[i]['normal'], sub[i]['area'],
        #         test_vertices_ss, test_areas_ss, config.vf_rays
        #     ) for i in range(N)
        #     )
        # for i, row in enumerate(ss_results):
        #     F_ss[i, :] = row
        # # compute subfacet-to-dome view factors
        # sd_results = Parallel(n_jobs=config.n_jobs)(
        #     delayed(calculate_view_factors)(
        #         sub[i]['vertices'], sub[i]['normal'], sub[i]['area'],
        #         test_vertices_sd, test_areas_sd, config.vf_rays
        #     ) for i in range(N)
        #     )
        # for i, row in enumerate(sd_results):
        #     F_sd[i, :] = row

        #########################################################
        # Geometric centroid-based view-factor computation
        #########################################################
        # Extract normals, centroids, and areas
        normals_ss = np.array([entry['normal'] for entry in sub], dtype=np.float64)
        centroids_ss = np.array([np.mean(entry['vertices'], axis=0) for entry in sub], dtype=np.float64)
        normals_sd = np.array([entry['normal'] for entry in dome], dtype=np.float64)
        centroids_sd = np.array([np.mean(entry['vertices'], axis=0) for entry in dome], dtype=np.float64)
        areas_ss = test_areas_ss
        areas_sd = test_areas_sd

        # Compute centroid-based view factors between subfacets
        for i in range(N):
            for j in range(N):
                vec_ij = centroids_ss[j] - centroids_ss[i]
                dist2 = np.dot(vec_ij, vec_ij)
                if dist2 <= 0:
                    continue
                r = np.sqrt(dist2)
                dir_ij = vec_ij / r
                cos_i = np.dot(normals_ss[i], dir_ij)
                cos_j = np.dot(normals_ss[j], -dir_ij)
                if cos_i <= 0 or cos_j <= 0:
                    continue
                F_ss[i, j] = cos_i * cos_j * areas_ss[j] / (np.pi * dist2)

        # Compute view factors to sky dome (F_sd) treating dome as "Sky at Infinity"
        # The dome mesh is generated as a bottom hemisphere (z < 0).
        # Each dome facet is treated as a directional bin of the sky.
        # Direction to sky bin j is -normals_sd[j] (since dome normals point out/down).
        
        # Calculate radius squared of the canonical dome from total area
        # For 90 deg cap, Area = 2 * pi * R^2. 
        # Total area of dome mesh sum(areas_sd) should be ~ 2*pi*R^2.
        # This R2 is needed to convert Area_j to Solid Angle dOmega.
        total_dome_area = areas_sd.sum()
        if total_dome_area > 0:
            R2 = total_dome_area / (2 * np.pi)
        else:
            R2 = 1.0 # Should not happen
            print(f"Warning: total_dome_area is 0")

        for i in range(N):
            for j in range(M):
                # Direction to this patch of sky
                # Original dome is bottom hemisphere, normals point out (down).
                # So direction TO sky is -normal.
                # However, we want to map this to the UPPER hemisphere.
                # So we take the vector -normals_sd[j], which points UP.
                dir_to_sky = -normals_sd[j]
                
                # Check visibility (horizon cutoff)
                # Angle between subfacet normal and sky direction
                cos_theta = np.dot(normals_ss[i], dir_to_sky)
                
                if cos_theta <= 0:
                    continue
                    
                # Solid angle of this sky patch
                # dOmega = Area_j / R^2
                if R2 > 0:
                    dOmega = areas_sd[j] / R2
                else:
                    dOmega = 0
                
                # View factor
                # F = (cos_theta * dOmega) / pi
                val = (cos_theta * dOmega) / np.pi
                
                F_sd[i, j] = val

        # Normalize view factors to ensure conservation of energy
        # F_ss is determined by local geometry and is assumed correct.
        # F_sd is scaled so that F_ss + F_sd sums to 1.0 (remainder goes to sky).
        
        # Track max correction for logging
        max_correction = 0.0
        max_correction_idx = -1
        
        for i in range(N):
            sum_ss = F_ss[i, :].sum()
            sum_sd = F_sd[i, :].sum()
            
            current_total = sum_ss + sum_sd
            remainder = 1.0 - sum_ss
            
            # Log if correction is significant (>1%)
            if abs(current_total - 1.0) > 0.01:
                correction_mag = abs(1.0 - current_total)
                if correction_mag > max_correction:
                    max_correction = correction_mag
                    max_correction_idx = i
            
            if remainder < 0:
                # Should not happen physically unless F_ss > 1 due to numerical error
                # In this case, scale F_ss down to 1.0 and set F_sd to 0
                F_ss[i, :] /= sum_ss
                F_sd[i, :] = 0.0
            elif sum_sd > 0:
                # Scale F_sd to fill the remainder
                scale_factor = remainder / sum_sd
                F_sd[i, :] *= scale_factor
            else:
                # No dome view factors calculated, but there is a remainder.
                # Distribute remainder uniformly across all dome facets
                if M > 0:
                    F_sd[i, :] = remainder / M
                    
        if max_correction > 0.01:
            print(f"Notice: View factor energy conservation required correction (Max missing: {max_correction*100:.1f}% at subfacet {max_correction_idx})")
            print(f"        F_sd was scaled to ensure sum(F_ss) + sum(F_sd) = 1.0")

        return F_ss, F_sd

    @staticmethod
    def _process_incident_packet(packet, world_to_local_rotation, aperture_area, sub_normals, sub_areas, sub_triangles, centers, albedo, emissivity):
        """Helper to process one incident energy packet for the depression in parallel."""
        import numpy as np
        flux_val, dir_world, type_flag = packet
        d_world_vec = np.array(dir_world)
        d_local = world_to_local_rotation.dot(d_world_vec)
        cos_theta_aperture = d_local[2]
        N = sub_normals.shape[0]
        E0_vis_local = np.zeros(N, dtype=np.float64)
        E0_th_local = np.zeros(N, dtype=np.float64)
        absorbed_solar = 0.0
        absorbed_thermal = 0.0

        if cos_theta_aperture <= 0 or flux_val * aperture_area * cos_theta_aperture <= 1e-9:
            return E0_vis_local, E0_th_local, absorbed_solar, absorbed_thermal

        E_total = flux_val * aperture_area * cos_theta_aperture

        # Vectorized: compute all cosines at once
        cos_theta_all = sub_normals.dot(d_local)
        front_facing = cos_theta_all > 0
        
        if not front_facing.any():
            # No subfacets face the incoming direction, distribute evenly
            e_per = E_total / N
            for i in range(N):
                if type_flag in ('solar', 'scattered_visible'):
                    E0_vis_local[i] = e_per
                    absorbed_solar += e_per * (1 - albedo)
                else:
                    E0_th_local[i] = e_per
                    absorbed_thermal += e_per * (1 - emissivity)
            return E0_vis_local, E0_th_local, absorbed_solar, absorbed_thermal
        
        # Get indices of front-facing subfacets
        candidate_indices = np.where(front_facing)[0]
        n_candidates = len(candidate_indices)
        
        # Vectorized shadow check: check all candidates at once
        d_local_repeated = np.tile(d_local.reshape(1, 3), (n_candidates, 1))
        candidate_centers = centers[candidate_indices]
        
        # For shadow check, each candidate should test against all OTHER subfacets
        # This is still complex, so we'll do a batch approach per candidate
        lit_mask = np.ones(n_candidates, dtype=bool)
        for idx_in_batch, global_idx in enumerate(candidate_indices):
            mask = np.arange(N) != global_idx
            triangles_for_shadow = sub_triangles[mask]
            indices_for_shadow = np.arange(triangles_for_shadow.shape[0])
            is_lit = calculate_shadowing(
                centers[global_idx:global_idx+1],
                d_local.reshape(1, 3),
                triangles_for_shadow,
                indices_for_shadow
            )
            lit_mask[idx_in_batch] = (is_lit == 1)
        
        lit_candidates = candidate_indices[lit_mask]
        
        if len(lit_candidates) > 0:
            # Compute contributions for lit subfacets
            contribs = sub_areas[lit_candidates] * cos_theta_all[lit_candidates]
            total_contrib = contribs.sum()
            
            if total_contrib > 1e-9:
                energies = E_total * (contribs / total_contrib)
                for idx, e in zip(lit_candidates, energies):
                    if type_flag in ('solar', 'scattered_visible', 'visible'):
                        E0_vis_local[idx] = e
                        if type_flag != 'visible':
                            absorbed_solar += e * (1 - albedo)
                    else:
                        E0_th_local[idx] = e
                        absorbed_thermal += e * (1 - emissivity)
            else:
                # Distribute evenly among lit subfacets
                e_per = E_total / len(lit_candidates)
                for idx in lit_candidates:
                    if type_flag in ('solar', 'scattered_visible', 'visible'):
                        E0_vis_local[idx] = e_per
                        if type_flag != 'visible':
                            absorbed_solar += e_per * (1 - albedo)
                    else:
                        E0_th_local[idx] = e_per
                        absorbed_thermal += e_per * (1 - emissivity)
        else:
            # No lit subfacets, distribute evenly to all
            e_per = E_total / N
            for i in range(N):
                if type_flag in ('solar', 'scattered_visible', 'visible'):
                    E0_vis_local[i] = e_per
                    if type_flag != 'visible':
                        absorbed_solar += e_per * (1 - albedo)
                else:
                    E0_th_local[i] = e_per
                    absorbed_thermal += e_per * (1 - emissivity)

        return E0_vis_local, E0_th_local, absorbed_solar, absorbed_thermal

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

        # Prepare emissive power vectors and absorbed flux accumulators
        E0_vis = np.zeros(N, dtype=np.float64)
        E0_th  = np.zeros(N, dtype=np.float64)
        self.depression_total_absorbed_solar_flux = 0.0
        self.depression_total_absorbed_thermal_flux = 0.0

        # Precompute subfacet geometry arrays once (cache at class level)
        if not hasattr(Facet, '_canonical_normals'):
            mesh = Facet._canonical_subfacet_mesh
            Facet._canonical_normals = np.array([entry['normal'] for entry in mesh], dtype=np.float64)
            Facet._canonical_areas = np.array([entry['area'] for entry in mesh], dtype=np.float64)
        
        normals = Facet._canonical_normals
        areas = Facet._canonical_areas
        triangles = Facet._canonical_subfacet_triangles
        centers = Facet._canonical_subfacet_centers

        # Process each incident packet sequentially (no Parallel overhead)
        for packet in self.parent_incident_energy_packets:
            E0v_loc, E0t_loc, abs_s, abs_t = Facet._process_incident_packet(
                packet,
                self.world_to_local_rotation,
                self.area,
                normals,
                areas,
                triangles,
                centers,
                simulation.albedo,
                simulation.emissivity
            )
            E0_vis += E0v_loc
            E0_th  += E0t_loc
            self.depression_total_absorbed_solar_flux += abs_s
            self.depression_total_absorbed_thermal_flux += abs_t
        # End sequential accumulation

        # Convert absorbed power E0_vis (W) into absorbed flux per subfacet area (W/m^2)
        # Note: E0_vis contains total energy on each sub-facet.
        # The canonical mesh is generated such that its projected aperture area is 1.0.
        # So we scale by self.area (physical aperture area) to get physical sub-facet areas.
        # We DO NOT normalize by the sum of surface areas, because surface area > aperture area for roughness.
        area_vector = areas * self.area
        
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

        # Store directional distributions as dicts (for backwards compatibility)
        self.depression_outgoing_flux_distribution['scattered_visible'] = {j: out_vis[j] for j in range(out_vis.size)}
        self.depression_outgoing_flux_distribution['thermal']          = {j: out_th[j]  for j in range(out_th.size)}
        # Also store as numpy arrays for fast access
        self.depression_outgoing_flux_array_vis = out_vis.copy()
        self.depression_outgoing_flux_array_th  = out_th.copy()

        # Clear incident packets
        self.parent_incident_energy_packets = []