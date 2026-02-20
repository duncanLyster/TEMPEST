"""
Generate High-Fidelity SPECTRAL Roughness Lookup Tables (LUTs) for TEMPEST.

This script simulates the thermal behavior of a rough surface (modeled as a spherical crater)
across a wide range of conditions to generate a radiometric correction table.

Changes in V2:
- SPECTRAL OUTPUT: Calculates radiance ratios using Planck function instead of T^4.
- High-Resolution internal simulation (720 steps) for stability.
- Uses Dimensionless Thermal Parameter (Theta) for grid.

Dimensions:
1. Thermal Parameter (Theta): Capture thermal memory.
2. Opening Angle: Capture crater shape (90 = Hemisphere).
3. Wavelength (Lambda): Capture spectral non-linearity.
4. Latitude: Capture diurnal history (Equatorial vs Polar).
5. Sun Phase (Time): The rotation angle of the body (0-360).
6. Emission Angle (View): Observer angle (0-90).
7. Azimuth Angle (View): Relative angle between Sun and Observer (0-180).

Output:
A 7D tensor containing the Radiance Factor:
    R(lambda) = Radiance_Rough(lambda) / Radiance_Smooth(lambda)
"""

import os
import sys
import time
import traceback
import numpy as np
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path

# Ensure root directory is in the Python path to allow 'from src...' imports
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.model.facet import Facet
from src.model.simulation import Simulation, ThermalData
from src.model.solvers import TemperatureSolverFactory
from src.utilities.config import Config
from src.utilities.locations import Locations
from src.utilities.utils import calculate_rotation_matrix, rays_triangles_intersection

# Import the core modules for the Standard Simulation
from src.model.insolation import calculate_insolation
from src.model.view_factors import calculate_all_view_factors, calculate_thermal_view_factors

# ============================================================================
# CONFIGURATION: LUT Generation Parameters
# ============================================================================
# Edit this section to customize your LUT generation.
# All parameters are defined here for easy modification.

# --- OUTPUT ---
OUTPUT_FILE = "roughness_lut_spectral_v1.h5"  # Name of the generated HDF5 file

# --- LUT GRID DIMENSIONS ---
# These define the dimensionality and resolution of your lookup table.
# Higher resolution = more accurate, but slower generation and larger files.

# Thermal Parameter (Theta = TI * sqrt(omega) / (emissivity * sigma * Tss^3))
# Controls how quickly surface responds to solar heating
# At P=10h: Theta=0.04TI=10 (dust), Theta=0.4TI=100 (regolith), Theta=8TI=2000 (rock)
THETA_VALUES = np.array([0.0316]) #np.logspace(-1.5, 1, 10)  # 10 points from ~0.03 to 10 (TI: 10-2500)
# For testing, use: THETA_VALUES = [0.5]  # TI≈100 at P=10h

# Crater Opening Angles (degrees)
# 90 = hemisphere, smaller = bowl-shaped
OPENING_ANGLES = [90.0]

# Wavelengths for spectral radiance calculations (microns)
# Separate bands allow capturing non-linear thermal emission
WAVELENGTHS_MICRONS = [5.0, 8.0, 15.0, 50.0, 100.0]

# Latitudes (degrees) - Surface position on rotating body
# Captures diurnal variation from equator (0°) to pole (90°)
LATITUDE_VALUES = np.arange(0.0, 92.5, 2.5)  # 37 points, 2.5° spacing
# For testing, use: LATITUDE_VALUES = [0.0, 30.0, 60.0, 85.0]

# Emission Angles (degrees) - Observer viewing angle
# 0° = looking straight down, 90° = grazing view
# MUST have ≥10 points to capture limbward darkening
EMISSION_ANGLES = np.linspace(0, 89, 20)

# Azimuth Angles (degrees) - Angle between sun and observer
# 0° = opposition (sun behind observer), 180° = full phase
# MUST have ≥10 points to capture sharp opposition effect at 0-30°
AZIMUTH_ANGLES = np.linspace(0, 180, 20)

# --- TEMPORAL RESOLUTION ---
LUT_TIMESTEPS = 90   # Output resolution: 360°/180 = 2° per step
SIM_TIMESTEPS = 360   # Internal simulation: 4x finer for stability

# --- CRATER GEOMETRY ---
# Resolution of the spherical crater mesh
CRATER_SUBFACETS = 100  # Facet count in crater (balance: 200-500 recommended)
                        # Lower = faster but coarser, Higher = slower but smoother
                        # Note: Actual count may differ slightly due to geodesic tessellation

# View factor ray tracing resolution
VIEW_FACTOR_RAYS = 2000  # Rays per facet for view factor calculation
                         # Higher = better energy conservation (aim for <5% error)
                         # Minimum 1000, recommended 2000-5000

# --- PHYSICAL PARAMETERS (Standard Reference Conditions) ---
# These define the "canonical" surface for the LUT
# Actual simulations scale these using the thermal parameter (Theta)

REFERENCE_ALBEDO = 0.12        # Bond albedo (typical for asteroids)
REFERENCE_EMISSIVITY = 0.95    # Infrared emissivity
ROTATION_PERIOD_HOURS = 10.0   # Body rotation period
SOLAR_DISTANCE_AU = 1.0        # Distance from sun (1 AU)
SOLAR_LUMINOSITY = 3.828e26    # Solar luminosity (W)

# Subsurface thermal model
N_LAYERS = 40           # Number of subsurface layers
MAX_DAYS = 50           # Maximum days to reach thermal equilibrium
MIN_DAYS = 3            # Minimum days before checking convergence
CONVERGENCE_TARGET = 1  # Convergence threshold (K for mean temperature change)

# ============================================================================
# End of User Configuration
# ============================================================================

def planck_function(wavelength_um, temp_k):
    """
    Calculate spectral radiance B(lambda, T).
    Result unit: W / (m^2 sr um)
    
    Constants:
    c1 = 2 * h * c^2 = 1.191042e8 W um^4 / (m^2 sr)
    c2 = h * c / k_B = 1.4387752e4 um K
    """
    c1 = 1.191042e8  
    c2 = 1.4387752e4 
    
    # Avoid overflow/div-by-zero for T=0 or very small
    t_safe = np.maximum(temp_k, 1e-5)
    
    # Calculate
    val = c1 / (wavelength_um**5 * (np.exp(c2 / (wavelength_um * t_safe)) - 1))
    return val

class ReferenceConfig(Config):
    """
    A minimal Config subclass that does not require a file.
    Used to generate the generic LUT under standard reference conditions.
    All values now pulled from configuration constants above.
    """
    def __init__(self):
        # Initialize Locations (required by base class, but we won't use paths)
        self.locations = Locations()
        
        # Build config_data from configuration constants
        self.config_data = {
            # Physical parameters from config
            'solar_distance_au': SOLAR_DISTANCE_AU,
            'solar_luminosity': SOLAR_LUMINOSITY,
            'albedo': REFERENCE_ALBEDO,
            'emissivity': REFERENCE_EMISSIVITY,
            'beaming_factor': 1.0,
            'subsurface_heating_flux': 0.0,
            'sunlight_direction': [1.0, 0.0, 0.0],  # Sun along +X axis
            
            # Solver Settings from config
            'temp_solver': 'tempest_implicit',
            'convergence_method': 'mean',
            'convergence_target': CONVERGENCE_TARGET,
            'max_days': MAX_DAYS,
            'min_days': MIN_DAYS,
            'n_layers': N_LAYERS,
            'include_shadowing': True,
            'n_scatters': 0, 
            'include_self_heating': True,
            'vf_rays': 10000,  # Not used (overridden in simulate_crater_diurnal_cycle)
            'intra_facet_scatters': 2,
            'silent_mode': True,  # Suppress verbose solver output
            
            # Kernel/Roughness specific from config
            'apply_kernel_based_roughness': True,
            'roughness_kernel': 'spherical_section',
            'kernel_subfacets_count': CRATER_SUBFACETS,
            'kernel_profile_angle_degrees': 90.0,  # Will be set per-case
            
            # Disable Plotting/Saving
            'plot_temp_curve': [],
            'save_temp_curve_data': False,
            'save_visible_phase_curve_data': False,
            'save_thermal_phase_curve_data': False,
            'animate_subfacets': False,
            'animate_final_day_temp_distribution': False,
            'animate_shadowing': False,
            
            # Shape model settings (dummy)
            'shape_model_file': 'cube_without_roughness.obj',
            'ra_degrees': 0.0,
            'dec_degrees': 90.0,
            'rotation_period_hours': ROTATION_PERIOD_HOURS,
            
            # Physical props (will be overwritten by generator)
            'density': 1000.0,
            'specific_heat_capacity': 1000.0,
            'thermal_inertia': 100.0 
        }
        
        # Also set attributes for Config compatibility if needed by other utils
        self.load_settings()
        
        # Override path to avoid errors
        self.path_to_shape_model_file = None


def rotate_to_lat(vector, lat_degrees):
    """
    Rotate a vector (initially up Z, Lat 90) to a new latitude.
    Rotation is around Y-axis by (90 - lat) degrees.
    """
    angle_rad = np.radians(90.0 - lat_degrees)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    Ry = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    return np.dot(Ry, vector)

def simulate_crater_diurnal_cycle(theta, opening_angle, latitude, config, n_timesteps):
    """
    Run a full diurnal simulation for a single crater configuration using STANDARD `Simulation` logic.
    Instead of internal Facet logic, we build a `shape_model` from the crater sub-facets
    and run the full rigorous simulation pipeline.
    """
    # 1. Setup Simulation Object
    simulation = Simulation(config)
    
    simulation.timesteps_per_day = n_timesteps
    simulation.rotation_period_hours = ROTATION_PERIOD_HOURS 
    simulation.density = 2000.0 # Standard Value
    simulation.specific_heat_capacity = 1000.0 # Standard Value
    omega = 2 * np.pi / (simulation.rotation_period_hours * 3600)
    simulation.angular_velocity = omega
    
    # Calculate Thermal Inertia from Dimensionless Theta
    # P = I * sqrt(omega) / (epsilon * sigma * Tss^3)
    # Tss = ((1-A) * S / (epsilon * sigma))^0.25

    
    solar_flux = simulation.solar_luminosity / (4 * np.pi * simulation.solar_distance_m**2)
    boltzmann = 5.670374419e-8
    epsilon = simulation.emissivity
    albedo = simulation.albedo
    
    # Sub-solar temperature
    val = ((1 - albedo) * solar_flux) / (epsilon * boltzmann)
    tss = val ** 0.25
    
    # Avoid division by zero
    if omega <= 0: omega = 1e-9
    
    thermal_inertia = (theta * epsilon * boltzmann * (tss**3)) / np.sqrt(omega)
    
    # Update Simulation with derived thermal inertia
    simulation.thermal_inertia = thermal_inertia
    # Update skin depth etc
    simulation.thermal_conductivity = (simulation.thermal_inertia**2) / (simulation.density * simulation.specific_heat_capacity)
    simulation.skin_depth = (simulation.thermal_conductivity / (simulation.density * simulation.specific_heat_capacity * simulation.angular_velocity)) ** 0.5
    simulation.layer_thickness = 8 * simulation.skin_depth / simulation.n_layers
    simulation.thermal_diffusivity = simulation.thermal_conductivity / (simulation.density * simulation.specific_heat_capacity)
    
    # 2. Generate Crater Mesh
    # Create Dummy Facet (Lat 90, Z-up) to generate canonical mesh
    dummy_normal = np.array([0.0, 0.0, 1.0])
    dummy_vs = [np.array([-1,-1,0]), np.array([1,-1,0]), np.array([0,1,0])]
    dummy_facet = Facet(dummy_normal, dummy_vs, 1, 1, 1, False)
    
    config.kernel_profile_angle_degrees = opening_angle
    # Use config value for subfacets_count (already set to 300)
    config.apply_kernel_based_roughness = True
    
    # Generate canonical mesh inside the class
    dummy_facet.generate_spherical_depression(config, simulation)
    
    # Extract Sub-Facets from CANONICAL
    if not hasattr(Facet, '_canonical_subfacet_mesh') or Facet._canonical_subfacet_mesh is None:
         # Force regeneration if mesh doesn't exist
         # (subfacets_count already set in config from CRATER_SUBFACETS)
         dummy_facet.generate_spherical_depression(config, simulation)
         
    mesh = Facet._canonical_subfacet_mesh
    n_facets = len(mesh)
    
    shape_model = []
    
    # Sun direction with declination (latitude proxy for solar illumination angle)
    # NOTE: This approach keeps crater at equator and tilts sun to vary illumination.
    # "Latitude" here is a PARAMETRIC variable controlling solar zenith angle,
    # not geographic position. This ensures crater is always illuminated for LUT generation.
    sun_declination = latitude
    lat_rad = np.radians(sun_declination)
    simulation.sunlight_direction = np.array([np.cos(lat_rad), 0.0, np.sin(lat_rad)])
    
    # Rotate canonical crater (opening toward +Z) to equator (opening toward +X)
    rotation_to_equator = calculate_rotation_matrix(np.array([0.0, 1.0, 0.0]), np.pi/2)
    
    for entry in mesh:
        canonical_n = np.array(entry['normal'])
        canonical_v = np.array(entry['vertices']) # (3,3)
        
        # Apply rotation: canonical → equator
        rotated_n = np.dot(rotation_to_equator, canonical_n)
        rotated_v = np.array([np.dot(rotation_to_equator, v) for v in canonical_v])
        
        # Create full Facet object
        new_f = Facet(
            rotated_n, 
            rotated_v, 
            simulation.timesteps_per_day, 
            simulation.max_days, 
            simulation.n_layers, 
            config.calculate_energy_terms
        )
        shape_model.append(new_f)
    
    # 3. Setup Thermal Data
    thermal_data = ThermalData(n_facets, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)
    
    # 4. Calculate Visible Facets
    # All facets in the crater see each other, but NOT themselves
    all_indices = np.arange(n_facets)
    for i, f in enumerate(shape_model):
        # Exclude self from visible facets
        f.visible_facets = np.concatenate([all_indices[:i], all_indices[i+1:]])
    # set_visible_facets expects a LIST of arrays (one per facet)
    visible_facets_list = [np.concatenate([all_indices[:i], all_indices[i+1:]]) for i in range(n_facets)]
    thermal_data.set_visible_facets(visible_facets_list) 
    
    # 5. Calculate View Factors (Standard + Thermal)
    # Use configured ray count for energy conservation
    all_view_factors = calculate_all_view_factors(shape_model, thermal_data, config, VIEW_FACTOR_RAYS)
    thermal_data.set_secondary_radiation_view_factors(all_view_factors)
    
    thermal_view_factors = calculate_thermal_view_factors(shape_model, thermal_data, config)
    # Correct conversion to Numba List
    from numba.typed import List
    numba_vfs = List()
    for vf in thermal_view_factors: # Note: function returns list of lists usually
        arr = np.array(vf, dtype=np.float64)
        if arr.size == 0:
            numba_vfs.append(np.array([], dtype=np.float64))
        else:
            numba_vfs.append(arr.flatten() if arr.ndim > 1 else arr)
    thermal_data.thermal_view_factors = numba_vfs

    # 6. Calculate Insolation (with Shadowing)
    thermal_data = calculate_insolation(thermal_data, shape_model, simulation, config)
    

    
    # 7. Run Solver (Standard Implicit)
    from src.model.solvers import TemperatureSolverFactory
    
    solver = TemperatureSolverFactory.create(config.temp_solver)
    thermal_data = solver.initialize_temperatures(thermal_data, simulation, config)
    
    res = solver.solve(thermal_data, shape_model, simulation, config)
    
    if res['final_day_temperatures'] is None:
        raise ValueError(f"Solver failed for Theta={theta}, lat={latitude}")
        
    rough_temps = res['final_day_temperatures'] # (N_facets, T)
    
    # 8. Solve Smooth Reference
    # Create flat facet at SAME LATITUDE as rough crater for proper comparison
    # Normal should point at same angle as crater opening (latitude from equator)
    lat_rad = np.radians(latitude)
    flat_n = np.array([1.0, 0.0, 0.0])  # Normal pointing toward sun at equator
    flat_n /= np.linalg.norm(flat_n)
    
    # Vertices perpendicular to normal
    # Use two perpendicular vectors in the plane perpendicular to flat_n
    # Scaled to give area = 1.0 to match crater opening area
    scale = 1.0 / np.sqrt(2.0)
    
    # Create perpendicular basis vectors
    if abs(flat_n[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])
    
    u = np.cross(flat_n, up)
    u /= np.linalg.norm(u)
    v = np.cross(flat_n, u)
    
    flat_v = [
        -scale * u - scale * v,
        scale * u - scale * v,
        scale * v
    ]
    
    flat_facet = Facet(flat_n, flat_v, simulation.timesteps_per_day, simulation.max_days, simulation.n_layers, config.calculate_energy_terms)
    
    # Set world_to_local_rotation to match the crater's rotation at equator
    flat_facet.world_to_local_rotation = rotation_to_equator.T
    
    smooth_shape = [flat_facet]
    
    smooth_data = ThermalData(1, simulation.timesteps_per_day, simulation.n_layers, simulation.max_days, config.calculate_energy_terms)
    # set_visible_facets expects a LIST of arrays (one per facet)
    # For a single isolated facet, it has no visible neighbors
    smooth_data.set_visible_facets([np.array([], dtype=np.int64)])
    flat_facet.visible_facets = np.array([], dtype=np.int64)
    
    # Dummy View Factors
    smooth_data.set_secondary_radiation_view_factors([np.array([])])
    # Empty thermal view factors
    from numba.typed import List
    numba_vfs_smooth = List()
    # Need to match the expected type inside Numba functions (float64[:])
    # The solver expects a list of arrays. For a single facet with no self-view, it's an empty array.
    numba_vfs_smooth.append(np.array([], dtype=np.float64))
    
    smooth_data.thermal_view_factors = numba_vfs_smooth
    
    smooth_data = calculate_insolation(smooth_data, smooth_shape, simulation, config)
    # Create fresh solver instance for smooth reference to avoid state contamination
    smooth_solver = TemperatureSolverFactory.create(config.temp_solver)
    smooth_data = smooth_solver.initialize_temperatures(smooth_data, simulation, config)
    res_smooth = smooth_solver.solve(smooth_data, smooth_shape, simulation, config)
    
    smooth_temps = res_smooth['final_day_temperatures'][0] # (T,)

    # Reconstruct Sun Vectors for downstream geometry calc
    timesteps = simulation.timesteps_per_day
    sun_vectors_body = np.zeros((timesteps, 3))
    
    for t in range(timesteps):
        rot = calculate_rotation_matrix(simulation.rotation_axis, (2 * np.pi / timesteps) * t)
        # S_body = R^T * S_inertial
        s_body = np.dot(rot.T, simulation.sunlight_direction)
        sun_vectors_body[t] = s_body / np.linalg.norm(s_body)

    # Return flat_facet as the geometry reference
    return rough_temps, smooth_temps, flat_facet, sun_vectors_body



def plot_wireframe(mesh, filename="TEMPEST_RAD/diagnostics/plots/crater_wireframe.png"):
    """
    Save a 3D wireframe plot of the crater mesh.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = []
    for entry in mesh:
        # entry['vertices'] is (3,3) array
        v = entry['vertices']
        # Close the loop for wireframe
        v_loop = np.vstack([v, v[0]])
        ax.plot(v_loop[:,0], v_loop[:,1], v_loop[:,2], 'k-', lw=0.5, alpha=0.3)
        vertices.append(v)
        
    vertices_flat = np.vstack(vertices).reshape(-1, 3)
    max_range = np.array([vertices_flat[:,0].max()-vertices_flat[:,0].min(), 
                          vertices_flat[:,1].max()-vertices_flat[:,1].min(), 
                          vertices_flat[:,2].max()-vertices_flat[:,2].min()]).max() / 2.0

    mid_x = (vertices_flat[:,0].max()+vertices_flat[:,0].min()) * 0.5
    mid_y = (vertices_flat[:,1].max()+vertices_flat[:,1].min()) * 0.5
    mid_z = (vertices_flat[:,2].max()+vertices_flat[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title(f"Crater Mesh ({len(mesh)} facets)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    out_dir = os.path.dirname(filename)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved wireframe to {filename}")

def process_single_case(theta, opening_angle, lat, config):
    """
    Run simulation and compute viewing geometry grid.
    Returns: 4D array (Time, Wave, Emi, Azi), 1D array (Normalization_Factor)
    """
    try:
        rough_temps_sim, smooth_temps_sim, facet, sun_vectors_sim = simulate_crater_diurnal_cycle(theta, opening_angle, lat, config, SIM_TIMESTEPS)
    except Exception as e:
        print(f"Error in simulation (Th={theta}, ang={opening_angle}, lat={lat}): {e}")
        traceback.print_exc()
        # Return scalar nan for factor
        return np.full((LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS), len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)), np.nan, dtype=np.float32), np.nan

    # Resample to LUT grid
    indices = np.linspace(0, SIM_TIMESTEPS-1, LUT_TIMESTEPS, dtype=int)
    rough_temps = rough_temps_sim[:, indices]
    smooth_temps = smooth_temps_sim[indices]
    sun_vectors = sun_vectors_sim[indices]
    
    # Result: Time x Wavelength x Emi x Azi
    result_grid = np.zeros((LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS), len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)), dtype=np.float32)
    
    # Geometry Arrays
    if not hasattr(Facet, '_canonical_normals') or Facet._canonical_normals is None:
        mesh = Facet._canonical_subfacet_mesh
        Facet._canonical_normals = np.array([entry['normal'] for entry in mesh], dtype=np.float64)
        Facet._canonical_areas = np.array([entry['area'] for entry in mesh], dtype=np.float64)
        Facet._canonical_subfacet_triangles = np.array([entry['vertices'] for entry in mesh], dtype=np.float64)
        if 'center' in mesh[0]:
            Facet._canonical_subfacet_centers = np.array([entry['center'] for entry in mesh], dtype=np.float64)
        else:
            Facet._canonical_subfacet_centers = np.array([np.mean(entry['vertices'], axis=0) for entry in mesh], dtype=np.float64)

    normals = Facet._canonical_normals
    areas = Facet._canonical_areas
    triangles = Facet._canonical_subfacet_triangles
    centers = Facet._canonical_subfacet_centers
    facet_normal = facet.normal / np.linalg.norm(facet.normal)
    
    # Pre-calculated Geometry terms
    em_rad = np.radians(EMISSION_ANGLES)
    az_rad = np.radians(AZIMUTH_ANGLES)
    d_em = np.diff(em_rad); d_em = np.append(d_em, d_em[-1])
    d_az = np.diff(az_rad); d_az = np.append(d_az, d_az[-1])

    # Pre-compute cos(e)*sin(e) solid-angle weight grid for normalization.
    # Energy conservation requires: ∫ R(e,φ) cos(e) sin(e) de dφ = π
    # i.e., the cos(e)*sin(e)-weighted mean of R must equal 1.0.
    cos_e_arr = np.cos(em_rad)
    sin_e_arr = np.sin(em_rad)
    cse_weights = np.zeros((len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)))
    for ie in range(len(EMISSION_ANGLES)):
        for ia in range(len(AZIMUTH_ANGLES)):
            cse_weights[ie, ia] = cos_e_arr[ie] * sin_e_arr[ie] * d_em[ie] * d_az[ia] * 2.0
    ref_integral = np.sum(cse_weights)  # ≈ π

    # Loop over Timesteps
    for t_idx in range(LUT_TIMESTEPS):
        t_sub = rough_temps[:, t_idx] # (N_sub,)
        t_smooth = smooth_temps[t_idx] # scalar
        
        # Sun Basis for Azimuth
        sun_vec_norm = sun_vectors[t_idx] / np.linalg.norm(sun_vectors[t_idx])
        s_dot_n = np.dot(sun_vec_norm, facet_normal)
        sun_proj = sun_vec_norm - s_dot_n * facet_normal
        if np.linalg.norm(sun_proj) > 1e-6:
            basis_x = sun_proj / np.linalg.norm(sun_proj) 
        else:
            up = np.array([0, 0, 1]) if np.abs(np.dot([0,0,1], facet_normal)) < 0.9 else np.array([1, 0, 0])
            basis_x = np.cross(facet_normal, up)
            basis_x /= np.linalg.norm(basis_x)
        basis_y = np.cross(facet_normal, basis_x)
        
        # Loop Viewing Angles
        for i_e, emi_deg in enumerate(EMISSION_ANGLES):
            emi_rad = np.radians(emi_deg)
            sin_e = np.sin(emi_rad)
            cos_e = np.cos(emi_rad)

            for i_a, azi_deg in enumerate(AZIMUTH_ANGLES):
                azi_rad = np.radians(azi_deg)
                
                # View Vector
                v_local_x = sin_e * np.cos(azi_rad)
                v_local_y = sin_e * np.sin(azi_rad)
                v_local_z = cos_e
                view_vec = v_local_x * basis_x + v_local_y * basis_y + v_local_z * facet_normal
                view_vec /= np.linalg.norm(view_vec)
                
                # --- Determine Visibility via ray-tracing ---
                # We use _process_incident_packet for its shadow-checking logic,
                # but we do NOT use the redistributed E_vis values for radiance.
                # E_vis > 0 identifies which subfacets are visible from this direction.
                packet = (1.0, view_vec, 'visible') 
                E_vis, _, _, _ = Facet._process_incident_packet(
                    packet, facet.world_to_local_rotation, facet.area, normals, areas, triangles, centers, 0.0, 0.0
                )
                visible_mask = E_vis > 0
                
                # Compute cos(theta_i) = dot(subfacet normal, view direction) in local frame
                d_local = facet.world_to_local_rotation.dot(view_vec)
                cos_i_all = normals.dot(d_local)
                
                # Loop Wavelengths
                for i_w, wave in enumerate(WAVELENGTHS_MICRONS):
                    # 1. Compute spectral intensity (power per steradian per micron)
                    # CORRECT formula for outgoing radiance from a rough crater:
                    #   I_rough = Σ_visible B(λ, T_i) * A_i * cos(θ_i)
                    # where cos(θ_i) is angle between subfacet normal and view direction.
                    rad_sub = planck_function(wave, t_sub)
                    cos_i_vis = np.maximum(cos_i_all[visible_mask], 0.0)
                    rough_intensity = np.sum(rad_sub[visible_mask] * areas[visible_mask] * cos_i_vis)
                    
                    # Smooth intensity: I_smooth = B(λ, T_smooth) * A_aperture * cos(e)
                    if t_smooth < 5.0:
                        smooth_intensity = 0.0
                    else:
                        rad_smooth_val = planck_function(wave, t_smooth)
                        smooth_intensity = rad_smooth_val * facet.area * cos_e

                    # Store radiance ratio: R = I_rough / I_smooth = L_rough / L_smooth
                    if smooth_intensity < 1e-15:
                        result_grid[t_idx, i_w, i_e, i_a] = 1.0 # Fallback
                    else:
                        ratio = rough_intensity / smooth_intensity
                        # Clamp to physical range to avoid interpolation artifacts
                        # at extreme viewing geometries (e~89°) or Wien-tail conditions
                        result_grid[t_idx, i_w, i_e, i_a] = np.clip(ratio, 0.0, 50.0)

    # --- PER-(TIME, WAVELENGTH) NORMALIZATION ---
    # For energy conservation, the cos(e)*sin(e)-weighted average of R over
    # all viewing directions must equal 1.0 at each timestep and wavelength.
    # This ensures that applying R to a Lambertian smooth surface conserves
    # total emitted power, regardless of T_smooth.
    #
    # Constraint: ∫ R(e,φ) * cos(e) * sin(e) de dφ = π
    #            ⟹ <R>_{cos·sin} = 1.0
    #
    # We enforce this independently for each (t, λ) slice.
    norm_factors_tw = np.zeros((LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS)))
    for t_idx in range(LUT_TIMESTEPS):
        for i_w in range(len(WAVELENGTHS_MICRONS)):
            raw_slice = result_grid[t_idx, i_w, :, :]  # (Emi, Azi)
            weighted_sum = np.sum(raw_slice * cse_weights)
            if weighted_sum > 1e-15:
                alpha = ref_integral / weighted_sum
            else:
                alpha = 1.0
            result_grid[t_idx, i_w, :, :] *= alpha
            norm_factors_tw[t_idx, i_w] = alpha
    
    # Diagnostic: average normalization factor across all timesteps/wavelengths
    norm_factor = float(np.mean(norm_factors_tw))
                
    return result_grid, norm_factor

def main():
    print("="*80)
    print("TEMPEST Roughness LUT Generator")
    print("="*80)
    start_time = time.time()
    
    # Create Reference Config
    config = ReferenceConfig()
    
    # Generate and Plot Wireframe (Requirement 5)
    print("Generating geometry preview...")
    dummy_normal = np.array([0.0, 0.0, 1.0])
    dummy_vs = [np.array([-1,-1,0]), np.array([1,-1,0]), np.array([0,1,0])]
    dummy_facet = Facet(dummy_normal, dummy_vs, 1, 1, 1, False)
    
    config.kernel_profile_angle_degrees = OPENING_ANGLES[0]
    config.apply_kernel_based_roughness = True
    dummy_facet.generate_spherical_depression(config, Simulation(config))
    plot_wireframe(Facet._canonical_subfacet_mesh)
    
    # Pre-calculate View Factors (Single Threaded to avoid race condition)
    print("Pre-calculating view factors to avoid parallel race condition...")
    # We need to construct a shape model exactly like simulate_crater_diurnal_cycle does
    # Use Lat 0.0 logic (rotation to equator is constant)
    mesh = Facet._canonical_subfacet_mesh
    n_facets = len(mesh)
    shape_model = []
    rotation_to_equator = calculate_rotation_matrix(np.array([0.0, 1.0, 0.0]), np.pi/2)
    for entry in mesh:
        canonical_n = np.array(entry['normal'])
        canonical_v = np.array(entry['vertices'])
        rotated_n = np.dot(rotation_to_equator, canonical_n)
        rotated_v = np.array([np.dot(rotation_to_equator, v) for v in canonical_v])
        new_f = Facet(rotated_n, rotated_v, 1, 1, 1, False)
        shape_model.append(new_f)
        
    thermal_data = ThermalData(n_facets, 1, 1, 1, False)
    # Visible facets logic
    all_indices = np.arange(n_facets)
    visible_facets_list = [np.concatenate([all_indices[:i], all_indices[i+1:]]) for i in range(n_facets)]
    thermal_data.set_visible_facets(visible_facets_list)
    
    # Trigger calculation and save
    calculate_all_view_factors(shape_model, thermal_data, config, VIEW_FACTOR_RAYS)
    calculate_thermal_view_factors(shape_model, thermal_data, config)
    print("View factor cache initialized.")

    # Tasks
    tasks = []
    for theta in THETA_VALUES:
        for angle in OPENING_ANGLES:
            for lat in LATITUDE_VALUES:
                tasks.append((theta, angle, lat))
    
    print(f"\nConfiguration:")
    print(f"  Theta values: {len(THETA_VALUES)} ({THETA_VALUES[0]:.3f} to {THETA_VALUES[-1]:.3f})")
    print(f"  Latitudes: {len(LATITUDE_VALUES)}° (0° to {LATITUDE_VALUES[-1]:.0f}°)")
    print(f"  Wavelengths: {len(WAVELENGTHS_MICRONS)} bands")
    print(f"  Viewing geometry: {len(EMISSION_ANGLES)} × {len(AZIMUTH_ANGLES)} angles")
    print(f"  Total cases: {len(tasks)}")
    print(f"  Parallel jobs: 4")
    print(f"\nOutput shape: {len(THETA_VALUES)}×{len(OPENING_ANGLES)}×{len(LATITUDE_VALUES)}×{LUT_TIMESTEPS}×{len(WAVELENGTHS_MICRONS)}×{len(EMISSION_ANGLES)}×{len(AZIMUTH_ANGLES)}")
    print(f"\nStarting generation...\n")
    
    n_jobs = 4
    
    # Use joblib's verbose mode + tqdm for real completion tracking
    print(f"  (Each '.' below = 1 completed case)\n")
    completed = [0]
    total = len(tasks)
    pbar = tqdm(total=total, desc="Processing cases", unit="case")
    
    def _run_and_track(theta, angle, lat, config):
        result = process_single_case(theta, angle, lat, config)
        return result
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_and_track)(theta, angle, lat, config) 
        for theta, angle, lat in tasks
    )
    pbar.update(total - pbar.n)  # Ensure bar reaches 100%
    pbar.close()
    
    # Assemble Tensor
    print("\nAssembling tensor...")
    lut_tensor = np.zeros((len(THETA_VALUES), len(OPENING_ANGLES), len(LATITUDE_VALUES), 
                           LUT_TIMESTEPS, len(WAVELENGTHS_MICRONS), 
                           len(EMISSION_ANGLES), len(AZIMUTH_ANGLES)), dtype=np.float32)
                           
    factors_tensor = np.zeros((len(THETA_VALUES), len(OPENING_ANGLES), len(LATITUDE_VALUES)), dtype=np.float32)
    
    idx = 0
    for i_th, theta in enumerate(THETA_VALUES):
        for i_ang, angle in enumerate(OPENING_ANGLES):
            for i_lat, lat in enumerate(LATITUDE_VALUES):
                if results[idx] is not None:
                    grid, factor = results[idx]
                    lut_tensor[i_th, i_ang, i_lat, :, :, :, :] = grid
                    factors_tensor[i_th, i_ang, i_lat] = factor
                    print(f"  Lat {lat:5.1f}°: norm_factor = {factor:.4f}")
                idx += 1
    
    print("Tensor assembled.")
                
    # Save
    print(f"\n{'='*80}")
    print(f"Saving to {OUTPUT_FILE}...")
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset("lut", data=lut_tensor)
        f.create_dataset("normalization_factors", data=factors_tensor)
        f.create_dataset("theta", data=THETA_VALUES)
        f.create_dataset("opening_angle", data=OPENING_ANGLES)
        f.create_dataset("wavelength", data=WAVELENGTHS_MICRONS)
        f.create_dataset("latitude", data=LATITUDE_VALUES)
        f.create_dataset("emission", data=EMISSION_ANGLES)
        f.create_dataset("azimuth", data=AZIMUTH_ANGLES)

    # Print summary
    lut_size_bytes = os.path.getsize(OUTPUT_FILE)
    lut_size_mb = lut_size_bytes / (1024 * 1024)
    elapsed_time = time.time() - start_time

    print(f"\nLUT generation complete!")
    print(f"  File: {OUTPUT_FILE}")
    print(f"  Size: {lut_size_mb:.1f} MB")
    print(f"  Time: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
    print(f"  Rate: {len(tasks)/elapsed_time:.2f} cases/second")
    print("="*80)

if __name__ == "__main__":
    main()
