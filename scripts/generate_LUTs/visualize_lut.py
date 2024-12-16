"""
Visualize BRDF lookup table data in 3D.
Shows incident ray (green) and exitant rays (red) with adjustable incidence angle.
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import sys

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parents[1] / "src"
sys.path.append(str(src_dir))

from src.utilities.locations import Locations

def create_ray_line(plotter, start, end, color, width=1.0):
    """Create a thin ray line."""
    line = pv.Line(start, end)
    return plotter.add_mesh(line, color=color, line_width=width)

def create_reference_disc():
    """Create a reference disc at the origin."""
    disc = pv.Disc(center=(0, 0, 0), normal=(0, 0, 1), inner=0, outer=0.2)
    return disc

def update_visualization(value):
    """Update the visualization based on slider value (incidence angle)."""
    global plotter, incident_ray, exit_rays
    
    # Clear previous rays
    if incident_ray is not None:
        plotter.remove_actor(incident_ray)
    for ray in exit_rays:
        plotter.remove_actor(ray)
    exit_rays.clear()
    
    # Convert slider value to radians
    inc_angle = np.radians(value)
    
    # Create incident ray as an arrow (starting above, ending at origin)
    incident_direction = np.array([np.sin(inc_angle), 0, -np.cos(inc_angle)])
    start_point = np.array([0, 0, 0]) - incident_direction  # Start 1 unit away from origin
    
    # Create arrow for incident ray
    incident_arrow = pv.Arrow(start=start_point, 
                            direction=incident_direction,
                            tip_length=0.1, 
                            tip_radius=0.01,
                            shaft_radius=0.005)
    incident_ray = plotter.add_mesh(incident_arrow, color='green')
    
    # Get current slice of BRDF data
    inc_idx = np.searchsorted(lut_data['incidence_angles'], value) - 1
    brdf_slice = lut_data['table'][inc_idx]
    
    # Sample every 10th angle
    theta_indices = range(0, len(lut_data['emission_angles']), 5)
    phi_indices = range(0, len(lut_data['azimuth_angles']), 5)
    
    # Create points and lines for all exit rays at once
    points = []
    lines = []
    point_count = 0
    
    # Always start from origin
    origin = np.array([0, 0, 0])
    
    for i in theta_indices:
        theta = lut_data['emission_angles'][i]
        for j in phi_indices:
            phi = lut_data['azimuth_angles'][j]
            if brdf_slice[i, j] > 0.01:  # Only show significant contributions
                theta_rad = np.radians(theta)
                phi_rad = np.radians(phi)
                
                # Convert spherical to Cartesian coordinates
                x = np.sin(theta_rad) * np.cos(phi_rad)
                y = np.sin(theta_rad) * np.sin(phi_rad)
                z = np.cos(theta_rad)
                
                direction = np.array([x, y, z])
                # Scale length by BRDF value NOTE: Cosine multiplication means you are not looking at the BRDF value
                length = brdf_slice[i, j] * np.cos(theta_rad) / np.max(brdf_slice)
                end_point = direction * length
                
                # Add points and line connectivity
                points.extend([origin, end_point])
                lines.append([2, point_count, point_count + 1])
                point_count += 2
    
    if points:  # Only create mesh if we have points
        points = np.array(points)
        lines = np.array(lines)
        poly_data = pv.PolyData(points, lines=lines)
        exit_rays = [plotter.add_mesh(poly_data, color='red', line_width=1.0)]
    
    plotter.render()

# Load LUT data
locations = Locations()
lut_path = locations.get_scattering_lut_path("lambertian.npy")
lut_data = np.load(lut_path, allow_pickle=True).item()

# Create visualization
plotter = pv.Plotter()
plotter.background_color = 'white'

# Add reference disc at origin
disc = create_reference_disc()
plotter.add_mesh(disc, color='gray', opacity=0.5)

# Initialize global variables for rays
incident_ray = None
exit_rays = []

# Add slider
plotter.add_slider_widget(
    update_visualization,
    [0, 90],
    title='Incidence Angle (degrees)',
    value=0,
    pointa=(0.1, 0.1),
    pointb=(0.9, 0.1),
    style='modern'
)

# Set initial camera position and view
plotter.camera_position = 'xz'
plotter.camera.zoom(1.5)

# Show the visualization
plotter.show()