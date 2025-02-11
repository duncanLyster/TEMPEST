# scripts/analyze_flyby_temperatures.py

import json
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from src.utilities.locations import Locations
from src.utilities.config import Config
from stl import mesh
from src.utilities.utils import rays_triangles_intersection
from src.model.emission import EPFLookupTable
import pyvista as pv

############### CONFIGURATION ###############
# Flyby parameters (all distances in km)
INITIAL_DISTANCE = 1000
CLOSEST_APPROACH = 400
FINAL_DISTANCE = 1000
N_POINTS = 100
SUBSOLAR_LONGITUDE = 340  # degrees
INSTRUMENT_IFOV = 7.3  # mrad - Only needed for visualisation
VISUALIZE = True # Set to True to visualize the flyby
CHECK_FOLDER = False # Set to True to check the folder
################################################

def get_output_folders(base_dir):
    """Retrieve all folders in the base directory sorted by modification time."""
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folders.sort(key=lambda f: os.path.getmtime(os.path.join(base_dir, f)), reverse=True)
    return folders

def get_user_confirmation(default_folder, available_folders):
    """Ask user if they want to use default most recent folder or choose another."""
    if CHECK_FOLDER:
        print(f"The most recent folder is: {default_folder}")
        use_default = input("Do you want to use this folder? (y/n): ").strip().lower()

        if use_default == 'y':
            return default_folder
        else:
            print("\nAvailable folders:")
            for i, folder in enumerate(available_folders):
                print(f"{i}: {folder}")
            
            while True:
                try:
                    choice = int(input("Select a folder by number: "))
                    if 0 <= choice < len(available_folders):
                        return available_folders[choice]
                    else:
                        print(f"Invalid selection. Please select a number between 0 and {len(available_folders) - 1}.")
                except ValueError:
                    print("Please enter a valid number.")

    else:
        return default_folder

def visualize_flyby(shape_model, observer_positions, temperatures):
    """
    Visualize the flyby geometry with the shape model and trajectory.
    """
    # Create PyVista plotter
    plotter = pv.Plotter()
    
    # Convert shape model to PyVista PolyData
    vertices = shape_model.vectors.reshape(-1, 3)
    n_triangles = len(shape_model.vectors)
    
    # Debug prints for vertices
    print("Vertex range:")
    print(f"X: {vertices[:, 0].min():.2f} to {vertices[:, 0].max():.2f}")
    print(f"Y: {vertices[:, 1].min():.2f} to {vertices[:, 1].max():.2f}")
    print(f"Z: {vertices[:, 2].min():.2f} to {vertices[:, 2].max():.2f}")
    
    # Create faces array in PyVista format
    faces = np.zeros(4 * n_triangles, dtype=np.int32)
    faces[::4] = 3
    faces[1::4] = np.arange(0, 3 * n_triangles, 3)
    faces[2::4] = np.arange(1, 3 * n_triangles, 3)
    faces[3::4] = np.arange(2, 3 * n_triangles, 3)
    
    mesh = pv.PolyData(vertices, faces)
    print(f"\nMesh bounds: {mesh.bounds}")
    
    # Debug prints
    print(f"\nNumber of vertices: {len(vertices)}")
    print(f"Number of faces: {n_triangles}")
    print(f"Temperature array shape: {temperatures.shape}")
    print(f"Observer positions range:")
    print(f"X: {observer_positions[:, 0].min():.2f} to {observer_positions[:, 0].max():.2f}")
    print(f"Y: {observer_positions[:, 1].min():.2f} to {observer_positions[:, 1].max():.2f}")
    print(f"Z: {observer_positions[:, 2].min():.2f} to {observer_positions[:, 2].max():.2f}")
    
    # Scale up the mesh to make it more visible
    scale_factor = CLOSEST_APPROACH  # Make the asteroid 10% of the closest approach distance
    mesh.points *= scale_factor
    
    # Add temperature data to mesh
    if len(temperatures) == n_triangles:
        mesh.cell_data['Temperature'] = temperatures
        plotter.add_mesh(mesh, scalars='Temperature', cmap='inferno', 
                        show_edges=True, lighting=True,
                        scalar_bar_args={'title': 'Temperature (K)'})
    else:
        # If temperatures don't match, just show the shape model in a solid color
        plotter.add_mesh(mesh, color='lightgray', show_edges=True, lighting=True)
        print("Warning: Temperature array length doesn't match number of triangles")
    
    # Add flyby trajectory
    trajectory = pv.PolyData(observer_positions)
    plotter.add_mesh(trajectory, color='red', render_points_as_spheres=True, 
                    point_size=10, line_width=2)
    
    # Connect trajectory points with a line
    lines = np.column_stack([
        np.full(len(observer_positions)-1, 2),
        np.arange(len(observer_positions)-1),
        np.arange(1, len(observer_positions))
    ]).flatten()
    trajectory_line = pv.PolyData(observer_positions, lines=lines)
    plotter.add_mesh(trajectory_line, color='red', line_width=2)
    
    # Add coordinate axes for scale
    plotter.add_axes()
    
    # Add text showing direction of motion
    start_pos = observer_positions[0]
    plotter.add_point_labels([start_pos], ['Flyby Start'])
    
    # Reset camera to see all objects
    plotter.reset_camera()
    
    # Show the plot
    plotter.show(interactive=True)

def calculate_mean_observed_temperature(shape_model, temperatures, observer_position, config):
    """Calculate mean temperature visible from a given observer position using radiance-weighted average."""
    STEFAN_BOLTZMANN = 5.67e-8
    
    # Initialize EPF lookup table using config's emission_lut setting
    epf_lut = EPFLookupTable(config.emission_lut)
    
    total_radiance = 0.0
    total_weight = 0.0
    
    for idx, facet in enumerate(shape_model.vectors):
        # Calculate facet normal and center
        v1, v2, v3 = facet
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)
        center = np.mean(facet, axis=0)
        
        # Vector from facet to observer
        to_observer = observer_position - center
        to_observer_norm = to_observer / np.linalg.norm(to_observer)
        
        # Check if facet is facing observer
        cos_emission = np.dot(normal, to_observer_norm)
        if cos_emission <= 0:
            continue
            
        # Calculate emission angle and get EPF value
        emission_angle = np.degrees(np.arccos(cos_emission))
        epf = epf_lut.query(emission_angle)
        
        # Calculate projected area
        area = 0.5 * np.linalg.norm(normal)
        projected_area = area * cos_emission
        
        # Check for occlusion using existing ray-triangle intersection code
        ray_origin = center
        ray_direction = to_observer_norm.reshape(1, 3)
        triangles = shape_model.vectors
        intersections, _ = rays_triangles_intersection(ray_origin, ray_direction, triangles)
        
        if np.any(intersections[0]):  # Facet is occluded
            continue
            
        # Calculate radiance using Stefan-Boltzmann law
        temperature = temperatures[idx]
        radiance = (temperature**4)
        
        # Add weighted contribution
        weight = projected_area * epf
        total_radiance += radiance * weight
        total_weight += weight
    
    if total_weight == 0:
        return np.nan
        
    # Convert back to effective temperature
    mean_radiance = total_radiance / total_weight
    mean_temperature = mean_radiance ** 0.25
    
    return mean_temperature

def visualize_ifov_flyby(shape_model, observer_positions, temperatures):
    """
    Visualize the flyby geometry with simplified target representation and instrument pointing.
    Shows global view (with target as dot) and detailed instrument perspective.
    """
    # Create PyVista plotter with two subplots side by side
    pl = pv.Plotter(shape=(1, 2))
    
    # Convert shape model to PyVista PolyData (needed only for instrument view)
    vertices = shape_model.vectors.reshape(-1, 3)
    n_triangles = len(shape_model.vectors)
    faces = np.zeros(4 * n_triangles, dtype=np.int32)
    faces[::4] = 3
    faces[1::4] = np.arange(0, 3 * n_triangles, 3)
    faces[2::4] = np.arange(1, 3 * n_triangles, 3)
    faces[3::4] = np.arange(2, 3 * n_triangles, 3)
    mesh = pv.PolyData(vertices, faces)
    
    # Add temperature data to mesh
    if len(temperatures) == n_triangles:
        mesh.cell_data['Temperature'] = temperatures
    
    # First subplot - Simplified global view
    pl.subplot(0, 0)
    pl.add_text("Global View", position='upper_edge')
    
    # Add target as green dot at origin
    target_point = pv.PolyData(np.array([[0, 0, 0]]))
    pl.add_mesh(target_point, color='green', render_points_as_spheres=True,
               point_size=15, name='target_point')
    
    # Add trajectory line
    lines = np.column_stack([
        np.full(len(observer_positions)-1, 2),
        np.arange(len(observer_positions)-1),
        np.arange(1, len(observer_positions))
    ]).flatten()
    trajectory_line = pv.PolyData(observer_positions, lines=lines)
    pl.add_mesh(trajectory_line, color='red', line_width=2, opacity=0.3)
    
    # Second subplot - Instrument view
    pl.subplot(0, 1)
    pl.add_text("Instrument View", position='upper_edge')
    
    # Create a separate mesh for the instrument view
    instrument_mesh = mesh.copy()
    if len(temperatures) == n_triangles:
        pl.add_mesh(instrument_mesh, scalars='Temperature', cmap='inferno',
                   show_edges=True, lighting=True,
                   scalar_bar_args={'title': 'Temperature (K)'})
    else:
        pl.add_mesh(instrument_mesh, color='lightgray', show_edges=True, lighting=True)
    
    class State:
        def __init__(self):
            self.current_frame = 0
            self.instrument_cam = None
            self.circle_actor = None
            self.is_playing = True
    state = State()
    
    def update_instrument_view(frame):
        state.current_frame = frame % len(observer_positions)  # Loop back to start
        
        # Get current observer position
        pos = observer_positions[state.current_frame]
        distance = np.linalg.norm(pos)
        
        # Calculate view direction (towards origin)
        view_dir = -pos / distance
        
        # Convert IFOV from mrad to radians
        ifov_rad = INSTRUMENT_IFOV / 1000.0
        
        # Calculate IFOV footprint diameter
        footprint_diameter = 2 * distance * np.tan(ifov_rad/2)
        
        # Update global view (subplot 0)
        pl.subplot(0, 0)
        
        # Update spacecraft position
        pl.remove_actor('spacecraft_point')
        spacecraft_point = pv.PolyData(pos.reshape(1, 3))
        pl.add_mesh(spacecraft_point, color='red', render_points_as_spheres=True,
                   point_size=15, name='spacecraft_point')
        
        # Update pointing arrow with fixed width
        pl.remove_actor('pointing_arrow')
        arrow_length = distance * 0.8
        # Create line for arrow shaft
        line_points = np.array([pos, pos + view_dir * arrow_length])
        line = pv.Line(line_points[0], line_points[1])
        pl.add_mesh(line, color='blue', line_width=3, name='pointing_arrow_shaft')
        
        # Add arrow head (small cone at the end)
        head_height = arrow_length * 0.05  # 5% of arrow length
        head_radius = head_height * 0.3  # Fixed aspect ratio
        arrow_head = pv.Cone(
            center=pos + view_dir * (arrow_length - head_height/2),
            direction=view_dir,
            height=head_height,
            radius=head_radius
        )
        pl.add_mesh(arrow_head, color='blue', name='pointing_arrow_head')
        
        # Update instrument view (subplot 1)
        pl.subplot(0, 1)
        
        # Set up parallel projection for instrument view
        pl.camera.parallel_projection = True
        
        # Position camera at observer position
        pl.camera.position = pos
        pl.camera.focal_point = [0, 0, 0]
        pl.camera.up = [0, 0, 1]
        
        # Disable camera interactions for instrument view
        pl.disable()
        
        # Calculate parallel scale to match IFOV
        parallel_scale = distance * np.tan(ifov_rad/2)
        pl.camera.parallel_scale = parallel_scale
        
        # Update IFOV circle
        pl.remove_actor('ifov_circle')
        
        # Create circle points in the plane perpendicular to view direction
        n_points = 100
        theta = np.linspace(0, 2*np.pi, n_points)
        circle_scale = parallel_scale * 0.95
        
        # Calculate basis vectors for the circle plane
        basis1 = np.cross(view_dir, [0, 0, 1])
        basis1 = basis1 / np.linalg.norm(basis1)
        basis2 = np.cross(view_dir, basis1)
        basis2 = basis2 / np.linalg.norm(basis2)
        
        # Generate circle points in the view plane
        circle_points = np.zeros((n_points, 3))
        for i in range(n_points):
            circle_points[i] = (basis1 * np.cos(theta[i]) + 
                              basis2 * np.sin(theta[i])) * circle_scale
        
        # Offset circle to be slightly in front of the target
        circle_center = -view_dir * (distance * 0.99)
        circle_points += circle_center
        
        # Create circle polydata
        circle_poly = pv.PolyData(circle_points)
        circle_lines = np.column_stack([np.arange(n_points), 
                                      np.roll(np.arange(n_points), -1)])
        circle_poly.lines = np.hstack([np.full(len(circle_lines), 2)[:, None], circle_lines])
        
        # Add circle to view
        pl.add_mesh(circle_poly, color='yellow', line_width=2, name='ifov_circle')
        
        # Add information text
        info_text = (f"Distance: {distance:.1f} km\n"
                    f"IFOV footprint: {footprint_diameter:.1f} km")
        pl.remove_actor('info_text')
        pl.add_text(info_text, position='lower_left', font_size=10, name='info_text')
        
        pl.render()
    
    def toggle_animation(caller=None, event=None):
        state.is_playing = not state.is_playing
    
    def step_forward(caller=None, event=None):
        state.is_playing = False
        update_instrument_view(state.current_frame + 1)
    
    def step_backward(caller=None, event=None):
        state.is_playing = False
        update_instrument_view(state.current_frame - 1)
    
    def animation_callback(caller, event):
        if state.is_playing:
            update_instrument_view(state.current_frame + 1)
    
    # Set up animation timer correctly
    pl.iren.add_observer('TimerEvent', animation_callback)
    pl.iren.create_timer(100)  # Creates a repeating timer that triggers every 100 ms
    
    # Add key bindings
    pl.add_key_event('space', toggle_animation)
    pl.add_key_event('Right', step_forward)
    pl.add_key_event('Left', step_backward)
    
    # Add coordinate axes to global view only
    pl.subplot(0, 0)
    pl.add_axes()
    
    # Initialize the view
    update_instrument_view(0)
    
    # Enable interaction only for global view
    pl.subplot(0, 0)
    pl.enable()
    pl.subplot(0, 1)
    pl.disable()
    
    # Add text instructions
    pl.subplot(0, 0)
    instructions = (
        "Controls:\n"
        "Space - Play/Pause\n"
        "← → - Step through frames"
    )
    pl.add_text(instructions, position='lower_left', font_size=10)
    
    # Show the plot
    pl.show()

def main():
    # Explicitly specify the config file path
    config_path = "/Users/duncan/Desktop/DPhil/TEMPEST/private/data/config/dinkinesh/dinkinesh_config.yaml"
    config = Config(config_path)
    
    # Initialize locations
    locations = Locations()
    base_dir = os.path.join(locations.project_root, "outputs/remote_outputs")
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Exiting.")
        return
    
    available_folders = get_output_folders(base_dir)
    if not available_folders:
        print(f"No folders found in {base_dir}. Exiting.")
        return
    
    # The most recent folder
    default_folder = available_folders[0]

    # Ask the user for confirmation or selection
    selected_folder = get_user_confirmation(default_folder, available_folders)

    # Set the file paths
    json_file = os.path.join(base_dir, selected_folder, 'animation_params.json')
    npz_file = os.path.join(base_dir, selected_folder, 'animation_params.npz')

    # Check if both files exist
    if not os.path.exists(json_file) or not os.path.exists(npz_file):
        print(f"Required files not found in {selected_folder}. Exiting.")
        return
    
    # Load data from files
    print("Loading data files...")
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    data = np.load(npz_file, allow_pickle=True)
    
    # Load shape model from STL file
    shape_model_path = os.path.join(locations.project_root, json_data['args'][0])
    print(f"Loading shape model from: {shape_model_path}")
    try:
        shape_model = mesh.Mesh.from_file(shape_model_path)
    except Exception as e:
        print(f"Failed to load shape model: {e}")
        return
    
    # Get timesteps_per_day from json data
    timesteps_per_day = json_data['kwargs']['timesteps_per_day']

    # Convert subsolar longitude to timestep
    timestep = int((SUBSOLAR_LONGITUDE / 360.0) * timesteps_per_day)
    print(f"Using timestep {timestep} for subsolar longitude {SUBSOLAR_LONGITUDE}°")
    
    # Extract temperature data
    temperatures = data['plotted_variable_array']

    # Generate flyby trajectory
    angles = np.linspace(-np.pi/2, np.pi/2, N_POINTS)
    
    # Add safety check for angles near ±π/2
    safe_angles = np.clip(angles, -np.pi/2 + 0.1, np.pi/2 - 0.1)
    distances = CLOSEST_APPROACH / np.cos(safe_angles)
    
    # Calculate observer positions
    observer_positions = np.array([
        [distance * np.cos(angle),
         distance * np.sin(angle),
         0] for distance, angle in zip(distances, safe_angles)
    ])

    # Calculate temperatures and store positions
    mean_temps = []
    observation_angles = np.degrees(safe_angles)

    # Visualize if requested
    if VISUALIZE:
        print("Number of triangles:", len(shape_model.vectors))
        print("Length of temperature array:", len(temperatures))
        visualize_ifov_flyby(shape_model, observer_positions, temperatures[:, timestep])
        exit()
    
    print("\nCalculating temperatures along trajectory...")
    for distance, angle in zip(distances, safe_angles):
        observer_position = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            0
        ])
        
        temp = calculate_mean_observed_temperature(
            shape_model,
            temperatures[:, timestep],
            observer_position,
            config
        )
        mean_temps.append(temp)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(observation_angles, mean_temps, 'b-', marker='o')
    plt.xlabel('Observation Angle (degrees)')
    plt.ylabel('Mean Observed Temperature (K)')
    plt.title(f'Mean Temperature vs Observation Angle\nTimestep {timestep}')
    plt.grid(True)
    plt.ylim(0, 250)
    
    plt.axvline(x=0, color='k', linestyle=':')
    plt.axhline(y=np.mean(mean_temps), color='r', linestyle='--', 
                label='Mean Temperature')
    
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, selected_folder,
                              f'flyby_temperature_analysis_t{timestep}.png')
    plt.savefig(output_path)
    plt.show()
    
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    main()