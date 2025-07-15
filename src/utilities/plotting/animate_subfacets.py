import numpy as np
from pathlib import Path
import pyvista as pv
from src.utilities.config import Config
from src.model.simulation import Simulation
from src.model.facet import Facet
from src.model.solvers import TemperatureSolverFactory
from src.utilities.plotting.animate_model import animate_model

def main(config_path, facet_index):
    # --- Setup ---
    cfg = Config(config_path)
    sim = Simulation(cfg)
    # read the global shape model (facets) and do one roughness pass
    from tempest import read_shape_model
    shape = read_shape_model(cfg.path_to_shape_model_file,
                             sim.timesteps_per_day,
                             sim.n_layers,
                             sim.max_days,
                             cfg.calculate_energy_terms)
    # generate depressions and solve subfacet conduction
    solver = TemperatureSolverFactory.create(cfg.temp_solver)
    for f in shape:
        f.generate_spherical_depression(cfg, sim)
        solver.initialize_temperatures(f.depression_thermal_data, sim, cfg)
        f.depression_temperature_result = solver.solve(
            f.depression_thermal_data,
            f.sub_facets,
            sim,
            cfg
        )

    # pick our facet
    parent = shape[facet_index]
    subs  = parent.depression_temperature_result["final_day_temperatures"]  # (N_sub x T)
    N, T = subs.shape

    # rebuild world-space triangles for subfacets
    mesh_entries = Facet._canonical_subfacet_mesh
    parent_radius = np.sqrt(parent.area/np.pi)
    scale = cfg.kernel_dome_radius_factor * parent_radius
    verts, faces = [], []
    vidx = 0
    for j, entry in enumerate(mesh_entries):
        # local→world
        tri = entry["vertices"] * scale
        world_tri = (parent.dome_rotation.dot(tri.T)).T + parent.position
        verts.extend(world_tri)
        faces.extend([3, vidx, vidx+1, vidx+2])
        vidx += 3
    verts = np.array(verts)
    faces = np.array(faces)

    # call the same animate_model routine
    # (we monkey-patch it by writing our mesh to a temporary STL file)
    tmp = Path("subfacet_temp.stl")
    from stl import mesh as stl_mesh
    m = stl_mesh.Mesh(np.zeros(N, dtype=stl_mesh.Mesh.dtype))
    for i in range(N):
        m.vectors[i] = verts[3*i:3*i+3]
    m.save(str(tmp))

    animate_model(
        str(tmp),                     # path_to_shape_model_file
        subs,                         # plotted_variable_array
        sim.rotation_axis,
        sim.sunlight_direction,
        sim.timesteps_per_day,
        sim.solar_distance_au,
        sim.rotation_period_hours,
        colour_map="coolwarm",
        plot_title=f"Sub-facet Temps for facet {facet_index}",
        axis_label="Temperature (K)",
        animation_frames=T,
        save_animation=False,
        save_animation_name=f"facet_{facet_index}_subfacets.gif",
        background_colour="white",
        animation_debug_mode=True
    )

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1]  if len(sys.argv)>1 else "data/config/example_config.yaml"
    idx = int(sys.argv[2]) if len(sys.argv)>2 else 37
    main(cfg, idx)