#!/usr/bin/env python3
"""
Calculate the mean number of visible facets for each asteroid model.
Uses the existing TEMPEST infrastructure to load/compute visible facets.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Ensure src directory is in the Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

from src.utilities.config import Config
from src.utilities.locations import Locations
from src.model.simulation import Simulation, ThermalData
from src.model.facet import Facet
from src.model.view_factors import calculate_and_cache_visible_facets
from src.utilities.utils import conditional_print
from stl import mesh as stl_mesh_module


def read_shape_model(filename, timesteps_per_day=1, n_layers=10, max_days=1, calculate_energy_terms=False):
    """Read shape model from STL file."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} was not found.")
    
    try:
        # Try reading as ASCII first
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        shape_model = []
        for i in range(len(lines)):
            if lines[i].strip().startswith('facet normal'):
                normal = np.array([float(n) for n in lines[i].strip().split()[2:]])
                vertex1 = np.array([float(v) for v in lines[i+2].strip().split()[1:]])
                vertex2 = np.array([float(v) for v in lines[i+3].strip().split()[1:]])
                vertex3 = np.array([float(v) for v in lines[i+4].strip().split()[1:]])
                facet = Facet(normal, [vertex1, vertex2, vertex3], timesteps_per_day, max_days, n_layers, calculate_energy_terms)
                shape_model.append(facet)
    except (UnicodeDecodeError, IndexError):
        # If ASCII reading fails, try binary
        shape_mesh = stl_mesh_module.Mesh.from_file(filename)
        shape_model = []
        for i in range(len(shape_mesh.vectors)):
            normal = shape_mesh.normals[i]
            vertices = shape_mesh.vectors[i]
            facet = Facet(normal, vertices, timesteps_per_day, max_days, n_layers, calculate_energy_terms)
            shape_model.append(facet)
    
    return shape_model


def calculate_visible_facet_statistics(config_path, model_name):
    """Calculate visible facet statistics for a given model."""
    print(f"\n{'='*70}")
    print(f"Calculating statistics for {model_name}")
    print(f"{'='*70}")
    
    # Load configuration
    config = Config(config_path)
    config.silent_mode = False
    
    # Read shape model
    print(f"Reading shape model from: {config.path_to_shape_model_file}")
    shape_model = read_shape_model(config.path_to_shape_model_file)
    n_facets = len(shape_model)
    print(f"Number of facets: {n_facets:,}")
    
    # Extract positions, normals, and vertices for visible facet calculation
    positions = np.array([facet.position for facet in shape_model])
    normals = np.array([facet.normal for facet in shape_model])
    vertices = np.array([facet.vertices for facet in shape_model])
    
    # Calculate or load visible facets
    print(f"Loading/calculating visible facets...")
    visible_indices = calculate_and_cache_visible_facets(
        config.silent_mode, 
        shape_model, 
        positions, 
        normals, 
        vertices, 
        config
    )
    
    # Calculate statistics
    total_visible_pairs = 0
    visible_counts = []
    
    for i, visible_facets in enumerate(visible_indices):
        n_visible = len(visible_facets)
        visible_counts.append(n_visible)
        total_visible_pairs += n_visible
    
    mean_visible_facets = total_visible_pairs / n_facets
    min_visible_facets = np.min(visible_counts)
    max_visible_facets = np.max(visible_counts)
    std_visible_facets = np.std(visible_counts)
    
    print(f"\nResults for {model_name}:")
    print(f"  Total facet pairs: {total_visible_pairs:,}")
    print(f"  Number of facets: {n_facets:,}")
    print(f"  Mean visible facets per facet: {mean_visible_facets:.2f}")
    print(f"  Min visible facets: {min_visible_facets}")
    print(f"  Max visible facets: {max_visible_facets}")
    print(f"  Std dev: {std_visible_facets:.2f}")
    
    return mean_visible_facets, total_visible_pairs, n_facets


if __name__ == "__main__":
    # We'll manually specify the model files since the configs use different resolutions
    from src.utilities.locations import Locations
    
    # Create temp configs for Bennu 3184 facets
    print("Setting up Bennu (3,184 facets) analysis...")
    bennu_model_path = "private/data/shape_models/Bennu/Bennu_3184_facets.stl"
    
    if os.path.exists(bennu_model_path):
        print(f"Found Bennu model: {bennu_model_path}")
        bennu_shape_model = read_shape_model(bennu_model_path)
        bennu_n = len(bennu_shape_model)
        print(f"Number of facets: {bennu_n:,}")
        
        # Extract positions, normals, and vertices
        positions = np.array([facet.position for facet in bennu_shape_model])
        normals = np.array([facet.normal for facet in bennu_shape_model])
        vertices = np.array([facet.vertices for facet in bennu_shape_model])
        
        # Calculate visible facets
        from src.model.view_factors import calculate_and_cache_visible_facets
        from src.utilities.config import Config
        config = Config("private/data/config/bennu/bennu_config.yaml")
        config.silent_mode = False
        
        print("Loading/calculating visible facets for Bennu...")
        bennu_visible = calculate_and_cache_visible_facets(
            config.silent_mode,
            bennu_shape_model,
            positions,
            normals,
            vertices,
            config
        )
        
        bennu_pairs = sum(len(v) for v in bennu_visible)
        bennu_mean = bennu_pairs / bennu_n
        print(f"Bennu results: {bennu_mean:.2f} mean visible facets ({bennu_pairs:,} total pairs)")
    else:
        print(f"ERROR: Bennu model not found at {bennu_model_path}")
        sys.exit(1)
    
    # Create configs for Itokawa 3144 facets  
    print("\nSetting up Itokawa (3,144 facets) analysis...")
    itokawa_model_path = "private/data/shape_models/Itokawa/Itokawa_3144_facets.stl"
    
    if os.path.exists(itokawa_model_path):
        print(f"Found Itokawa model: {itokawa_model_path}")
        itokawa_shape_model = read_shape_model(itokawa_model_path)
        itokawa_n = len(itokawa_shape_model)
        print(f"Number of facets: {itokawa_n:,}")
        
        # Extract positions, normals, and vertices
        positions = np.array([facet.position for facet in itokawa_shape_model])
        normals = np.array([facet.normal for facet in itokawa_shape_model])
        vertices = np.array([facet.vertices for facet in itokawa_shape_model])
        
        # Calculate visible facets
        config = Config("private/data/config/itokawa/itokawa_config.yaml")
        config.silent_mode = False
        
        print("Loading/calculating visible facets for Itokawa...")
        itokawa_visible = calculate_and_cache_visible_facets(
            config.silent_mode,
            itokawa_shape_model,
            positions,
            normals,
            vertices,
            config
        )
        
        itokawa_pairs = sum(len(v) for v in itokawa_visible)
        itokawa_mean = itokawa_pairs / itokawa_n
        print(f"Itokawa results: {itokawa_mean:.2f} mean visible facets ({itokawa_pairs:,} total pairs)")
    else:
        print(f"ERROR: Itokawa model not found at {itokawa_model_path}")
        sys.exit(1)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nFor your sentence:")
    print(f"  X (Bennu, {bennu_n:,} facets): {bennu_mean:.2f}")
    print(f"  Y (Itokawa, {itokawa_n:,} facets): {itokawa_mean:.2f}")
    print(f"\nComplete sentence:")
    print(f"Bennu and Itokawa were chosen to illustrate the impact of concavity.")
    print(f"Bennu, being a primarily convex shape runs substantially faster than")
    print(f"Itokawa does at the same resolution (number of facets). The mean number")
    print(f"of facets visible to each other facet is {bennu_mean:.2f} for the")
    print(f"{bennu_n:,} facet model of Bennu, and {itokawa_mean:.2f} for the")
    print(f"{itokawa_n:,} facet model of Itokawa and the greatest difference in")
    print(f"computation time was seen in the scattering and thermal solver processes.")
