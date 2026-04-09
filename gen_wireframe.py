#!/usr/bin/env python3
"""
Generate a clean wireframe visualization of a 10,000 facet crater.
"""
import sys
import os
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.model.spherical_cap_mesh import generate_canonical_spherical_cap

def create_crater_wireframe(n_facets=10000, output_file="crater_wireframe_10k.png"):
    """
    Create a wireframe visualization of a crater mesh.
    
    Parameters
    ----------
    n_facets : int
        Number of facets to generate
    output_file : str
        Output PNG filename
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    print(f"Generating {n_facets}-facet crater mesh...")
    mesh = generate_canonical_spherical_cap(n_facets, 90.0)  # 90° = hemisphere
    
    print(f"Creating wireframe visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot wireframe edges
    vertices_all = []
    for entry in mesh:
        v = entry['vertices']  # (3, 3) array of triangle vertices
        vertices_all.append(v)
        
        # Plot the three edges of the triangle
        v_loop = np.vstack([v, v[0]])  # Close the triangle
        ax.plot(v_loop[:, 0], v_loop[:, 1], v_loop[:, 2], 
                color='black', linewidth=0.3, alpha=0.7)
    
    # Calculate bounds for equal scaling
    vertices_flat = np.vstack(vertices_all).reshape(-1, 3)
    max_range = np.array([
        vertices_flat[:, 0].max() - vertices_flat[:, 0].min(),
        vertices_flat[:, 1].max() - vertices_flat[:, 1].min(),
        vertices_flat[:, 2].max() - vertices_flat[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (vertices_flat[:, 0].max() + vertices_flat[:, 0].min()) * 0.5
    mid_y = (vertices_flat[:, 1].max() + vertices_flat[:, 1].min()) * 0.5
    mid_z = (vertices_flat[:, 2].max() + vertices_flat[:, 2].min()) * 0.5
    
    # Set equal aspect ratio
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Completely hide axes and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Remove axis lines
    ax.xaxis.line.set_linewidth(0)
    ax.yaxis.line.set_linewidth(0)
    ax.zaxis.line.set_linewidth(0)
    
    # Hide labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title('')
    
    # Hide the panes and spines completely
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_linewidth(0)
    ax.yaxis.pane.set_linewidth(0)
    ax.zaxis.pane.set_linewidth(0)
    
    # Remove grid
    ax.grid(False)
    
    # Set white background  
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')
    
    # Set viewing angle for nice 3D perspective (looking down from above)
    ax.view_init(elev=32, azim=45)
    
    # Save with tight layout and no padding
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    plt.savefig(output_file, dpi=1000, bbox_inches='tight', pad_inches=0.0, 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Wireframe saved to: {output_file}")
    print(f"  Total facets: {len(mesh)}")

if __name__ == "__main__":
    # Generate 10,000 facet crater wireframe
    create_crater_wireframe(n_facets=10000, output_file="crater_wireframe_10k.png")
