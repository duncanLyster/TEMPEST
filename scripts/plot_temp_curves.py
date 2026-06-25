#!/usr/bin/env python3
"""
Plot all temperature curves saved in the temperature_curves directory.

This script reads all CSV files from the temperature_curves/ directory and plots
them together for comparison. The filename format is:
    facet_{facet_idx}_TI{thermal_inertia}_alb{albedo}_heat{heating_flux}_{timestamp}.csv
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_filename(filename):
    """
    Parse filename to extract parameters.
    
    Format: facet_{facet_idx}_TI{thermal_inertia}_alb{albedo}_heat{heating_flux}_{timestamp}.csv
    
    Returns:
        dict with keys: facet_idx, thermal_inertia, albedo, heating_flux, timestamp
    """
    basename = os.path.basename(filename)
    
    # Extract parameters using regex
    pattern = r'facet_(\d+)_TI([\d.]+)_alb([\d.]+)_heat([\d.]+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv'
    match = re.match(pattern, basename)
    
    if match:
        return {
            'facet_idx': int(match.group(1)),
            'thermal_inertia': float(match.group(2)),
            'albedo': float(match.group(3)),
            'heating_flux': float(match.group(4)),
            'timestamp': match.group(5),
            'filename': basename
        }
    else:
        # Fallback: try to extract what we can
        return {
            'facet_idx': None,
            'thermal_inertia': None,
            'albedo': None,
            'heating_flux': None,
            'timestamp': None,
            'filename': basename
        }

def get_curve_style(params, curve_config, debug=False):
    """
    Get color, label, and linestyle for a curve based on configuration.
    
    Parameters:
    -----------
    params : dict
        Curve parameters (thermal_inertia, heating_flux, albedo)
    curve_config : dict
        Configuration dictionary mapping (ti, heat, albedo) tuples to style dicts
    debug : bool
        If True, print debug information about matching
        
    Returns:
    --------
    dict with keys: 'color', 'label', 'linestyle'
    """
    ti = params['thermal_inertia']
    heat = params['heating_flux']
    albedo = params['albedo']
    
    if debug:
        print(f"\nDebug: Looking for match for TI={ti}, heat={heat}, albedo={albedo}")
    
    # Try to find matching configuration
    # Check exact matches first, then matches with None values
    for (config_ti, config_heat, config_albedo), style in curve_config.items():
        if debug:
            print(f"  Checking config: TI={config_ti}, heat={config_heat}, albedo={config_albedo}")
        
        # Match thermal_inertia (use small tolerance for float comparison)
        if config_ti is not None:
            if ti is None or abs(ti - config_ti) > 1e-6:
                if debug:
                    print(f"    TI mismatch: {ti} != {config_ti}")
                continue
        
        # Match heating_flux
        if config_heat is not None:
            if config_heat == '>0':
                # Match any positive heating flux
                if heat is None or heat <= 0.0:
                    if debug:
                        print(f"    Heat >0 mismatch: {heat} is not > 0")
                    continue
            elif config_heat == 0.0:
                if heat is None or abs(heat - 0.0) > 1e-6:
                    if debug:
                        print(f"    Heat=0 mismatch: {heat} != 0.0")
                    continue
            else:
                # Exact match for numeric value
                if heat is None or abs(heat - config_heat) > 1e-6:
                    if debug:
                        print(f"    Heat mismatch: {heat} != {config_heat}")
                    continue
        
        # Match albedo (if specified)
        if config_albedo is not None:
            if albedo is None or abs(albedo - config_albedo) > 1e-6:
                if debug:
                    print(f"    Albedo mismatch: {albedo} != {config_albedo}")
                continue
        
        # Found a match
        if debug:
            print(f"  ✓ MATCH FOUND!")
        return style.copy()
    
    # No match found - return default style
    if debug:
        print(f"  ✗ No match found, using default style")
    return {
        'color': None,  # Will use default colormap
        'label': f"TI={ti} Jm⁻²K⁻¹s⁻½, \nheat={heat} W/m², alb={albedo}" if ti is not None else params['filename'],
        'linestyle': '-' if (heat is not None and heat > 0.0) else '--'
    }

def plot_all_temp_curves(directory='temperature_curves', facet_idx=None, debug=False):
    """
    Plot all temperature curves from CSV files in the specified directory.
    
    Parameters:
    -----------
    directory : str
        Directory containing the CSV files (default: 'temperature_curves')
    facet_idx : int, optional
        If specified, only plot curves for this facet index
    """
    # Get the script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Construct full path to temperature_curves directory
    temp_curves_dir = project_root / directory
    
    if not temp_curves_dir.exists():
        print(f"Error: Directory '{temp_curves_dir}' does not exist.")
        print("Make sure you've run the model with save_temp_curve_data: true")
        return
    
    # Find all CSV files
    csv_files = list(temp_curves_dir.glob('*.csv'))
    
    if len(csv_files) == 0:
        print(f"No CSV files found in '{temp_curves_dir}'")
        return
    
    print(f"Found {len(csv_files)} temperature curve file(s)")
    
    # Parse and load all files
    curves = []
    for csv_file in sorted(csv_files):
        params = parse_filename(str(csv_file))
        
        # Filter by facet index if specified
        if facet_idx is not None and params['facet_idx'] != facet_idx:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            params['data'] = df
            curves.append(params)
            print(f"  Loaded: {params['filename']}")
            print(f"    → TI={params['thermal_inertia']}, heat={params['heating_flux']}, albedo={params['albedo']}")
        except Exception as e:
            print(f"  Warning: Could not load {csv_file}: {e}")
    
    if len(curves) == 0:
        print("No valid curves to plot.")
        return
    
    # ============================================================================
    # MANUAL CONFIGURATION: Customize colors, labels, and linestyles here
    # ============================================================================
    # Format: (thermal_inertia, heating_flux, albedo): {
    #     'color': '#hexcolor' or matplotlib color name,
    #     'label': 'Legend label text',
    #     'linestyle': '-' (solid), '--' (dashed), '-.' (dashdot), ':' (dotted)
    # }
    # Note: Use None for albedo to match any albedo value
    #       Use '>0' as string for heating_flux to match any positive value
    #       Order matters: more specific cases should come before general ones
    # ============================================================================
    curve_config = {
        # TI=60 cases
        (40.0, 0.0, None): {  # Light, dense, unheated region
            'color': '#1f77b4',  # Dark blue
            'label': 'Bright, porous, \nunheated region',
            'linestyle': '--'
        },
        (40.0, '>0', None): {  # Dark, porous, conductively heated
            'color': '#ff7f0e',  # Rust red
            'label': 'Bright, porous, \nconductively heated',
            'linestyle': '--'
        },
        # TI=100 cases
        (80.0, 0.0, None): {  # Dark, porous, unheated region
            'color': '#6b9bd4',  # Darker blue
            'label': 'Dark, dense, \nunheated region',
            'linestyle': '-'
        },
    }
    # ============================================================================
    
    # Set serif font
    plt.rcParams['font.family'] = 'serif'
    
    # Create plot
    fig, ax = plt.subplots(figsize=(5, 3))
    
    for i, curve in enumerate(curves):
        df = curve['data']
        
        # Check for required columns
        if 'Rotation (degrees)' not in df.columns or 'Temperature (K)' not in df.columns:
            print(f"Warning: {curve['filename']} missing required columns. Skipping.")
            continue
        
        rotation = df['Rotation (degrees)']
        temperature = df['Temperature (K)']
        
        # Convert rotation degrees to local time (hours)
        # 0-360 degrees corresponds to 0-24 hours
        local_time = (rotation / 360.0) * 24.0
        
        # Get style from configuration
        style = get_curve_style(curve, curve_config, debug=debug)
        
        # Use configured color, or fallback to colormap if None
        if style['color'] is None:
            color = plt.cm.tab10(i / max(len(curves) - 1, 1))
        else:
            color = style['color']
        
        ax.plot(local_time, temperature, label=style['label'], color=color, 
                linewidth=2, alpha=0.8, linestyle=style['linestyle'])
    
    ax.set_xlabel('Local time (hours)', fontsize=18)
    ax.set_ylabel('Temperature (K)', fontsize=18)
    ax.set_title(f'Diurnal Temperature Curves Comparison', fontsize=20)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 3))  # Every 3 hours
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=14)
    ax.tick_params(labelsize=14)
    
    # Add E-THEMIS accuracy indicator (3K vertical line)
    # Position at bottom right
    y_min, y_max = ax.get_ylim()
    x_pos = 21.5  # Bottom right area
    y_bottom = y_min + 0.05 * (y_max - y_min)  # Near bottom, with small margin
    y_top = y_bottom + 2.6  # 3K above the bottom
    
    # Draw vertical line (3K long)
    ax.plot([x_pos, x_pos], [y_bottom, y_top], color='black', 
            linewidth=2, zorder=10)
    
    # Add text label (next to the line)
    y_text = y_bottom + 1.3  # Middle of the line (line is 2.6K tall)
    ax.text(x_pos - 8, y_text, 'E-THEMIS\naccuracy at 90 K (3 K)', 
            fontsize=14, verticalalignment='center', horizontalalignment='left',
            fontfamily='serif',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPlotted {len(curves)} temperature curve(s)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot all temperature curves from temperature_curves directory')
    parser.add_argument('--directory', '-d', type=str, default='temperature_curves',
                        help='Directory containing CSV files (default: temperature_curves)')
    parser.add_argument('--facet', '-f', type=int, default=None,
                        help='Only plot curves for this facet index (default: all facets)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output to see configuration matching')
    
    args = parser.parse_args()
    
    plot_all_temp_curves(directory=args.directory, facet_idx=args.facet, debug=args.debug)
