#!/usr/bin/env python3
"""
Generate publication-quality temperature vs. time plots for different thermal inertia (TI) values.
Plots are saved as 150 DPI PNG files suitable for scientific publications.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Color definitions - sophisticated, publication-quality colors
COLORS = {
    'orange': [0.85, 0.45, 0.15],   # Warm rust/burnt orange (Facet 70 - equatorial)
    'blue': [0.25, 0.55, 0.85],     # Steel blue (Facet 127 - mid-latitude)
    'green': [0.2, 0.65, 0.35]      # Muted forest green (Facet 527 - polar)
}

FACET_LABELS = {
    70: {'name': 'equatorial latitude', 'color': 'orange'},
    127: {'name': 'mid latitude', 'color': 'blue'},
    527: {'name': 'polar latitude', 'color': 'green'}
}

# Albedo values for each thermal inertia
ALBEDO_MAP = {
    200: 0.30,
    300: 0.045,
    2000: 0.10
}

# Mapping from folder TI value to display TI value (in case they differ)
TI_DISPLAY_MAP = {
    1000: 2000  # Folder named TI_1000 should display as TI_2000
}

def extract_ti_value(folder_path):
    """Extract thermal inertia (TI) value from folder name (e.g., 'TI_200' -> 200)."""
    folder_name = os.path.basename(folder_path)
    if folder_name.startswith('TI_'):
        try:
            return int(folder_name.split('_')[1])
        except (ValueError, IndexError):
            return None
    return None


def find_temperature_csv(ti_folder):
    """Find the temperature CSV file in a TI folder."""
    # Look for CSV files in subdirectories
    for root, dirs, files in os.walk(ti_folder):
        for file in files:
            if file.startswith('temperature_k_vs_timestep_') and file.endswith('.csv'):
                return os.path.join(root, file)
    return None


def get_all_temperature_ranges(ti_folders):
    """
    Calculate the min and max temperatures across all TI folders to ensure consistent y-axis scaling.
    
    Parameters:
    -----------
    ti_folders : list
        List of TI folder paths
    
    Returns:
    --------
    tuple
        (min_temp, max_temp) across all data
    """
    all_temps = []
    for ti_folder in ti_folders:
        csv_path = find_temperature_csv(ti_folder)
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                all_temps.extend(df['Facet_127'].values)
                all_temps.extend(df['Facet_70'].values)
                all_temps.extend(df['Facet_527'].values)
            except:
                pass
    
    if not all_temps:
        return None, None
    
    min_temp = np.min(all_temps)
    max_temp = np.max(all_temps)
    
    # Add some margin for readability
    margin = (max_temp - min_temp) * 0.05
    return min_temp - margin, max_temp + margin


def create_ti_plot(csv_path, ti_value, output_dir, y_min, y_max):
    """
    Create a publication-quality temperature vs. time plot for a given TI value.
    
    Parameters:
    -----------
    csv_path : str
        Path to the temperature CSV file
    ti_value : int
        Thermal inertia value (folder-based)
    output_dir : str
        Output directory for the PNG file
    y_min : float
        Minimum y-axis value (for consistent scaling across plots)
    y_max : float
        Maximum y-axis value (for consistent scaling across plots)
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Get display TI value (may differ from folder TI value)
        display_ti_value = TI_DISPLAY_MAP.get(ti_value, ti_value)
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Extract time and temperature data
        time = df['Local Time (hours)'].values
        temp_127 = df['Facet_127'].values  # Mid-latitude
        temp_70 = df['Facet_70'].values     # Equatorial
        temp_527 = df['Facet_527'].values   # Polar
        
        # Clip to 24 hours
        max_time = 24.0
        mask = time <= max_time
        time = time[mask]
        temp_127 = temp_127[mask]
        temp_70 = temp_70[mask]
        temp_527 = temp_527[mask]
        
        # Create figure with high DPI for publication
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        # Plot the three temperature curves with specified colors
        ax.plot(time, temp_70, color=COLORS['orange'], linewidth=2.5, 
                label='Equatorial latitude (facet 70)')
        ax.plot(time, temp_127, color=COLORS['blue'], linewidth=2.5, 
                label='Mid latitude (facet 127)')
        ax.plot(time, temp_527, color=COLORS['green'], linewidth=2.5, 
                label='Polar latitude (facet 527)')
        
        # Set labels and title
        ax.set_xlabel('Local time (hours)', fontsize=12)
        ax.set_ylabel('Temperature (K)', fontsize=12)
        
        # Create TI and Albedo label with proper formatting
        # Using Unicode characters for superscript and Greek letter alpha
        albedo = ALBEDO_MAP.get(display_ti_value, 'N/A')
        ti_label = f'TI = {display_ti_value} J·m⁻²·K⁻¹·s⁻¹ᐟ²\nα = {albedo}'
        ax.text(0.98, 0.97, ti_label, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set axis limits
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(y_min, y_max)
        
        # Add legend at mid-point to avoid overlapping with data
        ax.legend(loc='center left', fontsize=11, framealpha=0.95)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Tight layout
        plt.tight_layout()
        
        # Create output filename (use display TI value)
        output_filename = f'temperature_vs_time_TI_{display_ti_value}.png'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save with 150 DPI
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Created: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing TI_{ti_value}: {e}")
        return False


def main(data_folder=None):
    """
    Main function to process all TI folders and create plots.
    
    Parameters:
    -----------
    data_folder : str, optional
        Path to folder containing TI_* subdirectories. 
        If None, uses /Users/duncan/Desktop/Asteroid 436724
    """
    if data_folder is None:
        data_folder = '/Users/duncan/Desktop/Asteroid 436724'
    
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found: {data_folder}")
        return False
    
    # Create output directory
    output_dir = os.path.join(data_folder, 'temperature_plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Find all TI folders
    ti_folders = []
    for item in os.listdir(data_folder):
        item_path = os.path.join(data_folder, item)
        if os.path.isdir(item_path) and item.startswith('TI_'):
            ti_folders.append(item_path)
    
    if not ti_folders:
        print("No TI_* folders found in data directory")
        return False
    
    # Sort by TI value numerically
    ti_folders.sort(key=lambda x: extract_ti_value(x) or 0)
    
    print(f"Found {len(ti_folders)} TI folder(s):\n")
    
    # Calculate temperature range across all folders for consistent y-axis scaling
    y_min, y_max = get_all_temperature_ranges(ti_folders)
    if y_min is None or y_max is None:
        print("Error: Could not determine temperature range from data")
        return False
    
    print(f"Temperature range across all data: {y_min:.1f} - {y_max:.1f} K\n")
    
    success_count = 0
    for ti_folder in ti_folders:
        ti_value = extract_ti_value(ti_folder)
        csv_path = find_temperature_csv(ti_folder)
        
        if csv_path is None:
            print(f"✗ No temperature CSV found in {os.path.basename(ti_folder)}")
            continue
        
        if create_ti_plot(csv_path, ti_value, output_dir, y_min, y_max):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Successfully created {success_count}/{len(ti_folders)} plot(s)")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")
    
    return success_count == len(ti_folders)


if __name__ == '__main__':
    # Accept optional folder path as command-line argument
    data_folder = sys.argv[1] if len(sys.argv) > 1 else None
    success = main(data_folder)
    sys.exit(0 if success else 1)
