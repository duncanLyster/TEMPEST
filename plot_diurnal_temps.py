#!/usr/bin/env python3
"""
Plot diurnal temperature curves from CSV files.
Handles files with naming convention: TI_<value>_a_<value>.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import sys

def extract_ti_albedo(filename):
    """Extract thermal inertia and albedo from filename."""
    pattern = r'TI_(\d+(?:\.\d+)?)_a_(\d+(?:\.\d+)?)'
    match = re.search(pattern, filename)
    if match:
        ti = float(match.group(1))
        albedo = float(match.group(2))
        return ti, albedo
    return None, None

def plot_temperature_file(filepath, output_dir=None):
    """Plot a single temperature CSV file."""
    filename = Path(filepath).name
    ti, albedo = extract_ti_albedo(filename)
    
    if ti is None:
        print(f"Could not extract TI and albedo from {filename}")
        return
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4.2))
    
    # Plot the three facets
    ax.plot(df['Local Time (hours)'], df['Facet_987'], 
            linewidth=2, label='Equatorial latitude (facet 987)')
    ax.plot(df['Local Time (hours)'], df['Facet_240'], 
            linewidth=2, label='Mid latitude (facet 240)')
    ax.plot(df['Local Time (hours)'], df['Facet_15'], 
            linewidth=2, label='Polar latitude (facet 15)')
    
    # Set x-axis from 0 to 24 with markers every 6 hours
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.set_xlabel('Local Time (hours)', fontsize=12)
    
    # Set y-axis label and limits
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_ylim(120, 280)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Add text box with TI and albedo in top right
    textstr = f'TI = {ti:.0f} J·m$^{{-2}}$·K$^{{-1}}$·s$^{{-1/2}}$\n$\\alpha$ = {albedo:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir is None:
        output_dir = Path(filepath).parent
    output_path = Path(output_dir) / f"temp_plot_{ti:.0f}_{albedo:.3f}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()

def main():
    """Main function to process all temperature files."""
    if len(sys.argv) > 1:
        # Process specified file
        filepath = sys.argv[1]
        plot_temperature_file(filepath)
    else:
        # Process all TI_*_a_*.csv files in current directory and subdirectories
        current_dir = Path.cwd()
        csv_files = list(current_dir.glob('**/TI_*_a_*.csv'))
        
        if not csv_files:
            print(f"No TI_*_a_*.csv files found in {current_dir}")
            return
        
        print(f"Found {len(csv_files)} temperature files")
        for filepath in sorted(csv_files):
            print(f"\nProcessing: {filepath.name}")
            plot_temperature_file(filepath)

if __name__ == '__main__':
    main()
