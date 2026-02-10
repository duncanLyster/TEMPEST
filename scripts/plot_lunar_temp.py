#!/usr/bin/env python3
"""
Quick script to plot bolometric temperature vs local time for a specific location.
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# Target coordinates
target_lat = -3.0
target_lon = 336.6  # Convert to -180 to 180 range
target_lon_180 = target_lon - 360 if target_lon > 180 else target_lon

# Box size (2 degree by 2 degree)
lat_min = target_lat - 1.0
lat_max = target_lat + 1.0
lon_min = target_lon_180 - 1.0
lon_max = target_lon_180 + 1.0

# Read the data file
data_file = 'global_cumul_avg_cyl_10s00s_002.tab.txt'
print(f"Loading data from {data_file}...")
print(f"Target: lat={target_lat}±1.0, lon={target_lon} ({target_lon_180:.2f}±1.0 in -180/180)")

# Extract data for the 2x2 degree box using awk
print("Extracting data for 2x2 degree box...")
awk_extract = f"""awk -F',' 'NR>1 {{
    lon=$1+0; lat=$2+0; tbol=$11+0;
    if (lon >= {lon_min} && lon <= {lon_max} && lat >= {lat_min} && lat <= {lat_max} && tbol != -9999) {{
        print $3, $11
    }}
}}' {data_file}"""

result = subprocess.run(awk_extract, shell=True, capture_output=True, text=True)

# Parse the output
lines = result.stdout.strip().split('\n')
if not lines or not lines[0].strip():
    print("No data found!")
    exit(1)

data_pairs = [line.split() for line in lines if line.strip()]
ltim_array = np.array([float(p[0]) for p in data_pairs])
tbol_array = np.array([float(p[1]) for p in data_pairs])

print(f"Found {len(ltim_array)} valid data points")

# Group by local time and calculate mean and std dev
# Round local time to nearest 0.1 hour for binning
ltim_binned = np.round(ltim_array * 10) / 10
unique_times = np.unique(ltim_binned)

ltim_mean = []
tbol_mean = []
tbol_std = []

for t in unique_times:
    mask = ltim_binned == t
    tbol_values = tbol_array[mask]
    ltim_mean.append(t)
    tbol_mean.append(np.mean(tbol_values))
    tbol_std.append(np.std(tbol_values))

ltim_mean = np.array(ltim_mean)
tbol_mean = np.array(tbol_mean)
tbol_std = np.array(tbol_std)

# Sort by local time
sort_idx = np.argsort(ltim_mean)
ltim_sorted = ltim_mean[sort_idx]
tbol_mean_sorted = tbol_mean[sort_idx]
tbol_std_sorted = tbol_std[sort_idx]

print(f"Binned into {len(ltim_sorted)} time bins")

# Load TEMPEST simulation data
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'user_saved_data',
                         '1D_10-02-2026_07-24-19', 'temperature_(k)_vs_timestep_072419.csv')
csv_path = os.path.normpath(csv_path)
ltim_csv, temp_csv = np.loadtxt(csv_path, delimiter=',', skiprows=1, unpack=True)

print(f"Loaded {len(ltim_csv)} points from TEMPEST simulation")

# Plot with error bars (no line)
plt.figure(figsize=(10, 6))
plt.errorbar(ltim_sorted, tbol_mean_sorted, yerr=tbol_std_sorted,
             fmt='o', markersize=5, capsize=3, capthick=1.5, elinewidth=1.5,
             label='Diviner observed')
plt.plot(ltim_csv, temp_csv, 'b-', linewidth=1.5, alpha=0.8, label='TEMPEST simulation')
plt.legend()
plt.xlabel('Local Time', fontsize=12)
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 3))
plt.ylabel('Bolometric Temperature (K)', fontsize=12)
plt.title(f'Bolometric Temperature vs Local Time\nLat={target_lat}±1.0°, Lon={target_lon:.1f}° ({target_lon_180:.2f}±1.0°)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
output_file = 'lunar_temp_plot.png'
plt.savefig(output_file, dpi=150)
print(f"Plot saved to {output_file}")

plt.show()
