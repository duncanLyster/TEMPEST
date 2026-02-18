
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def print_stats(data_subset, label, theta_val):
    clean = data_subset[~np.isnan(data_subset)]
    if len(clean) == 0:
        print(f"No valid data for {label}")
        return
        
    print(f"\n--- Stats for {label} (Theta={theta_val:.4f}) ---")
    print(f"Mean: {np.mean(clean):.4f}")
    print(f"Median: {np.median(clean):.4f}")
    print(f"Max: {np.max(clean):.4f}")
    print(f"Min: {np.min(clean):.4f}")
    
    gt_1 = np.sum(clean > 1.0)
    lt_1 = np.sum(clean < 1.0)
    total = len(clean)
    
    print(f"Count > 1.0 (Beaming?): {gt_1} ({100*gt_1/total:.2f}%)")
    print(f"Count < 1.0 (Cooling/Shadow?): {lt_1} ({100*lt_1/total:.2f}%)")
    print("------------------------")

def plot_polar_slice(data_block, theta_val, label):
    # data_block shape: (Inc, Lat, Time, Wave, Emission, Azimuth)
    
    # We want a slice at Noon (Phase=0 approx).
    # Noon corresponds to min(Incidence)
    # Let's find index where Incidence is closest to 0.
    # But wait, Time defines Phase.
    
    # To simplify, let's find the maximum value in the block (the "Hot Spot")
    # and plot the slice through that point in Emission/Azimuth space.
    
    flat = data_block.flatten()
    if np.all(np.isnan(flat)):
        return

    # Find the indices of the maximum value (Beaming Peak)
    # Using nanargmax to ignore NaNs
    max_idx_flat = np.nanargmax(flat)
    max_idx = np.unravel_index(max_idx_flat, data_block.shape)
    
    # max_idx = (Inc, Lat, Time, Wave, Em, Az)
    # Fix the first 4 dimensions
    inc_idx, lat_idx, time_idx, wave_idx = max_idx[:4]
    
    # Extract 2D slice: (Emission, Azimuth)
    # Emission is dim 4, Azimuth is dim 5
    slice_2d = data_block[inc_idx, lat_idx, time_idx, wave_idx, :, :]
    
    n_em, n_az = slice_2d.shape
    
    # Construct angular grids
    # Emission: Usually 0 to 90
    em_rad = np.linspace(0, 90, n_em)
    
    # Azimuth: Usually 0 to 180 (Assumed from generator.py)
    # We mirror to 0..360 for polar plot.
    
    az_deg = np.linspace(0, 180, n_az)
    # Mirror: [0, ..., 180, ..., 360]
    # We concatenate 0..180 with reverse(1..179) mirrored against 360.
    
    az_mirror = 360 - az_deg[1:-1][::-1]
    az_full = np.concatenate([az_deg, az_mirror])
    
    # Convert to radians for polar plot
    az_rad = np.radians(az_full)
    
    # Mirror Data
    # slice_2d columns correspond to azimuths.
    # We need to mirror columns
    slice_mirror_data = slice_2d[:, 1:-1][:, ::-1]
    slice_full = np.concatenate([slice_2d, slice_mirror_data], axis=1)
    
    # Meshgrid must match dimensions
    # R (Emission) is Y-axis, Theta (Azimuth) is X-axis (Angle)
    # pcolormesh(Theta, R, Values)
    # Theta: (n_az_full,), R: (n_em,)
    # Values: (n_em, n_az_full)
    
    Theta_grid, R_grid = np.meshgrid(az_rad, em_rad)
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # Rotate so 0 is up (Sun direction usually)
    ax.set_theta_zero_location("N")
    
    mesh = ax.pcolormesh(Theta_grid, R_grid, slice_full, cmap='inferno', shading='auto')
    plt.colorbar(mesh, label='Ratio')
    
    ax.set_title(f'Polar BRDF Slice via Hot Spot\nTheta={theta_val:.1f} ({label})\nInc={inc_idx}, Lat={lat_idx}, Time={time_idx}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 90) 
    
    out_name = f'lut_polar_theta_{theta_val:.1f}.png'
    plt.savefig(out_name)
    print(f"Saved polar plot to {out_name}")
    plt.close()


def generate_plots():
    filename = 'roughness_lut_spectral_v1.h5'
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    print(f"Opening {filename}...")
    with h5py.File(filename, 'r') as f:
        lut = f['lut'][:]
        theta_vals = f['theta'][:]
        
        # Indices for Theta ~ 1.0 and Theta ~ 20.0
        # Check available theta values first
        print(f"Available Theta values: {theta_vals}")
        
        # Use close approximation
        if np.any(np.abs(theta_vals - 1.0) < 0.1):
             idx_low = np.argmin(np.abs(theta_vals - 1.0))
        else:
             idx_low = 0 # Default to first
             
        if np.any(np.abs(theta_vals - 20.0) < 1.0):
             idx_high = np.argmin(np.abs(theta_vals - 20.0))
        else:
             idx_high = -1 # Default to last
        
        indices = list(set([idx_low, idx_high])) # Unique
        indices.sort()
        
        for idx in indices:
            theta_val = theta_vals[idx]
            label = "Low Inertia" if theta_val < 5 else "High Inertia"
            
            # Slice for this Theta
            data_block = lut[idx, ...]
            
            # 1. Stats & Histogram
            data_flat = data_block.flatten()
            clean_data = data_flat[~np.isnan(data_flat)]
            
            print_stats(clean_data, label, theta_val)
            
            plt.figure(figsize=(10, 6))
            plot_data = clean_data[(clean_data > 0.1) & (clean_data < 5.0)]
            plt.hist(plot_data, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Smooth (1.0)')
            plt.title(f'Roughness LUT Histogram (Log Scale)\nTheta={theta_val:.2f} ({label})')
            
            plt.xlabel('Radiance Ratio (Rough / Smooth)')
            plt.ylabel('Count')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            hist_name = f'lut_hist_theta_{theta_val:.1f}.png'
            plt.savefig(hist_name)
            print(f"Saved histogram to {hist_name}")
            plt.close()
            
            # 2. Polar Plot
            plot_polar_slice(data_block, theta_val, label)

if __name__ == "__main__":
    generate_plots()
