
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os

def analyze_lut_v2():
    filename = 'roughness_lut_spectral_v1.h5'
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    print(f"Opening {filename}...")
    with h5py.File(filename, 'r') as f:
        lut = f['lut'][:]
        theta_vals = f['theta'][:]
        wave_vals = f['wavelength'][:]
        lat_vals = f['latitude'][:]
        emission_vals = f['emission'][:]
        azimuth_vals = f['azimuth'][:]
        
        # Dimensions: (Theta, Angle, Lat, Time, Wave, Emission, Azimuth) (5, 1, 4, 90, 5, 10, 10)
        
        print("\nLUT Stats:")
        print(f"  Shape: {lut.shape}")
        print(f"  Min: {np.nanmin(lut):.4f}")
        print(f"  Max: {np.nanmax(lut):.4f}")
        print(f"  Mean: {np.nanmean(lut):.4f}")
        
        # 1. Histogram
        data_flat = lut.flatten()
        data_clean = data_flat[~np.isnan(data_flat)]
        plt.figure(figsize=(10, 6))
        plt.hist(data_clean, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Smooth (1.0)')
        plt.title(f'Roughness LUT Histogram (Mean={np.mean(data_clean):.3f})')
        plt.xlabel('Radiance Ratio')
        plt.ylabel('Frequency')
        plt.xlim(0,5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('lut_histogram_v2.png')
        plt.close()
        print("Saved lut_histogram_v2.png")
        
        # Function to plot polar slice
        def plot_polar_slice(slice_data, title, filename, vmin=None, vmax=None):
            # slice_data shape: (Emission, Azimuth)
            # Emission = Radius (0 to 90)
            # Azimuth = Angle (0 to 180)
            
            fig = plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection='polar')
            
            # Create full 360 degree plot by mirroring
            # Right half: 0 to 180
            az_rad = np.radians(azimuth_vals)
            emi_grid, az_grid = np.meshgrid(emission_vals, az_rad)
            
            # Left half: 360 to 180 (Mirror)
            az_rad_left = np.radians(360 - azimuth_vals)
            emi_grid_left, az_grid_left = np.meshgrid(emission_vals, az_rad_left)
            
            # Data: (Emi, Azi) -> Transpose to (Azi, Emi) for pcolormesh
            data_right = slice_data.T
            data_left = slice_data.T # Symmetric
            
            # Determine vmin/vmax if not provided
            if vmin is None: vmin = np.nanmin(slice_data)
            if vmax is None: vmax = np.nanmax(slice_data)
            
            # Center colormap on 1.0 if possible
            if vmin < 1.0 < vmax:
                # Make symmetric around 1.0? Or just divergent
                cmap = 'coolwarm'
                div_norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
                kw = {'norm': div_norm, 'cmap': cmap}
            else:
                cmap = 'viridis'
                kw = {'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
                
            # Plot
            mesh1 = ax.pcolormesh(az_grid, emi_grid, data_right, shading='auto', **kw)
            mesh2 = ax.pcolormesh(az_grid_left, emi_grid_left, data_left, shading='auto', **kw)
            
            ax.set_ylim(0, 90)
            ax.set_yticks([0, 30, 60, 90])
            ax.set_yticklabels(['0', '30', '60', '90'])
            ax.set_theta_zero_location("N") # 0 is North (Sun direction usually implies 0 azimuth?)
            # In Generator: Azimuth 0 is "Towards Sun"? 
            # "Azimuth Angle (View): Relative angle between Sun and Observer (0-180)"
            # Usually Azi=0 means Observer is looking "With the Sun" (Zero Phase).
            
            plt.colorbar(mesh1, ax=ax, label='Radiance Ratio')
            plt.title(title)
            plt.savefig(filename)
            plt.close()
            print(f"Saved {filename} (Min={np.nanmin(slice_data):.3f}, Max={np.nanmax(slice_data):.3f})")

        # Select Indices
        # Theta: 0=1.0, 4=20.0
        # Lat: 0=0, 3=85
        # Wave: 0=5um, 4=100um
        # Time: Max Insolation?
        # At Lat=0, Max Insolation is at angle=0? Or angle=180?
        # Generator: angle = 2*pi*t / 90. Sun vector computed from angle.
        # Sun z = sin(declination) = 0.
        # Sun x = cos(-angle). Sun y = sin(-angle).
        # At t=0 -> angle=0 -> x=1, y=0, z=0.
        # Normal (Lat=0) = (1, 0, 0).
        # Dot = 1. So t=0 is NOON.
        
        # However, indices might be shifted if range is 0..90
        # Let's assume t=0 is Noon.
        t_noon = 0
        t_afternoon = 10 # 40 degrees past noon?
        t_sunset = 22 # 88 degrees? (90 steps = 4 degrees/step). 22*4 = 88.
        
        # Plot 1: Standard Noon (Theta=1.0, Lat=0, Wave=5um)
        data = lut[0, 0, 0, t_noon, 0, :, :]
        plot_polar_slice(data, "Standard Noon\n(Theta=1.0, Lat=0, Wave=5um)", "lut_polar_1_noon_shortwave.png", vmin=0.8, vmax=1.5)
        
        # Plot 2: High Thermal Inertia Noon (Theta=20.0, Lat=0, Wave=5um)
        data = lut[4, 0, 0, t_noon, 0, :, :]
        plot_polar_slice(data, "High Inertia Noon\n(Theta=20.0, Lat=0, Wave=5um)", "lut_polar_2_noon_highinertia.png", vmin=0.8, vmax=1.5)

        # Plot 3: Long Wavelength Noon (Theta=1.0, Lat=0, Wave=100um)
        data = lut[0, 0, 0, t_noon, 4, :, :]
        plot_polar_slice(data, "Long Wavelength Noon\n(Theta=1.0, Lat=0, Wave=100um)", "lut_polar_3_noon_longwave.png", vmin=0.8, vmax=1.5)
        
        # Plot 4: Sunset (Grazing) (Theta=1.0, Lat=0, Wave=5um, Time=Sunset)
        # Note: At sunset, smooth surface is dark/cold? Reference might be 0?
        # If Reference is 0, Ratio is 1.0 (handled in generator).
        # Let's try Late Afternoon (60 deg phase). t=15 (15*4=60).
        t_late = 15
        data = lut[0, 0, 0, t_late, 0, :, :]
        plot_polar_slice(data, "Late Afternoon (60 deg)\n(Theta=1.0, Lat=0, Wave=5um)", "lut_polar_4_afternoon.png", vmin=0.5, vmax=2.0)
        
        # Plot 5: Polar Region (Theta=1.0, Lat=85, Wave=5um)
        # Time doesn't matter much at pole? Sun changes longitude.
        # At Lat=85, Sun elev is 5 deg. Constant.
        data = lut[0, 0, 3, 0, 0, :, :]
        plot_polar_slice(data, 'Polar Region (Lat=85)\n(Theta=1.0, Wave=5um)', "lut_polar_5_pole.png")

if __name__ == "__main__":
    analyze_lut_v2()
