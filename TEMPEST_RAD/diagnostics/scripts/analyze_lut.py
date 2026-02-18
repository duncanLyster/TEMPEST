
import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_lut():
    filename = 'roughness_lut_spectral_v1.h5'
    
    with h5py.File(filename, 'r') as f:
        print("Groups in H5:", list(f.keys()))
        
        # Dimensions: (Theta=5, Angle=1, Wave=5, Lat=4, Phase=90, Emission=10, Azimuth=10)
        
        lut = f['lut'][:]
        theta = f['theta'][:]
        wave = f['wavelength'][:]
        lat = f['latitude'][:]
        emission = f['emission'][:]
        azimuth = f['azimuth'][:]
        
        print("\nLUT Stats:")
        print(f"  Shape: {lut.shape}")
        print(f"  Min Value: {np.nanmin(lut):.4f}")
        print(f"  Max Value: {np.nanmax(lut):.4f}")
        print(f"  Mean Value: {np.nanmean(lut):.4f}")
        
        # 1. Full Histogram (Log Scale)
        data_flat = lut.flatten()
        data_clean = data_flat[~np.isnan(data_flat)]
        
        plt.figure(figsize=(10, 6))
        # Logbins for x-axis to encompass wide range
        try:
            logbins = np.logspace(np.log10(max(1e-2, np.min(data_clean))), np.log10(np.max(data_clean)), 50)
            plt.hist(data_clean, bins=logbins, color='skyblue', edgecolor='black', alpha=0.7)
        except:
             plt.hist(data_clean, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

        plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Smooth (1.0)')
        plt.title(f'Full Distribution (Max={np.max(data_clean):.1f})')
        plt.xlabel('Radiance Ratio (Log Scale)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig('lut_histogram_full.png')
        print("Saved full histogram to lut_histogram_full.png")
        
        # Investigate High Values (> 10)
        threshold = 10.0
        high_vals = data_clean[data_clean > threshold]
        pct = len(high_vals) / len(data_clean) * 100
        print(f"\nValues > {threshold}: {len(high_vals)} / {len(data_clean)} ({pct:.4f}%)")
        
        if len(high_vals) > 0:
            # Find indices of max value
            max_idx = np.unravel_index(np.argmax(lut), lut.shape)
            print(f"\nMax Value Analysis ({np.max(data_clean):.2f}):")
            print(f"  Theta: {theta[max_idx[0]]} (Index {max_idx[0]})")
            print(f"  Lat:   {lat[max_idx[2]]} (Index {max_idx[2]})")
            print(f"  Time:  Index {max_idx[3]} / 90")
            print(f"  Wave:  {wave[max_idx[4]]} um (Index {max_idx[4]})")
            print(f"  Emi:   {emission[max_idx[5]]} (Index {max_idx[5]})")
            print(f"  Azi:   {azimuth[max_idx[6]]} (Index {max_idx[6]})")
            
        # 2. Polar Plot (Fixed)
        # Slice: Theta=1.0 (Index 0), Lat=0 (Index 0), Noon (Time=0), Wave=8um (Index 1)
        # Note: Generator defines Time=0 as Noon (Sun aligned with normal at Lat=0)
        
        time_noon = 0
        slice_noon = lut[0, 0, 0, time_noon, 1, :, :] # (Emi=10, Azi=10)
        
        print(f"\nNoon Slice (Theta=1.0, Lat=0) Stats:")
        print(f"  Min: {np.nanmin(slice_noon):.3f}")
        print(f"  Max: {np.nanmax(slice_noon):.3f}")
        print(f"  Mean: {np.nanmean(slice_noon):.3f}")

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')
        
        # Azimuths are 0 to 180. We need to mirror to 360 for full plot
        azimuths_rad = np.radians(azimuth) # 0 to pi
        
        # Meshgrid for right half
        R, Theta = np.meshgrid(emission, azimuths_rad) # (10, 10)
        
        # Plot Right Half
        # pcolormesh(Theta, R, Values). Values shape must match Theta.
        # slice_noon is (Emi, Azi) -> Transpose to (Azi, Emi)
        vals_right = slice_noon.T
        
        # Plot Left Half (Mirror)
        # 360 - azimuth gives angles like 360, 340... 180.
        azimuths_left = np.radians(360 - azimuth) 
        R_left, Theta_left = np.meshgrid(emission, azimuths_left)
        vals_left = slice_noon.T
        
        # Use vmin/vmax centered on 1.0 to see deviations clearly
        # Or auto-scale if range is huge
        vmin = 0.8
        vmax = 1.2
        cmap = 'coolwarm'
        
        # Note: shading='auto' or 'nearest' to avoid dimensions warning
        c1 = ax.pcolormesh(Theta, R, vals_right, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        c2 = ax.pcolormesh(Theta_left, R_left, vals_left, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

        plt.colorbar(c1, ax=ax, label='Roughness Factor')
        plt.title(f'Radiance Ratio at Noon (Theta=1.0)\nSun Overhead')
        plt.ylim(0, 90)
        
        plt.savefig('lut_polar_noon_fixed.png')
        print("Saved polar plot lut_polar_noon_fixed.png")

if __name__ == "__main__":
    analyze_lut()
