import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Physical constants
h = 6.62607015e-34  # Planck constant (J·s)
c = 2.99792458e8    # Speed of light (m/s)
c_cm = 2.99792458e10  # Speed of light (cm/s)
k = 1.380649e-23    # Boltzmann constant (J/K)

def planck_wavenumber_correct(wavenumber_cm, temperature):
    """
    Calculate Planck function in wavenumber units
    
    Parameters:
    wavenumber_cm: wavenumber in cm^-1
    temperature: temperature in K
    
    Returns:
    Spectral radiance in W/(m^2·sr·cm^-1)
    """
    # Convert wavenumber to frequency: ν = c * ν̃
    nu_Hz = wavenumber_cm * 100 * c  # cm^-1 * (100 cm/m) * (m/s) = Hz
    
    # Planck function in frequency form: B_ν
    exponent = h * nu_Hz / (k * temperature)
    if exponent > 700:
        return 0.0
    
    B_nu = (2 * h * nu_Hz**3 / c**2) / (np.exp(exponent) - 1)  # W/(m²·sr·Hz)
    
    # Convert from per Hz to per cm^-1: dν/dν̃ = c (in cm/s)
    B_nu_tilde = B_nu * c_cm  # W/(m²·sr·cm^-1)
    
    return B_nu_tilde

# Load datasets
file_path1 = "/Users/duncan/Desktop/DPhil/TEMPEST/data/output/remote_outputs/animation_outputs_2026-01-15_11-00-36/animation_params.npz"
data1 = np.load(file_path1)
temperatures1 = data1['plotted_variable_array'][:, 2500]

file_path2 = "/Users/duncan/Desktop/DPhil/TEMPEST/data/output/remote_outputs/animation_outputs_2026-01-15_11-06-20/animation_params.npz"
data2 = np.load(file_path2)
temperatures2 = data2['plotted_variable_array'][:, 2500]

print(f"Dataset 1: {len(temperatures1)} facets, T range: {temperatures1.min():.1f}-{temperatures1.max():.1f} K")
print(f"Dataset 2: {len(temperatures2)} facets, T range: {temperatures2.min():.1f}-{temperatures2.max():.1f} K")

# Create wavenumber grid
wavenumbers = np.linspace(10, 500, 1000)  # cm^-1

# Initialize radiance arrays
radiance1 = np.zeros_like(wavenumbers)
radiance2 = np.zeros_like(wavenumbers)
radiance_60K = np.zeros_like(wavenumbers)
radiance_70K = np.zeros_like(wavenumbers)

print("\nCalculating blackbody radiances...")

# Calculate weighted radiance for dataset 1
for temp in temperatures1:
    if temp > 0:
        for i, wn in enumerate(wavenumbers):
            radiance1[i] += planck_wavenumber_correct(wn, temp) / len(temperatures1)

# Calculate weighted radiance for dataset 2
for temp in temperatures2:
    if temp > 0:
        for i, wn in enumerate(wavenumbers):
            radiance2[i] += planck_wavenumber_correct(wn, temp) / len(temperatures2)

# Calculate 60 K blackbody curve
for i, wn in enumerate(wavenumbers):
    radiance_60K[i] = planck_wavenumber_correct(wn, 60.0)

# Calculate 70 K blackbody curve
for i, wn in enumerate(wavenumbers):
    radiance_70K[i] = planck_wavenumber_correct(wn, 70.0)

print("Calculation complete!")

# Create plot
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

fig, ax = plt.subplots(figsize=(12, 8))

# Colors
title_color = '#12253a'
orange_color = '#d57b5c'
blue_color = '#4682B4'
purple_color = '#9370DB'

ax.plot(wavenumbers, radiance1, linewidth=2.5, label='Smooth surface', color=blue_color, linestyle='-')
ax.plot(wavenumbers, radiance2, linewidth=2.5, label='Realistic terrain', color=blue_color, linestyle='-.', dash_capstyle='round')
ax.plot(wavenumbers, radiance_60K, linewidth=2.5, label='60 K Blackbody', color=orange_color, linestyle='--')
ax.plot(wavenumbers, radiance_70K, linewidth=2.5, label='70 K Blackbody', color=purple_color, linestyle='--')

ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=22)
ax.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ cm$^{-1}$)', fontsize=22)
ax.set_title('Blackbody Radiance Spectra', fontsize=26, fontweight='bold', pad=15, color=title_color)
ax.set_xlim(0, 500)

# Format y-axis in scientific notation
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

ax.legend(fontsize=18, loc='upper right')
ax.tick_params(labelsize=18)
ax.grid(alpha=0.3)

output_path = "/Users/duncan/Desktop/DPhil/TEMPEST/radiance_spectra_comparison.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nRadiance spectra saved to: {output_path}")

# Print some statistics
peak1_idx = np.argmax(radiance1)
peak2_idx = np.argmax(radiance2)
peak_60K_idx = np.argmax(radiance_60K)
peak_70K_idx = np.argmax(radiance_70K)
print(f"\nSmooth surface: Peak at {wavenumbers[peak1_idx]:.1f} cm^-1, Peak radiance: {radiance1[peak1_idx]:.3e} W/(m^2·sr·cm^-1)")
print(f"Realistic terrain: Peak at {wavenumbers[peak2_idx]:.1f} cm^-1, Peak radiance: {radiance2[peak2_idx]:.3e} W/(m^2·sr·cm^-1)")
print(f"60 K Blackbody: Peak at {wavenumbers[peak_60K_idx]:.1f} cm^-1, Peak radiance: {radiance_60K[peak_60K_idx]:.3e} W/(m^2·sr·cm^-1)")
print(f"70 K Blackbody: Peak at {wavenumbers[peak_70K_idx]:.1f} cm^-1, Peak radiance: {radiance_70K[peak_70K_idx]:.3e} W/(m^2·sr·cm^-1)")

plt.close()
