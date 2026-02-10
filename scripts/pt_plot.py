import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- SETTINGS ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 13

# Postage stamp size for side panel
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# --- AXES CONFIGURATION ---
# Primary Y: Pressure (MPa). 
max_pressure = 40 
ax.set_ylim(max_pressure, 0) # Inverted: 0 MPa at surface
ax.set_xlim(40, 300)

# --- 0. BACKGROUND SHADING (PASTEL ZONES) ---

# Zone 1: The Gap (Cold) - Pastel Red
# Range: ~40K to 100K
ax.axvspan(40, 100, color='#ffb3ba', alpha=0.5)
ax.text(70, 38, "Significant\nData Gap", color='#c0392b', 
        fontweight='bold', fontsize=12, ha='center', va='bottom')

# Zone 2: Limited Data (Intermediate) - Pastel Orange
# Range: 100K to 218K
ax.axvspan(100, 218, color='#ffdfba', alpha=0.5)
ax.text(159, 38, "Limited\nData", color='#d35400', 
        fontweight='bold', fontsize=12, ha='center', va='bottom')

# Zone 3: Rich Data (Warm/Earth) - Pastel Green
# Range: 218K to 300K
ax.axvspan(218, 300, color='#baffc9', alpha=0.5)
ax.text(259, 38, "Rich Data\n(Lab)", color='#27ae60', 
        fontweight='bold', fontsize=12, ha='center', va='bottom')


# --- 1. DATA REGIONS (OVERLAYS) ---

# Europa Context Polygon (The "Target")
# Surface: 50K - 110K @ 0 MPa
# Base: 260K - 270K @ ~36 MPa
eur_x = [50, 110, 270, 260] 
eur_y = [0, 0, 36, 36] 
poly_eur = patches.Polygon(list(zip(eur_x, eur_y)), closed=True,
                           facecolor='#5DADE2', edgecolor='none', alpha=0.6)
ax.add_patch(poly_eur)

# Label for Europa (Directly on plot)
ax.text(80, 5, "Europa Surface", color='#154360', fontweight='bold', fontsize=11, ha='center')
# Moved to align better with the slope of the blue region
ax.text(180, 22, "Europa Ice Shell", color='#154360', fontweight='bold', fontsize=12, 
        ha='center', rotation=-40)




# --- 4. FORMATTING & SECONDARY AXIS ---
ax.set_xlabel("Temperature (K)", fontsize=13, fontweight='bold')
ax.set_ylabel("Pressure (MPa)", fontsize=13, fontweight='bold')
ax.set_title("Thermal Conductivity Knowledge Gap", fontsize=14, fontweight='bold', pad=8)

# Reduce number of ticks (half as many)
ax.set_xticks([50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 10, 20, 30, 40])

# Secondary Axis: Convert Pressure (MPa) to Europa Depth (km)
# Factor: ~1.21 MPa per km (g=1.315, rho=920)
def pressure2europa(p):
    return p / 1.21
def europa2pressure(d):
    return d * 1.21

secax = ax.secondary_yaxis('right', functions=(pressure2europa, europa2pressure))
secax.set_ylabel('Europa Depth (km)', fontsize=13)
# Set fewer ticks on secondary axis too
secax.set_yticks([0, 10, 20, 30])
secax.tick_params(labelsize=12)

# Tick label sizes
ax.tick_params(labelsize=12)

# Grid
ax.grid(True, linestyle=':', alpha=0.4, color='gray')

plt.tight_layout()
plt.show()
