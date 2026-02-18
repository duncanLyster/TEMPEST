"""
Calculate realistic Theta values for different scenarios.

Theta = (TI * sqrt(omega)) / (epsilon * sigma * Tss^3)

Where TI is inverted in the generator as:
TI = (theta * epsilon * sigma * Tss^3) / sqrt(omega)
"""

import numpy as np

# Constants
sigma = 5.67e-8  # Stefan-Boltzmann
epsilon = 0.95
albedo = 0.12
solar_flux_1AU = 1361  # W/m^2

# Calculate subsolar temperature
Tss = ((1 - albedo) * solar_flux_1AU / (epsilon * sigma))**0.25
print(f"Subsolar temperature at 1 AU: {Tss:.1f} K")

# Scenarios to test
scenarios = [
    ("Fast asteroid (2h)", 2.0, 20),
    ("Typical asteroid (6h)", 6.0, 100),
    ("Slow asteroid (12h)", 12.0, 200),
    ("Reference config (10h)", 10.0, 100),
    ("Moon (708h)", 708.0, 50),
    ("Very slow rotator (100h)", 100.0, 100),
]

print("\nThermal Inertia → Theta conversions:")
print("=" * 80)
print(f"{'Scenario':<30} {'Period (h)':<12} {'TI':<10} {'Theta':<10}")
print("-" * 80)

for name, period_hours, TI in scenarios:
    omega = 2 * np.pi / (period_hours * 3600)
    theta = (TI * np.sqrt(omega)) / (epsilon * sigma * (Tss**3))
    print(f"{name:<30} {period_hours:<12.1f} {TI:<10.0f} {theta:<10.3f}")

print("\n" + "=" * 80)

# Typical TI ranges
print("\nTypical Thermal Inertia ranges:")
print("  - Dust/regolith: 10-50 J m^-2 K^-1 s^-1/2")
print("  - Lunar regolith: 50-100")
print("  - Sand/fine particles: 30-200")
print("  - Bare rock: 300-2500")

# Calculate Theta range for reference period (10h)
print(f"\nFor rotation period = 10 hours:")
period_hours = 10.0
omega = 2 * np.pi / (period_hours * 3600)

TI_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
print(f"{'TI':<10} {'Theta':<10}")
print("-" * 20)
for TI in TI_values:
    theta = (TI * np.sqrt(omega)) / (epsilon * sigma * (Tss**3))
    print(f"{TI:<10.0f} {theta:<10.4f}")

# Reverse: What TI does Theta=1 to 100 give?
print(f"\n{'Theta':<10} {'TI':<10} {'Type':<30}")
print("-" * 50)
for theta in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    TI = (theta * epsilon * sigma * (Tss**3)) / np.sqrt(omega)
    if TI < 50:
        type_str = "Dust (unrealistically low?)"
    elif TI < 200:
        type_str = "Regolith/sand"
    elif TI < 1000:
        type_str = "Compacted regolith/loose rock"
    elif TI < 3000:
        type_str = "Solid rock"
    else:
        type_str = "Too high (unrealistic)"
    print(f"{theta:<10.1f} {TI:<10.0f} {type_str:<30}")

# Recommended range
print("\n" + "=" * 80)
print("RECOMMENDED Theta range for comprehensive LUT:")
print("  - Fast rotators (2-6h), TI=20-500: Theta ≈ 0.1 to 2.0")
print("  - Medium rotators (10-24h), TI=50-2000: Theta ≈ 0.2 to 8.0")
print("  - Slow rotators (100-1000h), TI=50-500: Theta ≈ 0.02 to 0.5")
print("\nFor broad coverage at P=10h:")
print("  → Theta from 0.04 to 8.0 covers TI from 10 to 2000")
print("  → Use: np.logspace(-1.5, 1, 10)  # 0.032 to 10")
print("=" * 80)
