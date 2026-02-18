#!/usr/bin/env python3
"""
Estimate memory requirements for different LUT resolutions
"""
import numpy as np

print("="*70)
print("LUT Memory Requirement Calculator")
print("="*70)

# Fixed dimensions (you probably want to keep these)
n_theta = 5  # Thermal parameter values
n_angle = 1  # Opening angle (90° hemisphere)
n_time = 90  # Body rotation phases
n_wave = 5   # Wavelengths
n_emi = 10   # Emission angles
n_azi = 10   # Azimuth angles

# Variable: latitude resolution
configs = [
    ("Current (test)", 4, [0, 30, 60, 85]),
    ("Low-res", 7, "0° to 90° in 15° steps"),
    ("Medium-res", 13, "0° to 90° in 7.5° steps"),
    ("High-res", 19, "0° to 90° in 5° steps"),
    ("Very high-res", 37, "0° to 90° in 2.5° steps"),
]

print(f"\nFixed dimensions:")
print(f"  Theta: {n_theta}")
print(f"  Angle: {n_angle}")
print(f"  Time: {n_time}")
print(f"  Wavelength: {n_wave}")
print(f"  Emission: {n_emi}")
print(f"  Azimuth: {n_azi}")

print(f"\n{'Configuration':<20} {'N_lat':<8} {'Total Elements':<18} {'Memory (float64)':<20}")
print("-"*70)

for name, n_lat, desc in configs:
    total_elements = n_theta * n_angle * n_lat * n_time * n_wave * n_emi * n_azi
    
    # float64 = 8 bytes
    memory_bytes = total_elements * 8
    memory_mb = memory_bytes / (1024**2)
    memory_gb = memory_bytes / (1024**3)
    
    if memory_gb >= 1.0:
        mem_str = f"{memory_gb:.2f} GB"
    else:
        mem_str = f"{memory_mb:.1f} MB"
    
    print(f"{name:<20} {n_lat:<8} {total_elements:>15,}    {mem_str:<20}")
    if isinstance(desc, str):
        print(f"  └─ {desc}")

print("\n" + "="*70)
print("RECOMMENDATIONS for 6GB RAM:")
print("="*70)

print("""
1. **Low-res (7 latitudes)**: ~86 MB
   - Good for testing and validation
   - 15° gaps acceptable for most applications
   - Latitude points: [0, 15, 30, 45, 60, 75, 90]

2. **Medium-res (13 latitudes)**: ~156 MB
   - Better interpolation accuracy
   - 7.5° gaps capture physics well
   - Still fits easily in RAM

3. **High-res (19 latitudes)**: ~228 MB  ✓ RECOMMENDED
   - 5° gaps minimize interpolation artifacts
   - Captures beaming→shadowing transition smoothly
   - Only 228 MB - plenty of headroom for processing

4. **Very high-res (37 latitudes)**: ~445 MB
   - 2.5° gaps (possibly overkill)
   - Use if you need ultra-high fidelity

Note: These are just the LUT storage costs. Generation requires:
  - Crater thermal simulation per (lat, theta, angle)
  - Processing ~5-10x the final LUT size
  - For 19 latitudes: ~2-3GB total during generation (safe for 6GB)

VERDICT: Go with **19 latitudes** (5° steps). This will:
- Eliminate interpolation artifacts
- Still be fast to generate and use
- Leave plenty of RAM for other operations
""")

print("="*70)
