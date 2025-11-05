# SPICE Integration Guide for TEMPEST

This guide explains how to use NASA's SPICE toolkit with TEMPEST to model realistic mission scenarios with time-varying solar illumination, body rotation states, and observer viewing geometry.

## Table of Contents

1. [What is SPICE?](#what-is-spice)
2. [Installation](#installation)
3. [Obtaining SPICE Kernels](#obtaining-spice-kernels)
4. [Configuration](#configuration)
5. [Running TEMPEST with SPICE](#running-tempest-with-spice)
6. [Understanding SPICE Output](#understanding-spice-output)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

## What is SPICE?

SPICE (Spacecraft Planet Instrument C-matrix Events) is an essential toolkit developed by NASA's Navigation and Ancillary Information Facility (NAIF) for computing geometric information needed by space missions. It provides:

- **Ephemerides**: Positions and velocities of solar system bodies and spacecraft
- **Orientation**: Rotation states and attitude information
- **Time conversions**: Between different time systems
- **Reference frames**: Coordinate system transformations
- **Instrument information**: Field of view and pointing data

When integrated with TEMPEST, SPICE enables:
- Time-varying solar distance and illumination angles
- Realistic body rotation including precession and nutation
- Accurate observer (spacecraft/telescope) viewing geometry
- Mission-specific thermal modeling scenarios

## Installation

SpiceyPy (Python wrapper for SPICE) is included in TEMPEST's requirements:

```bash
pip install spiceypy
```

Or if you're installing all TEMPEST dependencies:

```bash
pip install -r requirements.txt
```

## Obtaining SPICE Kernels

SPICE kernels are data files that contain the information SPICE needs. You'll typically need several types:

### Required Kernels

1. **LSK (Leap Seconds Kernel)**: Converts between time systems
   - Example: `naif0012.tls`
   - Location: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/

2. **SPK (Spacecraft/Planet Kernel)**: Positions and velocities
   - Planetary ephemeris: `de438.bsp` or `de440.bsp`
   - Location: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
   - Body-specific: Mission-specific archives

3. **PCK (Planetary Constants Kernel)**: Physical properties and orientation
   - Example: `bennu_v01.tpc` for Bennu
   - Location: Mission-specific archives or generic kernels

### Optional Kernels

4. **CK (C-Kernel)**: Orientation/attitude data for bodies or spacecraft
   - For tumbling bodies or spacecraft pointing

5. **FK (Frames Kernel)**: Reference frame definitions

6. **IK (Instrument Kernel)**: Instrument field-of-view definitions

### Where to Download

- **NAIF Generic Kernels**: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
- **Mission Archives**: https://naif.jpl.nasa.gov/naif/data.html
  - OSIRIS-REx (Bennu): https://naif.jpl.nasa.gov/pub/naif/OSIRIS-REX/
  - Lucy: https://naif.jpl.nasa.gov/pub/naif/LUCY/
  - New Horizons: https://naif.jpl.nasa.gov/pub/naif/NH/
  - Psyche: https://naif.jpl.nasa.gov/pub/naif/PSYCHE/

### Organizing Kernels

Create a `kernels/` directory in your TEMPEST root:

```
TEMPEST/
├── kernels/
│   ├── naif0012.tls          # Leap seconds
│   ├── de438.bsp             # Planetary ephemeris
│   ├── bennu_v01.tpc         # Bennu physical constants
│   ├── bennu_spk.bsp         # Bennu orbit
│   ├── bennu_ck.bc           # Bennu rotation
│   └── orex_v08.bsp          # OSIRIS-REx trajectory
├── data/
├── src/
└── ...
```

## Configuration

Add SPICE parameters to your config YAML file:

```yaml
# Enable SPICE mode
use_spice: true

# List of kernel files to load
spice_kernels:
  - "kernels/naif0012.tls"
  - "kernels/de438.bsp"
  - "kernels/bennu_v01.tpc"
  - "kernels/bennu_spk.bsp"
  - "kernels/bennu_ck.bc"
  - "kernels/orex_v08.bsp"

# Target body (NAIF name or ID)
spice_target_body: "BENNU"  # or "2101955"

# Observer (spacecraft, telescope, or location)
spice_observer: "OSIRIS-REX"  # or "EARTH", "SUN", etc.

# Time range
spice_start_time: "2019-03-01T00:00:00"  # UTC format
spice_duration_hours: 4.297              # One Bennu rotation

# Update frequency for geometry
spice_update_frequency: "per_timestep"   # Options: per_timestep, per_day, static

# Aberration correction
spice_illumination_aberration: "LT+S"    # Light time + stellar aberration

# Observer field of view (optional)
observer_fov_degrees: 0.8  # For instrument-specific calculations
```

### Configuration Options Explained

**`spice_update_frequency`**:
- `per_timestep`: Update geometry every timestep (most accurate, slower)
- `per_day`: Update geometry once per simulated day (faster, less accurate)
- `static`: Use single geometry snapshot (fastest, for spin-stabilized bodies)

**`spice_illumination_aberration`**:
- `NONE`: No corrections
- `LT`: Light time correction only
- `LT+S`: Light time + stellar aberration (recommended)
- `CN`: Converged Newtonian light time
- `CN+S`: Converged Newtonian + stellar aberration

## Running TEMPEST with SPICE

### Basic Usage

```bash
python tempest.py --config data/config/bennu_spice_example.yaml
```

### Validation

Before running a full simulation, validate your SPICE setup:

```bash
python scripts/validate_spice_geometry.py --config data/config/bennu_spice_example.yaml
```

This will:
- Check that all kernels load correctly
- Verify geometry calculations
- Plot solar distance and sun direction over time
- Save validation plots to `spice_geometry_validation.png`

### What TEMPEST Does with SPICE

When SPICE mode is enabled, TEMPEST:

1. **Loads kernels** at simulation initialization
2. **Precomputes geometry** for all timesteps (or as specified)
3. **Calculates insolation** using SPICE-derived sun positions
4. **Accounts for** time-varying solar distance
5. **Uses body orientation** from attitude kernels (if available)
6. **Calculates observer radiances** (if observer is specified)
7. **Cleans up** SPICE resources on completion

## Understanding SPICE Output

### Console Output

TEMPEST prints SPICE information at startup:

```
=== SPICE Mode Enabled ===
Target body: BENNU
Observer: OSIRIS-REX
Start time: 2019-03-01T00:00:00
Duration: 4.297 hours
Update frequency: per_timestep
Solar distance range: 1.2456 - 1.2458 AU
========================
```

### Observer Radiances

If an observer is specified (and not "SUN"), TEMPEST calculates and saves:

```
outputs/observer_radiances_YYYY-MM-DD_HH-MM-SS.csv
```

Contains:
- `Timestep`: Simulation timestep index
- `Total_Radiance_W_m2_sr`: Disk-integrated radiance

### Geometry Data

Access SPICE-derived geometry in your code:

```python
from src.model.simulation import Simulation
from src.utilities.config import Config

config = Config(config_path="your_config.yaml")
simulation = Simulation(config)

# Access precomputed arrays
solar_distances = simulation.spice_solar_distances  # meters
sun_directions = simulation.spice_sun_directions    # unit vectors
body_orientations = simulation.spice_body_orientations  # rotation matrices

# Query geometry at specific timestep
geometry = simulation.get_geometry_at_timestep(100)
print(f"Sun direction: {geometry['sun_direction']}")
print(f"Solar distance: {geometry['solar_distance_m']/1.496e11:.4f} AU")
```

## Troubleshooting

### Common Issues

**Problem**: `FileNotFoundError` for kernel files

**Solution**: 
- Check kernel paths in config are correct (relative to TEMPEST root)
- Verify kernel files exist
- Use absolute paths if needed

---

**Problem**: `SpiceKERNELNOTFOUND` error

**Solution**:
- Ensure leap seconds kernel (LSK) is loaded first
- Check NAIF ID/name spelling (use validation script)
- Verify planetary ephemeris covers your time range

---

**Problem**: `SpiceINVALIDTIME` error

**Solution**:
- Check time format: `"YYYY-MM-DDTHH:MM:SS"`
- Ensure time is covered by your kernels
- Verify leap seconds kernel is loaded

---

**Problem**: Body frame not found

**Solution**:
- Load appropriate PCK or FK kernel
- Check body name spelling
- Some bodies require specific frame kernels

---

**Problem**: Geometry seems incorrect

**Solution**:
- Run validation script to visualize geometry
- Check aberration correction setting
- Verify coordinate system assumptions
- Ensure shape model and SPICE use compatible frames

### Getting Help

1. **Run validation script** first: `python scripts/validate_spice_geometry.py --config your_config.yaml`
2. **Check NAIF documentation**: https://naif.jpl.nasa.gov/naif/tutorials.html
3. **Review SPICE error messages**: They usually indicate the specific problem
4. **Test with minimal kernels**: Start with just LSK and SPK, add others incrementally

## Examples

### Example 1: Bennu with OSIRIS-REx

Model Bennu's thermal behavior as observed by OSIRIS-REx:

```yaml
use_spice: true
spice_kernels:
  - "kernels/naif0012.tls"
  - "kernels/de438.bsp"
  - "kernels/bennu_v01.tpc"
  - "kernels/bennu_spk.bsp"
  - "kernels/bennu_ck.bc"
  - "kernels/orex_v08.bsp"
spice_target_body: "BENNU"
spice_observer: "OSIRIS-REX"
spice_start_time: "2019-03-01T00:00:00"
spice_duration_hours: 4.297
```

### Example 2: Earth-based Observation

Model an asteroid as seen from Earth:

```yaml
use_spice: true
spice_kernels:
  - "kernels/naif0012.tls"
  - "kernels/de438.bsp"
  - "kernels/asteroid_spk.bsp"
spice_target_body: "ASTEROID_NAME"
spice_observer: "EARTH"
spice_start_time: "2024-06-01T00:00:00"
spice_duration_hours: 24
```

### Example 3: Tumbling Body with CK

For a tumbling asteroid with attitude data:

```yaml
use_spice: true
spice_kernels:
  - "kernels/naif0012.tls"
  - "kernels/de438.bsp"
  - "kernels/asteroid_ck.bc"  # Contains rotation state
spice_target_body: "TUMBLING_ASTEROID"
spice_observer: "SUN"
spice_start_time: "2024-01-01T00:00:00"
spice_duration_hours: 48
spice_update_frequency: "per_timestep"  # Important for tumbling!
```

### Example 4: Mission-Specific Flyby

Model a spacecraft flyby:

```yaml
use_spice: true
spice_kernels:
  - "kernels/naif0012.tls"
  - "kernels/de438.bsp"
  - "kernels/target_spk.bsp"
  - "kernels/spacecraft_spk.bsp"
  - "kernels/spacecraft_ck.bc"
spice_target_body: "TARGET"
spice_observer: "SPACECRAFT"
spice_start_time: "2025-07-15T12:00:00"
spice_duration_hours: 6  # Flyby duration
observer_fov_degrees: 2.0
```

## Advanced Topics

### Custom Observer Positions

For non-SPICE observers, use manual positions:

```yaml
use_spice: true
# ... kernel config for target only ...
spice_observer: "SUN"  # or omit observer config

# Then in Python:
from src.model.observer import Observer
observer = Observer(
    name="GroundTelescope",
    manual_position=np.array([6.4e6, 0, 0])  # 6400 km from center
)
```

### Multiple Observers

Calculate radiances for multiple observers:

```python
from src.model.observer import Observer
from src.model.radiance import calculate_observed_radiance

observers = [
    Observer("OBSERVER1", spice_manager=simulation.spice_manager),
    Observer("OBSERVER2", spice_manager=simulation.spice_manager)
]

for obs in observers:
    radiance = calculate_observed_radiance(
        shape_model, thermal_data, simulation, config, obs, timestep=100
    )
    print(f"{obs.name}: {radiance['total_radiance']:.3e} W/m²/sr")
```

### Time-Varying Properties

Access geometry arrays for analysis:

```python
import matplotlib.pyplot as plt

# Plot solar flux variation
flux = simulation.solar_luminosity / (4 * np.pi * simulation.spice_solar_distances**2)
plt.plot(flux)
plt.xlabel('Timestep')
plt.ylabel('Solar Flux (W/m²)')
plt.show()
```

## References

- **NAIF SPICE Homepage**: https://naif.jpl.nasa.gov/naif/
- **SpiceyPy Documentation**: https://spiceypy.readthedocs.io/
- **SPICE Tutorials**: https://naif.jpl.nasa.gov/naif/tutorials.html
- **SPICE Required Reading**: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/

## Citation

If you use SPICE with TEMPEST in your research, please cite both:

1. TEMPEST (see main README.md)
2. SPICE: Acton, C.H.; "Ancillary Data Services of NASA's Navigation and Ancillary Information Facility"; Planetary and Space Science; Vol. 44, No. 1, pp. 65-70; 1996.

