# TEMPEST_RAD Bug Fix: Sun Phase Axis Mismatch

## Problem Summary

The TEMPEST_RAD roughness LUT contains correction factors that SHOULD make rough surfaces appear warmer at certain viewing geometries (e.g., disk center at midday), but in practice the rough model appears cooler for **all** viewing geometries.

## Root Cause

There is a **misalignment between the LUT's sun_phase axis and the actual local time** represented by the thermal simulation data:

### What the LUT sun_phase axis claims:
- `sun_phase = 0°` should mean "local noon" (facet pointing at sun)
- `sun_phase = 180°` should mean "local midnight" (facet pointing away from sun)

### What the LUT data actually contains:
- `t_idx = 0` (`sun_phase = 0°`): **NIGHT/DAWN** (correction factor = 0.955)
- `t_idx = 67` (`sun_phase = 268°`): **AFTERNOON/PEAK** (correction factor = 1.422)

### Result:
When the retrieval code asks for `sun_phase = 0°` (expecting bright midday),  
it gets `t_idx = 0` which contains **cold nighttime data**, resulting in correction factors < 1.0.

## Evidence

From `check_time_pattern.py`:
```
Time Idx | Sun Phase | Factor | Actual Time
    0    |     0.0°  | 0.955  | Night/Dawn (COLD)
   22    |    88.0°  | 1.038  | Getting brighter
   45    |   180.0°  | 1.336  | Midday
   67    |   268.0°  | 1.422  | Afternoon (PEAK)
   89    |   356.0°  | 0.968  | Evening/Night
```

The peak temperature/brightness occurs at `t_idx ≈ 67` (268°), not at `t_idx = 0` (0°).

## Physics Behind the Offset

This pattern is actually **physically correct** due to **thermal inertia**:
- The simulation starts at an arbitrary phase
- Thermal lag causes peak temperature to occur AFTER local solar noon
- For the thermal parameter θ used, the lag is ~70-90° (a few hours after noon)

The problem is that the **sun_phase axis was incorrectly defined** when assembling the LUT.

## Solution Options

### Option 1: Regenerate LUT with Correct Axis (RECOMMENDED)

In `generator.py`, the sun_phase values need to be aligned with the actual local time:

**Current code (line ~621):**
```python
f.create_dataset("lut", data=lut_tensor)
# Implicit: sun_phases = linspace(0, 360, 90)
```

**Fix:** Determine the actual local time of each timestep and create the sun_phase axis accordingly. This requires:

1. Track which timestep corresponds to "local noon" (when facet normal points at sun)
2. Shift the sun_phase axis so that `sun_phase = 0°` aligns with that timestep
3. Or, rearrange the LUT data so that the thermal noon is at `t_idx = 0`

**Implementation:** Add this to `process_single_case()` before storing results:

```python
# Find the timestep where the crater is pointing most directly at the sun
# This is when sun_vec dot facet_normal is maximum
sun_dots = [np.dot(sun_vectors[t], facet_normal) for t in range(LUT_TIMESTEPS)]
noon_idx = np.argmax(sun_dots)

# Roll the array so that noon is at index 0
result_grid = np.roll(result_grid, -noon_idx, axis=0)
rough_temps = np.roll(rough_temps, -noon_idx, axis=1)
smooth_temps = np.roll(smooth_temps, -noon_idx)
```

### Option 2: Apply Offset in Retrieval Code (WORKAROUND)

Modify the LUT loader to apply a phase offset:

In `TEMPEST_RAD/lut.py`, modify `get_correction_factors()`:

```python
def get_correction_factors(self, latitudes, sun_phases, emissions, azimuths, wavelength=None):
    if not self.is_loaded:
        return np.ones_like(emissions)
        
    # WORKAROUND: Apply empirical phase offset
    # The LUT data is offset by ~268° (peak brightness)
    # We want peak at sun_phase=0°, so shift by -268° (or +92°)
    sun_phases_corrected = (sun_phases + 92.0) % 360.0
    
    # Wrap phases
    sun_phases_corrected = np.mod(sun_phases_corrected, 360.0)
    # ... rest of function uses sun_phases_corrected
```

**WARNING:** The offset value (92° used here) is **empirical** and may vary with:
- Thermal inertia (theta parameter)
- Latitude
- Crater geometry

This is a HACK and not recommended for production.

### Option 3: Fix the Time Axis Definition (PROPER FIX)

The underlying issue is in how the thermal simulation is initialized and sampled.

In `generator.py::simulate_crater_diurnal_cycle()`, ensure the simulation starts with the crater at local noon:

**After line 262 (before thermal simulation):**
```python
# Rotate the crater so that t=0 corresponds to local noon
# Initial rotation should place crater facing the sun
initial_phase = 0.0  # Noon
simulation.initial_rotation_angle = initial_phase
```

Then the sun_phase axis will naturally align:
```python
sun_phases = np.linspace(0, 360, n_time, endpoint=False)  # Correct!
```

## Testing the Fix

After applying the fix, run this test:

```python
# Test at disk center, midday
lats = np.array([0.0])
phases = np.array([0.0])  # Local noon
emis = np.array([0.0])     # Disk center
azis = np.array([0.0])

factors = lut.get_correction_factors(lats, phases, emis, azis, wavelength=8.0)
print(f"Correction factor at disk center, noon, 8um: {factors[0]:.3f}")
print(f"Expected: >1.3 (enhancement)")
print(f"Status: {'✓ PASS' if factors[0] > 1.2 else '✗ FAIL'}")
```

## Impact

This bug causes:
- ✗ Rough surfaces appear universally cooler than smooth
- ✗ No visible limb brightening (it's there, but obscured by the offset)
- ✗ Phase curves are inverted or shifted
- ✗ Retrievals interpret cool rough surfaces as low thermal inertia

After fixing:
- ✓ Rough surfaces show proper thermal beaming at zero phase
- ✓ Temperature enhancements visible at appropriate geometries
- ✓ Phase curves have correct shape
- ✓ Thermal property retrievals become accurate

## Recommended Action

**Regenerate the LUT** using Option 1 or Option 3. The existing LUT contains correct physics but incorrect axis labeling, making it unusable for retrievals without significant workarounds.

Estimated time to fix and regenerate: 2-4 hours (depending on computational resources).
