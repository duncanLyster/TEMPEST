# Roughness LUT Latitude Correlation Issue

## Current Status (2026-02-18 14:30)

### Problem
The roughness correction factors show **strong negative correlation with latitude** (r ≈ -0.975), indicating a systematic geometry error rather than random discretization noise.

### Test Results

Latest LUT: `roughness_lut_moon_test.h5` (generated 14:27)

**Correction Factors at Noon (180°), Disk Center, 15µm:**
```
Latitude    Factor    Status
   0.0°      2.31      ✓ (thermal beaming present)
  22.5°      2.21      ✓
  45.0°      1.83      ✓
  67.5°      1.19      ✓
  90.0°      0.95      ✗ (< 1.0, beaming lost!)

Correlation: r = -0.975, p = 0.005
```

**Expected:** |r| < 0.3 (random variation from discretization)  
**Actual:** |r| = 0.975 (systematic trend)  
**Status:** ❌ FAIL

---

## Bugs Fixed

### ✅ Bug #1: Phase Parameter Mismatch
**Location:** [simulator.py](TEMPEST_RAD/simulator.py#L92-L177) `compute_geometry()`

**Problem:** Per-facet local hour angles passed to LUT instead of global body rotation phase

**Fix:** Calculate global phase from sun's equatorial projection:
```python
# Before: phase[i] = angle from facet normal to sun (0-360° per facet)
# After:  phase[i] = global_phase for all facets (body rotation angle)

sun_xy = sun_vec[:2] / np.linalg.norm(sun_vec[:2])
reference = np.array([1.0, 0.0])  # +X direction
global_phase = np.degrees(np.arctan2(sun_xy[1], sun_xy[0]))
return np.full(n_facets, global_phase)
```

**Verification:** Phase std dev: 137° → 0° ✓

---

### ✅ Bug #2: Normalization Discretization Mismatch  
**Location:** [generator.py](TEMPEST_RAD/generator.py#L594-L640)

**Problem:** Smooth reference used analytic formula π×B(T), rough used numerical integration. Discretization errors didn't cancel in ratio.

**Fix:** Both use same numerical integration:
```python
# Before (smooth): flux_smooth = np.pi * planck_function(wave, temp)
# After (smooth):  flux_smooth = sum(emission_i * solid_angle_i)  # Same as rough
```

**Result:** Made correlation WORSE (r: -0.715 → -0.973), proving bug was elsewhere ⚠️

---

### ❌ Bug #3: Sun Declination Geometry (ROOT CAUSE - UNRESOLVED)

**Problem:** The "latitude" parameter uses sun_declination approach:
- Crater placed at **equator** (opening toward +X)
- Sun **tilted** by declination angle: `sun_vec = [cos(lat), 0, sin(lat)]`
- Body rotates around Z-axis

**Why this causes correlation:**

At **Latitude 0° (Equator):**
- Sun overhead at noon: `sun = [1, 0, 0]`
- Crater normal: `n = [1, 0, 0]`  
- Sun perpendicular to crater opening → maximum illumination
- Strong thermal self-heating between crater walls

At **Latitude 90° (Pole):**
- Sun grazing from side: `sun = [0, 0, 1]`
- Crater normal: `n = [1, 0, 0]`
- Sun illuminates crater at shallow angle → less self-heating
- Cooler internal facets → less thermal beaming → lower correction factor

**Physical Issue:** At high latitudes, the declination approach creates weaker thermal self-heating because sun grazes the crater differently. This is NOT equivalent to a polar crater with the sun circling the horizon.

---

## Alternative Approach Attempted (Failed)

**Idea:** Rotate crater to latitude, keep sun at equator
- Crater at pole: `n = [0, 0, 1]`
- Sun at equator: `sun = [1, 0, 0]`, circles XY plane
- Body rotation creates correct sun path for that latitude

**Result:** 46% of LUT filled with **zeros**
- At lat 45°: 55% zeros
- At lat 67.5°: 82% zeros  
- At lat 90°: 76% zeros

**Why it failed:** Deep craters at high latitudes never receive direct sunlight when sun circles the horizon. This is physically correct for permanently shadowed regions but breaks LUT generation.

---

## Root Cause Analysis

The issue is **fundamental to how "latitude" is parameterized:**

### Current Interpretation (Sun Declination):
- "Latitude" = solar declination angle
- Crater always at equator
- Different thermal self-heating at different "latitudes"
- ❌ Creates systematic correlation

### Geographic Interpretation (Crater Position):
- "Latitude" = actual geographic position  
- Sun at equator
- No illumination at poles
- ❌ Creates zeros in LUT

### Correct Interpretation (Not Implemented):
- "Latitude" should parameterize **average solar illumination angle**
- Should NOT affect crater thermal self-heating geometry
- Self-heating should remain constant; only top-level illumination varies
- Need to decouple these two effects

---

## Solution Required

**Rethink LUT Parameterization:**

The latitude axis should represent variations in **external solar illumination conditions**, not the crater's internal thermal geometry.

Possible approaches:
1. **Separate parameters:** Crater geometry (fixed) + solar zenith angle (varying)
2. **Fixed geometry:** Always simulate at equator with full illumination, parameterize viewing geometry separately
3. **Illumination-only:** Pre-compute crater thermal response, then modulate by solar illumination function

Currently, the sun_declination approach conflates these two effects, causing systematic bias.

---

## User Insight (Key)

> *"I don't like that this correction factor seems to have such a clear trend with latitude... If we have conservation of energy the total radiant flux out of a facet for one full rotation should be the same whether it is smooth or rough, so I'd expect the normalisation to just be caused by the discretisation of the geometry... and it should look random against latitude"*

This physical reasoning exposed the fundamental problem: systematic trends violate energy conservation principles and indicate a physics error, not discretization noise.

---

## Next Steps

1. **Decision Point:** How should "latitude" be parameterized in the LUT?
   - Should it affect crater geometry? (causes correlation)
   - Should it only affect illumination? (how to implement?)
   - Should we use different axis name? (e.g., "solar_zenith_angle")

2. **Test Hypothesis:** Run simulation with crater at equator, sun at equator, NO latitude variation
   - All correction factors should be identical
   - Proves geometry is the issue

3. **Redesign LUT:** Consider decoupling internal crater geometry from external illumination conditions

---

## Files Modified

- ✅ [TEMPEST_RAD/simulator.py](TEMPEST_RAD/simulator.py) - Phase calculation fixed
- ✅ [TEMPEST_RAD/generator.py](TEMPEST_RAD/generator.py) - Normalization + geometry (reverted)
- ✅ [TEMPEST_RAD/generator_quick_test.py](TEMPEST_RAD/generator_quick_test.py) - Fast testing (2 min)
- ✅ [roughness_lut_moon_test.h5](roughness_lut_moon_test.h5) - Test LUT with fixes (still shows correlation)

---

## Diagnostics

Run these cells in the notebook to verify current status:
- **Cell 30:** Direct HDF5 check of correction factors  
- **Cell 31:** Full correlation test with visualization

Both should show r ≈ -0.975 with current LUT.
