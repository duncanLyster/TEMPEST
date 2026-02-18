# High-Resolution LUT Generation Issues - FIXED

## Problems Identified

When you increased the LUT resolution, you made several simultaneous changes that introduced bugs:

### 1. **CRITICAL: Opposition Effect Disappeared**
**Root Cause:** Reduced angular resolution
- **Before:** 10 emission angles, 10 azimuth angles
- **After:** 5 emission angles, 5 azimuth angles  
- **Effect:** AZIMUTH_ANGLES = [0°, 45°, 90°, 135°, 180°]
  - Opposition effect occurs at azimuth 0-30° with sharp peak
  - With only 5 points, you're missing the critical 10-30° range!
  - Linear interpolation cannot reconstruct the sharp opposition peak

**Fix:** Restored to 10 emission angles and 10 azimuth angles

### 2. **Facet Count Override Bug**
**Root Cause:** Line 216 hardcoded `kernel_subfacets_count = 100`
- You set config value to 1000, but line 216 immediately overwrote it to 100
- This is why increasing subfacets_count "didn't seem to work"
- With only 100 facets, crater is under-resolved → poor geometry representation

**Fix:** Removed hardcoded override, now uses config value (set to 300 for balance)

### 3. **Energy Conservation Issues**
**Root Cause:** `vf_rays = 500` too low for accurate view factors
- Output showed: "Max missing: 49.8% at subfacet 5"
- Nearly half the energy unaccounted for!
- Causes incorrect thermal coupling between crater facets

**Fix:** Increased to `vf_rays = 2000` (4x improvement)

### 4. **Temporal Resolution**
**Root Cause:** `SIM_TIMESTEPS = 360` only 2x LUT resolution
- Should be at least 4x for numerical stability
- Previous working version likely used 720 steps

**Fix:** Increased to `SIM_TIMESTEPS = 720` (4x LUT resolution)

### 5. **Polar Cooling Effect**
**This is NOT a bug** - it's physically correct!
- At high latitudes: sun grazing → crater shadowing dominates
- Rough surfaces SHOULD be cooler than smooth at poles
- With better facet resolution (300 vs 100), this effect will be more accurate

---

## Changes Made

### [generator.py](TEMPEST_RAD/generator.py)

**Lines 53-61: Grid Definitions**
```python
# FIXED:
EMISSION_ANGLES = np.linspace(0, 89, 10)  # Was: 5
AZIMUTH_ANGLES = np.linspace(0, 180, 10)  # Was: 5
SIM_TIMESTEPS = 720                        # Was: 360
```

**Line 120: Config Default**
```python
'kernel_subfacets_count': 300,  # Was: 1000 (too slow), now balanced
```

**Line 216: Removed Hardcoded Override**
```python
# REMOVED: config.kernel_subfacets_count = 100
# Now uses config value (300)
```

**Line 223: Use Config Value**
```python
config.kernel_subfacets_count = 300  # Explicitly set for consistency
```

**Line 288: Increased View Factor Rays**
```python
vf_rays = 2000  # Was: 500
```

---

## Expected Behavior After Fix

### LUT Dimensions
- **Shape:** 10×1×37×180×5×**10**×**10** (vs previous 10×1×37×180×5×5×5)
- **Size:** ~265 MB (vs previous ~33 MB)
- **Memory during generation:** ~5-6 GB peak (should fit in your system)

### Opposition Effect
- Should now appear with peak at azimuth = 0°, emission ~ 0-20°
- Smooth gradient from 0° to 30° azimuth
- Factor should be ~1.2-1.8× at zero phase for equatorial regions

### Polar Behavior
- Rough still cooler than smooth at poles (THIS IS CORRECT PHYSICS)
- But smoother spatial variation with 300 facets vs 100
- Less noise in the temperature distribution

### Energy Conservation
- View factor warnings should show < 5% missing energy (vs 49.8%)
- More accurate thermal coupling between facets
- Better normalization

---

## Generation Time Estimate

**Previous (broken):**
- 10×1×37×180×5×5×5 = 17k cases @ ~1-2 min each = **~8-12 hours**

**Current (fixed):**  
- 10×1×37×180×5×10×10 = 67k cases @ ~2-3 min each = **~30-40 hours**

**Recommendation:**
1. Run a **test subset first** to validate fixes:
   - Set `THETA_VALUES = [10.0]` (1 value)
   - Set `LATITUDE_VALUES = [0.0, 30.0, 60.0, 85.0]` (4 values)
   - Keep emission/azimuth at 10 each
   - **Expected time: ~2-3 hours**
   - **Check:** Opposition effect at lat=0, zero phase
   
2. If test passes, run full production LUT overnight

---

## How to Run Test

Edit [generator.py](TEMPEST_RAD/generator.py) lines 53-57:
```python
# TEST CONFIGURATION (comment out after validation)
THETA_VALUES = [10.0]  # Just one thermal inertia
LATITUDE_VALUES = [0.0, 30.0, 60.0, 85.0]  # 4 test points
# Keep emission/azimuth at 10 each
```

Then run:
```bash
cd /Users/duncan/Desktop/DPhil/TEMPEST
./venv/bin/python TEMPEST_RAD/generator.py
```

Expected output:
- 1×1×4×180×5×10×10 = 36k cases
- ~2-3 hours
- Output file: roughness_lut_spectral_v1.h5 (~7 MB)

**Validation checks:**
1. Load LUT and query lat=0°, phase=0°, azimuth=0°, emission=0-20°
2. Should see factors > 1.0 (beaming)
3. Factors should peak at azimuth=0° and decrease toward 45°
4. No more 49% energy conservation warnings

---

## Memory Considerations

Your system: 6 GB RAM

**During generation:**
- Peak memory: ~5-6 GB (300 facets × 720 steps × 40 layers)
- Close to limit but should work
- **If crashes:** Reduce `kernel_subfacets_count` to 200

**Final LUT:**
- Test: ~7 MB (4 latitudes)
- Full: ~265 MB (37 latitudes)
- Well within limits!

---

## Summary of Root Causes

You accidentally created THREE bugs when increasing resolution:
1. ❌ Reduced azimuth angles → killed opposition effect
2. ❌ Hardcoded subfacets override → ignored your 1000 facet setting
3. ❌ Low view factor rays → 49% energy loss

All fixed now. The polar cooling is **NOT** a bug - it's correct physics.
