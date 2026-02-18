# LUT Latitude Resolution Issue - Summary & Solution

## Problem Identified

**Current LUT has only 4 latitude points: [0, 30, 60, 85]°**

This causes:
1. **30° gaps** in latitude coverage
2. **Linear interpolation artifacts** - physics changes nonlinearly with latitude
3. **Systematic cold bias at poles** - factors drop from 1.16 at 0° to 0.38 at 85°

### Is the Physics Correct?

**YES** - The low factors at polar latitudes are **physically realistic**:

- **Equatorial craters** (lat=0-30°): Sun overhead → thermal beaming works → factors > 1.0 ✓
- **Polar craters** (lat>60°): Sun grazing → deep shadows dominate → factors < 1.0 ✓

At high latitudes, rough surfaces have **more shadowing** than smooth surfaces at the same temperature, so rough actually appears **cooler** - this is correct!

### What's Wrong Then?

With only 4 points, the interpolator:
- Makes **linear estimates** in 30° gaps where physics changes rapidly
- Cannot capture the **beaming→shadowing transition** smoothly
- May **underestimate** values between grid points

Your notebook shows:
- **45-60° facets**: mean factor = 0.81 (interpolated in 30-60° gap)
- **60-75° facets**: mean factor = 0.62 (interpolated in 60-85° gap)

These might be slightly wrong due to coarse sampling.

## Solution: Increase Latitude Resolution

### Memory Analysis

Current LUT dimensions:
- Theta: 5, Angle: 1, **Lat: 4**, Time: 90, Wave: 5, Emi: 10, Azi: 10
- **Total: 900K elements = 6.9 MB**

Proposed resolutions:

| Configuration | N_lat | Gap | Memory | Use Case |
|--------------|-------|-----|--------|----------|
| **Low-res** | 7 | 15° | 12 MB | Quick testing |
| **Medium-res** | 13 | 7.5° | 22 MB | Good balance |
| **High-res** ✓ | **19** | **5°** | **33 MB** | **Recommended** |
| Very high-res | 37 | 2.5° | 64 MB | Ultra-fidelity |

### Why 19 Latitudes (5° steps)?

1. **Captures physics smoothly**: 5° gaps minimize interpolation error
2. **Trivial memory cost**: Only 33 MB (vs 7 MB currently)
3. **Fast generation**: ~2-3GB peak RAM during generation (safe for 6GB system)
4. **Industry standard**: 5° is typical for planetary thermal models

### Implementation

In [generator.py](TEMPEST_RAD/generator.py), line 57:

**Current:**
```python
LATITUDE_VALUES = [0.0, 30.0, 60.0, 85.0]
```

**Recommended:**
```python
LATITUDE_VALUES = np.arange(0.0, 95.0, 5.0)  # 0, 5, 10, ..., 85, 90
```

or explicitly:
```python
LATITUDE_VALUES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 
                   45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
```

### Generation Time

- Current (4 lats × 5 thetas × 1 angle): **20 cases**
- Proposed (19 lats × 5 thetas × 1 angle): **95 cases**

If each case takes ~2 minutes: **~3 hours total** (reasonable overnight run)

## Expected Results After Regeneration

1. **Smoother spatial distribution**: No abrupt changes
2. **More accurate factors**: Better capture of transition zone (40-70°)
3. **Polar regions still cool**: Physics is correct - this won't change
4. **Overall**: Rough will be **warmer at equator, cooler at poles** - exactly as expected!

## What About Polar Cooling?

**This is physically correct!** At polar latitudes:
- Smooth disk-averaged brightness: Already low due to grazing sun
- Rough disk-averaged brightness: Even lower due to shadowing
- Result: `Δ = rough - smooth < 0` at poles is **expected**

Your disk-integrated temperature will depend on:
- **Latitude distribution of visible facets**
- **Area weighting** (poles have fewer facets on a sphere)
- **Mixed equatorial warming + polar cooling**

Net effect should still be **slight warming** overall because equatorial facets dominate the disk-integrated flux.

## Action Items

1. **Edit [generator.py](TEMPEST_RAD/generator.py) line 57** to use 19 latitudes
2. **Regenerate LUT**: `python TEMPEST_RAD/generator.py` (~3 hours)
3. **Verify with [verify_lut_fix.py](verify_lut_fix.py)**: Check factors at grid points
4. **Re-run notebook**: Should see smoother behavior, same overall trends

## Alternative: Normalize Per-Latitude Differently?

Current generator normalizes **time-averaged hemisphere-integrated flux = 1.0 per latitude**.

This is correct! It ensures energy conservation. The fact that some latitudes have factors < 1.0 is **physical**, not a bug.

If you wanted all latitudes to have mean(factor) ≈ 1.0, you'd be:
- Violating energy conservation
- Making polar craters artificially bright
- Hiding real physics

**Keep the current normalization** - it's correct.
