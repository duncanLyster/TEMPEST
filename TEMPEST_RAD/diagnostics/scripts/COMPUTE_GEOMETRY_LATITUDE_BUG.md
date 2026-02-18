# COMPUTE_GEOMETRY LATITUDE BUG

## Summary
**CRITICAL BUG #3 DISCOVERED**: `compute_geometry()` calculates latitude from facet **normal direction** instead of facet **center position**, causing 100% of LUT queries to fall outside the valid latitude range [0, 85]°.

## The Bug

In [simulator.py](TEMPEST_RAD/simulator.py), lines 114-115:
```python
sin_lats = np.dot(normals, rot_axis)
latitudes = np.degrees(np.arcsin(np.clip(sin_lats, -1.0, 1.0)))
```

This calculates: **Latitude = arcsin(Normal · RotAxis)**

### For a Spherical Mesh:
- **Normal direction**: Points radially outward from sphere center
- **Most facets**: Have normals nearly aligned/anti-aligned with rotation axis
- **Result**: Most latitudes are ±90° (poles)
- **Very few facets**: Have perpendicular normals (equator)

### Test Results:
```
Visible facets: 640/1280
Latitudes (visible):
  Min:  -90.00°
  Max:  90.00°  
  Mean: 1.69°
  Std:  89.98°

Latitudes outside LUT range [0, 85]:
  640/640 facets (100.0%)
```

**Every single query falls outside the LUT's valid range!**

## The LUT Expectation

The LUT was generated with craters at specific **geographic latitudes** on the body (0°, 30°, 60°, 85° based on typical crater simulations). It expects queries like:

- "What is the roughness correction for a crater at 30° geographic latitude?"
- NOT: "What is the correction for a facet whose normal points 30° from the rotation axis?"

## Impact

When `RoughnessLUT.get_correction_factors()` receives latitudes of ±90°, the `RegularGridInterpolator` must **extrapolate** far outside the [0, 85]° data range, producing garbage values (factors ~0.3-0.7 instead of expected ~0.7-1.8).

This explains why the rough model appears cooler everywhere - we're querying the LUT at completely wrong locations!

## The Fix

Calculate latitude from **facet center position**, not normal direction:

```python
# OLD (WRONG):
sin_lats = np.dot(normals, rot_axis)
latitudes = np.degrees(np.arcsin(np.clip(sin_lats, -1.0, 1.0)))

# NEW (CORRECT):
centers = np.array([f.center for f in facets])
# Normalize center positions to unit sphere
center_norms = np.linalg.norm(centers, axis=1)
centers_unit = centers / center_norms[:, np.newaxis]
# Latitude = arcsin(position · rotation_axis)
sin_lats = np.dot(centers_unit, rot_axis)
latitudes = np.degrees(np.arcsin(np.clip(sin_lats, -1.0, 1.0)))
```

For a sphere, this gives a proper distribution of latitudes matching the body's geometry.

## Testing

After fix, latitudes should span the full [0, 90]° range (or [-90, 90]° if using signed latitudes), with a reasonable distribution reflecting the shape model geometry.

## Related Issues  

This is the **third bug** in the geometry calculation:
1. ✓ **FIXED**: LUT time axis misalignment (peak at 268° instead of 0°)
2. ✓ **FIXED** (notebook only): `sun_phases` returning per-facet local times instead of global body phase
3. → **CURRENT**: `latitudes` calculated from normal direction instead of center position
