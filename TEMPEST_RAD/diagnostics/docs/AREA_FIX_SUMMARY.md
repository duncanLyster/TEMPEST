# Area Mismatch Bug Fix - Summary

## Problem Identified

The roughness LUT was showing correction factors **universally < 1.0**, meaning the rough surface appeared cooler than the smooth reference at **all viewing angles**. This violated expected thermal beaming physics:
- At zero phase angle, rough surfaces should show **beaming effect** (factors > 1.0)
- After energy-conserving normalization, some angles should be warmer, others cooler

## Root Cause

The flat reference facet and crater had mismatched areas:
- **Crater opening area**: 1.0 m² (from canonical mesh normalization)
- **Flat facet area**: 2.0 m² (from original vertex definition)

This 2× mismatch caused several issues in the flux ratio calculation:

### In `generator.py::simulate_crater_diurnal_cycle`:

**Original flat facet vertices (Lines 330-340)**:
```python
flat_v = [
    np.array([0.0, -1.0, -1.0]),
    np.array([0.0, 1.0, -1.0]),
    np.array([0.0, 0.0, 1.0])
]
# This triangle has area = 2.0
```

### Why This Matters:

Even though the aperture area term cancels in the final ratio calculation:
```python
ratio = [aperture * cos_e * sum(rad_sub * area * cos) / sum(area * cos)] / [rad_smooth * aperture * cos_e]
      = sum(rad_sub * area * cos) / [rad_smooth * sum(area * cos)]
```

The **thermal simulations** were comparing surfaces of different sizes:
1. Crater simulation: Unit-sized crater (opening = 1.0)
2. Flat simulation: 2× larger flat surface (area = 2.0)
3. The area mismatch could affect temperature distributions and thermal inertia effects

For a fair comparison, both surfaces must have identical opening/projected area.

## Solution

**Fixed the flat facet vertices to match crater opening area = 1.0**:

```python
# Area scales as s², so s = 1/sqrt(2) gives area = 1.0
scale = 1.0 / np.sqrt(2.0)
flat_v = [
    np.array([0.0, -scale, -scale]),
    np.array([0.0, scale, -scale]),
    np.array([0.0, 0.0, scale])
]
# Verified: area = 1.0000
```

### Verification:
```python
v1_v0 = flat_v[1] - flat_v[0]
v2_v0 = flat_v[2] - flat_v[0]
cross = np.cross(v1_v0, v2_v0)
area = 0.5 * np.linalg.norm(cross)  # = 1.0 ✓
```

## Additional Bug Fixed

Found and fixed an **UnboundLocalError** in debug code (Line 476):
- Variable `i_w` was referenced before the wavelength loop where it's defined
- Removed the premature `i_w == 0` condition from the zero-phase debug check

## Expected Outcomes

After this fix, the LUT should show:

1. **Thermal Beaming**: Correction factors > 1.0 at zero phase angle (emission = 0°, azimuth = 0°)
   - Rough surface concentrates emission toward observer
   - Hot illuminated crater walls visible, shadows hidden

2. **Energy Conservation**: After normalization, distribution of factors around 1.0
   - Some angles > 1.0 (beaming, hot spots)
   - Some angles < 1.0 (limb darkening, cool regions)
   - Daily integrated flux remains conserved

3. **Physical Consistency**: Rough ≠ universally cooler
   - Viewing geometry matters
   - Directional effects dominate

## Files Modified

1. `/Users/duncan/Desktop/DPhil/TEMPEST/TEMPEST_RAD/generator.py`
   - **Lines 330-345**: Fixed flat facet vertices (area 2.0 → 1.0)
   - **Line 476**: Removed undefined `i_w` reference from debug condition

## Next Steps for Verification

Run the LUT generator with the fixed code:

```bash
cd /Users/duncan/Desktop/DPhil/TEMPEST/TEMPEST_RAD
python3 -c "
from generator import main
main()
"
```

Then check the new LUT in `retrieval_analysis.ipynb`:
- Verify correction factors > 1.0 at zero phase
- Check distribution shows both warming and cooling angles
- Confirm mean correction factor ≈ 1.0 (energy conservation)

## Technical Notes

### Why Area Cancellation Isn't Enough:

While the radio calculation shows area cancels:
```python
# Both numerator and denominator scale by aperture_area
ratio = [aperture * A] / [aperture * B] = A / B
```

The **thermal simulations run independently** before ray-tracing:
- Crater thermal model uses physical size from mesh (area = 1.0)
- Flat thermal model uses physical size from facet (area = 2.0)
- Different physical sizes → potentially different temperature distributions
- Ray-trace stage compares these temperatures→ mismatch persists

### Canonical Mesh Normalization:

From `spherical_cap_mesh.py::generate_canonical_spherical_cap` (Lines 164-174):
```python
# Normalize opening area to exactly 1
last_pts = [points[last_start + i] for i in range(N_last)]
opening_area = sum(triangle_area(center, p[i], p[i+1]) for i in range(N_last))
scale = sqrt(1.0 / opening_area)
# Apply scale to all facets
for f in facets:
    f['vertices'] *= scale
    f['area'] *= scale * scale
```

This ensures all craters start with unit opening area, providing a consistent baseline for comparison.

## References

- Thermal beaming: Spencer et al. (1989), Rozitis & Green (2011)
- Energy conservation in statistical roughness: Hapke roughness correction
- View factor theory: Howell's "Thermal Radiation Heat Transfer"
