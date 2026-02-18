# Why Rough Surfaces Appear Cooler at High Latitudes

## Your Intuition vs. Reality

**Your intuition:** "Anything casting a shadow is catching sunlight, so should heat up. High latitudes = fewer facets receive MORE sunlight = higher radiance"

**What's missing:** The difference between *integrated absorbed flux* vs. *observed directional radiance*

## The Key: What Does "Radiance Factor < 1.0" Actually Mean?

The LUT stores: `R(λ) = Radiance_Rough(λ) / Radiance_Smooth(λ)`

- **Smooth surface at pole:** Uniform temperature across entire surface, moderate radiance in all directions
- **Rough surface at pole:** Hot spots (sunlit crater walls) + cold spots (shadowed areas/floors)

When R < 1.0, it means **the observer sees lower radiance from rough than smooth**.

## Why This Happens at High Latitudes

### 1. **Viewing Geometry Effect**
At the poles, sun is grazing (low elevation). Consider observer looking down at disk center:

**Smooth surface:**
- Sees entire surface at moderate, uniform temperature
- All surface elements contribute equally

**Rough crateLooks into crater from above:**
- Sunlit walls are on the *far side* (facing away from observer)
- Shadowed walls and floor are on the *near side* (facing toward observer)
- Observer preferentially sees the COLD parts!

### 2. **Energy Conservation Constraint**
The LUT is normalized so that **time-averaged, hemisphere-integrated flux = 1.0** at each latitude.

Both rough and smooth receive the same total energy (by normalization). But:
- **Smooth:** Energy distributed uniformly → moderate temperature everywhere
- **Rough:** Energy concentrated into hot spots, but leaves cold shadows → high variance

The hot spots may emit MORE than smooth, but the cold spots emit MUCH LESS. When integrated over viewing angles, the cold dominates **for certain geometries**.

### 3. **Solid Angle Weighting**
Your statement "fewer facets receive MORE sunlight" is correct for absorbed flux. 

But radiance depends on:
1. Temperature of the surface element
2. Its orientation relative to the observer
3. Its visibility from the observer's viewpoint

At high latitudes:
- Sunlit crater walls emit strongly **but face away from disk-center observers**
- Shadowed areas emit weakly **but face toward disk-center observers**
- Result: Observed radiance is lower than smooth, even though peak temperatures are higher

## Analogy: Craters on the Moon

Consider lunar polar craters viewed from Earth:
- Some crater walls are sunlit (hot, bright)
- Some walls and floors are in permanent shadow (cold, dark)
- **What you see depends on your viewing angle**

If you look straight down into a polar crater:
- You mostly see the shadowed floor (cold, dim)
- The sunlit walls are steep and face away from you

Result: The cratered area appears **darker than a smooth area** at the same latitude, even though some parts of the crater are hotter.

## The Opposite at the Equator (R > 1.0)

At the equator, sun is overhead:
- **Smooth:** Uniform moderate temperature
- **Rough:** Crater walls perpendicular to sunlight heat up strongly
- **Observer looking down:** Sees hot crater walls directly! (thermal beaming)
- Result: R > 1.0 (rough appears warmer)

## Summary

**It's not that rough surfaces are COOLER at poles** (peak temperatures may be higher).

**It's that the OBSERVED RADIANCE is lower** because:
1. The hot parts face away from the observer
2. The cold parts face toward the observer  
3. Geometry matters as much as temperature

The LUT captures this **directional** effect through the emission and azimuth angle dimensions. At zero phase (sun behind observer) and disk center (emission angle ~ 0°), you preferentially see shadowed areas at high latitudes.

---

**Physics Check:** Try querying the LUT at:
- **Lat = 85°, Azimuth = 0°, Emission = 0°:** Should see R < 1.0 (looking into shadows)
- **Lat = 85°, Azimuth = 90°, Emission = 70°:** Might see R > 1.0 (grazing view catches sunlit walls)

The directional dependence is what you're trying to capture!
