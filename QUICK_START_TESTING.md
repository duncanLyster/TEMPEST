# Quick Start: Testing Radiance Retrieval

## ✅ What's Been Fixed

1. **TEMPEST Hang**: Removed timestep override HACK (was forcing 5000 timesteps = extremely slow)
2. **Config Path**: Updated to `private/data/config/bennu/bennu_config.yaml`
3. **SPICE Kernels**: All downloaded ✓

## 🚨 Critical: Enable Roughness

Your `bennu_config.yaml` has `apply_kernel_based_roughness: false` but radiance retrieval needs it **true**!

**Edit `private/data/config/bennu/bennu_config.yaml`**:
```yaml
apply_kernel_based_roughness: true  # Change from false
```

## Testing Steps

### 1. Run TEMPEST (with roughness enabled)
```bash
python tempest.py --config private/data/config/bennu/bennu_config.yaml
```

**First run will be slow** (10-30 min):
- Calculating view factors (gets cached)
- With 1966 facets + self-heating = takes time
- **This is normal!** Subsequent runs will be fast

**Look for**:
- "Calculating view factors using X parallel jobs"
- Progress messages
- Output: `data/output/remote_outputs/animation_outputs_*/combined_animation_data_rough_*.h5`

### 2. Verify TEMPEST Output
Check HDF5 file contains `subfacet_data/temps_full`:
```bash
python3 -c "import h5py; f=h5py.File('data/output/remote_outputs/animation_outputs_*/combined_animation_data_rough_*.h5','r'); print('Has temps_full:', 'temps_full' in f['subfacet_data'] if 'subfacet_data' in f else False)"
```

### 3. Run Radiance Retrieval
```bash
python retrieve_radiance.py --config private/data/config/radiance_retrievals/Bennu_OTES.yaml
```

**Expected**:
- Loads TEMPEST output ✓
- Loads SPICE kernels ✓
- Calculates geometry ✓
- Finds visible facets/sub-facets ✓
- Calculates radiance ✓
- Saves to `output/radiance_retrievals/radiance_*.npz` ✓

## SPICE Kernels Status

All downloaded:
- ✓ `naif0012.tls` (leap seconds)
- ✓ `pck00010.tpc` (planetary constants)
- ✓ `orx_v14.tf` (OSIRIS-REx frame)
- ✓ `orx_shape_v03.tf` (Bennu surface IDs)
- ✓ `bennu_v17.tpc` (Bennu physical constants)
- ✓ `orx_181203_190302_190104_od085_v1.bsp` (SPK - 28MB)

**Note**: If `BENNU_FIXED` frame doesn't work, try `IAU_BENNU` (SPICE constructs from PCK).

## Troubleshooting

**TEMPEST seems hung?**
- Check CPU usage (should be high during view factor calculation)
- Wait 30+ minutes for first run
- Look for progress messages

**Radiance retrieval fails?**
- Ensure TEMPEST was run with `apply_kernel_based_roughness: true`
- Check that `subfacet_data/temps_full` exists in HDF5 file
- Verify config paths are correct
