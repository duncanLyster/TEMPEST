"""
Roughness Lookup Table (LUT) Handler.

This module manages the loading and interpolation of the 7D High-Fidelity Roughness LUT.
Dimensions: (Theta, OpeningAngle, Latitude, SunPhase, Wavelength, Emission, Azimuth)
"""

import numpy as np
import h5py
import os
from scipy.interpolate import RegularGridInterpolator
from src.utilities.utils import conditional_print

SIGMA_SB = 5.670374419e-8
AU_M = 1.496e11
# Solar luminosity the generator used when it built the LUTs, for files written
# before reference_tss was stored as an attribute.
_GENERATOR_SOLAR_LUMINOSITY = 3.828e26


class RoughnessLUT:
    def __init__(self, lut_path, target_theta=None, target_rms=None, target_wavelength=None,
                 target_tss=None):
        """
        target_tss : subsolar temperature of the body this LUT is being applied to (K).

        R = I_rough/I_smooth is a ratio of Planck functions over subfacets that all
        scale as T = T_ss * g(Theta, geometry). The lambda^5 cancels and every
        exponential is exp(c2 / (lambda T_ss g)), so R depends on the PRODUCT
        lambda*T_ss, not on lambda alone. Matching Theta fixes the diurnal shape but
        not the temperature scale.

        The LUT is built at its own reference T_ss (set by (1-A)S/eps at the
        reference heliocentric distance). Applying it to a body at a different T_ss
        requires reading it at

            lambda_LUT = lambda_body * (T_ss_body / T_ss_LUT)

        so that both sit the same distance into the Wien tail. Omitting this
        under-applies beaming for a colder body: for T_ss = 258 K against the 1 AU
        reference (386 K), by ~8% at 10 um and up to 44% at short wavelengths.

        Pass None to disable the rescale and reproduce the historical behaviour.
        """
        self.lut_path = lut_path
        self.interpolator = None
        self.axes = {}
        self.is_loaded = False
        self.target_tss = target_tss
        self.reference_tss = None
        self.wavelength_scale = 1.0
        # Interpolator for the smooth reference temperature T_ref(lat, sun_phase), if the
        # file has one. Used for the per-facet wavelength rescale in get_correction_factors.
        self.tsmooth_interp = None

        # Load only the subset if targets are provided
        if os.path.exists(lut_path):
            self.load_subset(target_theta, target_rms, target_wavelength)
        else:
            print(f"Warning: Roughness LUT not found at {lut_path}")

    def load_subset(self, target_theta, target_rms, target_wavelength):
        """
        Load a specific subset of the LUT matching the target parameters.
        This saves RAM by avoiding loading the entire 7D tensor.

        Supports two on-disk formats:
          v2_per_theta  – one file per theta, lut shape (lat, time, wave, emi, azi)
          v1 legacy     – combined file, lut shape (theta, angle, lat, time, wave, emi, azi)
        """
        try:
            with h5py.File(self.lut_path, 'r') as f:

                # Detect format
                gen_version = f.attrs.get('generator_version', b'')
                if isinstance(gen_version, bytes):
                    gen_version = gen_version.decode()
                is_per_theta = (gen_version == 'v2_per_theta')

                # Reference subsolar temperature of this LUT. Stored directly by
                # newer generator runs; recomputed from the reference conditions
                # otherwise. Needed for the lambda*T_ss rescale (see __init__).
                if 'reference_tss' in f.attrs:
                    self.reference_tss = float(f.attrs['reference_tss'])
                elif all(k in f.attrs for k in
                         ('reference_albedo', 'reference_emissivity', 'solar_distance_au')):
                    A = float(f.attrs['reference_albedo'])
                    eps = float(f.attrs['reference_emissivity'])
                    au = float(f.attrs['solar_distance_au'])
                    S = _GENERATOR_SOLAR_LUMINOSITY / (4 * np.pi * (au * AU_M) ** 2)
                    self.reference_tss = ((1 - A) * S / (eps * SIGMA_SB)) ** 0.25
                if self.target_tss is not None and self.reference_tss:
                    self.wavelength_scale = float(self.target_tss) / self.reference_tss

                wave_axis = f['wavelength'][:]
                lat_axis  = f['latitude'][:]
                emi_axis  = f['emission'][:]
                azi_axis  = f['azimuth'][:]

                if is_per_theta:
                    # v2 per-theta: lut shape is (lat, time, wave, emi, azi)
                    theta_val = float(f.attrs['theta'])
                    print(f"LUT: v2 per-theta file, Theta={theta_val:.6f}")

                    n_time     = int(f['sun_phase_steps'][()])
                    sun_phases = np.linspace(0, 360, n_time, endpoint=False)
                    # Older files were not sampled at even phases; use the phases they were
                    # actually sampled at.
                    sim_steps = int(f.attrs.get('sim_timesteps', 0))
                    if f.attrs.get('lut_time_sampling', '') != 'every_nth' and sim_steps > 0:
                        sampled = np.linspace(0, sim_steps - 1, n_time, dtype=int)
                        sun_phases = sampled * 360.0 / sim_steps

                    # Wrap the phase axis to 360 deg so the last bin interpolates.
                    if 't_smooth' in f:
                        t_sm = np.asarray(f['t_smooth'][:], dtype=float)
                        ph_ext = np.append(sun_phases, 360.0)
                        t_ext = np.concatenate([t_sm, t_sm[:, :1]], axis=1)
                        self.tsmooth_interp = RegularGridInterpolator(
                            (lat_axis, ph_ext), t_ext,
                            bounds_error=False, fill_value=None)

                    if target_wavelength is not None:
                        idx_wave = np.abs(wave_axis - target_wavelength).argmin()
                        print(f"LUT: Selected Wave={wave_axis[idx_wave]} (Target={target_wavelength})")
                        lut_subset = f['lut'][:, :, idx_wave, :, :]
                        points = (lat_axis, sun_phases, emi_axis, azi_axis)
                        self.spectral_mode = False
                    else:
                        lut_subset = f['lut'][:]
                        points = (lat_axis, sun_phases, wave_axis, emi_axis, azi_axis)
                        self.spectral_mode = True
                        self.axes['wavelength'] = wave_axis

                else:
                    # v1 legacy: lut shape is (theta, angle, lat, time, wave, emi, azi)
                    theta_axis = f['theta'][:]

                    if target_theta is not None:
                        idx_theta = np.abs(theta_axis - target_theta).argmin()
                        print(f"LUT: Selected Theta={theta_axis[idx_theta]} (Target={target_theta})")
                    else:
                        idx_theta = 0

                    if 'opening_angle' in f:
                        rms_axis = f['opening_angle'][:]
                        print("LUT: Using 'opening_angle' axis.")
                    elif 'rms' in f:
                        rms_axis = f['rms'][:]
                        print("LUT: Using 'rms' axis.")
                    else:
                        raise KeyError("LUT missing 'opening_angle' or 'rms' dataset.")

                    if target_rms is not None:
                        idx_rms = np.abs(rms_axis - target_rms).argmin()
                        print(f"LUT: Selected Angle={rms_axis[idx_rms]} (Target={target_rms})")
                    else:
                        idx_rms = 0

                    dset_name = 'correction_factors' if 'correction_factors' in f else 'lut'

                    if target_wavelength is not None:
                        idx_wave = np.abs(wave_axis - target_wavelength).argmin()
                        print(f"LUT: Selected Wave={wave_axis[idx_wave]} (Target={target_wavelength})")
                        lut_subset = f[dset_name][idx_theta, idx_rms, :, :, idx_wave, :, :]
                        n_time = lut_subset.shape[1]
                        sun_phases = np.linspace(0, 360, n_time, endpoint=False)
                        points = (lat_axis, sun_phases, emi_axis, azi_axis)
                        self.spectral_mode = False
                    else:
                        lut_subset = f[dset_name][idx_theta, idx_rms, :, :, :, :, :]
                        n_time = lut_subset.shape[1]
                        sun_phases = np.linspace(0, 360, n_time, endpoint=False)
                        points = (lat_axis, sun_phases, wave_axis, emi_axis, azi_axis)
                        self.spectral_mode = True
                        self.axes['wavelength'] = wave_axis

            # Handle accidental NaNs (a failed (theta, lat) cell) by filling with 1.
            if np.isnan(lut_subset).any():
                lut_subset = np.nan_to_num(lut_subset, nan=1.0)

            # Drop the e = 90 deg row, where R = 0/0. Kept as NaN it would make every
            # query at the last finite node (where EMI_CAP clamps) return NaN.
            # Emission is the second-last axis in both modes.
            i_emi = len(points) - 2
            emi_vals = np.asarray(points[i_emi], dtype=float)
            keep = ~np.isclose(emi_vals, 90.0)
            if not keep.all():
                points = list(points)
                points[i_emi] = emi_vals[keep]
                points = tuple(points)
                lut_subset = np.take(np.asarray(lut_subset, dtype=float),
                                     np.where(keep)[0], axis=i_emi)

            # Interpolate ln R against 1/lambda. R is a ratio of Planck functions, so this
            # is close to linear, whereas linear interpolation in lambda overshoots between
            # nodes by a few per cent at short wavelengths. LUT_INTERP=linear gives the old
            # scheme.
            self.interp_kind = os.environ.get('LUT_INTERP', 'wien')
            if self.interp_kind == 'wien' and self.spectral_mode:
                i_wav = len(points) - 3
                inv_w = 1.0 / np.asarray(points[i_wav], dtype=float)
                order = np.argsort(inv_w)
                pts = list(points)
                pts[i_wav] = inv_w[order]
                lut_subset = np.take(lut_subset, order, axis=i_wav)
                with np.errstate(invalid='ignore', divide='ignore'):
                    log_lut = np.log(np.maximum(np.asarray(lut_subset, dtype=float), 1e-12))
                self.interpolator = RegularGridInterpolator(
                    tuple(pts), log_lut, bounds_error=False, fill_value=None)
                self._wien_axis = True
            else:
                self.interpolator = RegularGridInterpolator(
                    points, lut_subset, bounds_error=False, fill_value=None)
                self._wien_axis = False
            self.is_loaded = True
            
        except Exception as e:
            print(f"Error loading Roughness LUT subset: {e}")
            self.is_loaded = False

    def get_smooth_temperature(self, latitudes, sun_phases):
        """Smooth reference temperature (K) at (lat, sun_phase), or None if the
        LUT file predates the t_smooth dataset."""
        if self.tsmooth_interp is None:
            return None
        lats = np.abs(np.asarray(latitudes, dtype=float))
        phases = np.mod(np.asarray(sun_phases, dtype=float), 360.0)
        return self.tsmooth_interp(np.column_stack((lats, phases)))

    def get_correction_factors(self, latitudes, sun_phases, emissions, azimuths, wavelength=None,
                               facet_temps=None):
        """
        Query the LUT.

        facet_temps : optional per-facet temperatures (K). If given and the file has
            t_smooth, the query wavelength is rescaled per facet,
            lambda_i = wavelength * T_facet_i / T_ref(lat_i, phase_i), because R depends
            on lambda * T. Otherwise the global target_tss rescale is used.
        """
        if not self.is_loaded:
            return np.ones_like(emissions)

        # Wrap phases
        sun_phases = np.mod(sun_phases, 360.0)
        
        # Mirror Azimuths
        azimuths = np.abs(azimuths)
        azimuths = np.where(azimuths > 180, 360 - azimuths, azimuths)
        
        # Clip Latitudes (just in case)
        latitudes = np.abs(latitudes)
        
        # Cap the emission angle. R = I_rough / I_smooth is ill-conditioned near the limb
        # (cos e -> 0). For an isothermal crater R should be 1 at every angle; with binary
        # (centroid) visibility that test converges with sub-facet count only up to about
        # 80 deg, hence the default. LUTs made with partial visibility pass it to within
        # 2% out to 87.5 deg and can be queried with LUT_EMI_CAP=87.5.
        EMI_CAP = float(os.environ.get('LUT_EMI_CAP', '80'))
        emissions = np.minimum(emissions, EMI_CAP)
        
        if self.spectral_mode:
            if wavelength is None:
                # Default to first wavelength if none provided
                wavelength = self.axes['wavelength'][0]

            wl_axis = self.axes['wavelength']
            n = len(latitudes)

            if facet_temps is not None and self.tsmooth_interp is not None:
                # Per-facet rescale against the T_ref of the cell being read. The 1 K floor
                # only guards the division: cells that cold store R = 1.
                t_ref = self.tsmooth_interp(np.column_stack((latitudes, sun_phases)))
                scale = np.asarray(facet_temps, dtype=float) / np.maximum(t_ref, 1.0)
                waves = np.clip(wavelength * scale, wl_axis[0], wl_axis[-1])
            else:
                # Global rescale (see __init__)
                wavelength = wavelength * self.wavelength_scale
                if wavelength < wl_axis[0] - 1e-9:
                    if not getattr(self, '_warned_wl_floor', False):
                        print(f"  WARNING: rescaled wavelength {wavelength:.3f} um is below the "
                              f"LUT axis floor {wl_axis[0]:.3f} um; clamping. Regenerate the LUT "
                              f"with a shorter-wavelength node to remove this.")
                        self._warned_wl_floor = True
                    wavelength = float(wl_axis[0])
                waves = np.full(n, wavelength)
            
            # Query: (Lat, Time, Wave, Emi, Azi); the wien axis stores 1/lambda
            wcol = 1.0 / waves if self._wien_axis else waves
            query_points = np.column_stack((
                latitudes, sun_phases, wcol, emissions, azimuths
            ))
        else:
            # Fixed wavelength mode
            # Query: (Lat, Time, Emi, Azi)
            query_points = np.column_stack((
                latitudes, sun_phases, emissions, azimuths
            ))

        vals = self.interpolator(query_points)
        if self.spectral_mode and self._wien_axis:
            vals = np.exp(vals)
        return vals
