"""
Rotation-integrated energy closure of a roughness LUT (paper Sec. 3.1 check).

For every (theta, latitude) in a LUT directory this compares

    E_emit = sum_t  eps * sigma * T_ref(t)^4 * f(t) * dt,
             f(t) = sum_lam w_lam * Int R(lam,e,phi,t) cos(e) dOmega / (pi * sum_lam w_lam),
             w_lam = B(lam, T_ref(t)) * dlam

(the energy leaving the kernel aperture over one rotation, built only from what the LUT
stores: R and T_ref) with

    E_abs  = sum_t  sum_i Q_i(t) A_i / A_aperture * dt

(the solar energy absorbed by the kernel sub-facets, including intra-crater scattering),
computed by re-running the generator's insolation step on the same kernel mesh and
view factors.  Absorption is independent of theta, so it is computed once per latitude.

The same comparison for the smooth reference facet (E_emit from T_ref vs its own absorbed
energy) is reported as a baseline: it measures the error of the method itself (36-sample
time quadrature of the stored T_ref, solver convergence), which is common to both.

Usage (env must match the LUT's generation settings; see diagnostics/run_summary.txt):
    LUT_CRATER_SUBFACETS=200 LUT_SIM_TIMESTEPS=144 LUT_LUT_TIMESTEPS=36 LUT_N_SCATTERS=3 \
    LUT_N_JOBS=4 python TEMPEST_RAD/energy_closure_check.py TEMPEST_RAD/lut_ab_thermrefl_off_probe
"""
import glob
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TEMPEST_RAD import generator as G
from src.model.facet import Facet
from src.model.simulation import Simulation, ThermalData
from src.model.insolation import calculate_insolation
from src.model.view_factors import calculate_all_view_factors, calculate_thermal_view_factors
from src.utilities.utils import calculate_rotation_matrix

SIGMA = 5.670374419e-8


def build_kernel(config):
    """Kernel mesh + view factors exactly as generator.main() builds them (VFs from cache)."""
    dummy = Facet(np.array([0.0, 0.0, 1.0]),
                  [np.array([-1, -1, 0]), np.array([1, -1, 0]), np.array([0, 1, 0])], 1, 1, 1, False)
    config.kernel_profile_angle_degrees = G.OPENING_ANGLES[0]
    config.apply_kernel_based_roughness = True
    dummy.generate_spherical_depression(config, Simulation(config))
    mesh = Facet._canonical_subfacet_mesh
    n = len(mesh)
    rot_eq = calculate_rotation_matrix(np.array([0.0, 1.0, 0.0]), np.pi / 2)
    rn = np.array([rot_eq @ np.array(e['normal']) for e in mesh])
    rv = np.array([[rot_eq @ v for v in e['vertices']] for e in mesh])
    areas = np.array([e['area'] for e in mesh], dtype=np.float64)

    shape = [Facet(rn[i], rv[i], 1, 1, 1, False) for i in range(n)]
    td = ThermalData(n, 1, 1, 1, False)
    idx = np.arange(n)
    td.set_visible_facets([np.concatenate([idx[:i], idx[i + 1:]]) for i in range(n)])
    vfs = calculate_all_view_factors(shape, td, config, G.VIEW_FACTOR_RAYS)
    td.set_secondary_radiation_view_factors(vfs)
    tvfs = calculate_thermal_view_factors(shape, td, config)
    return rn, rv, areas, [np.asarray(v, float) for v in vfs], tvfs


def absorbed_energy(lat, config, rn, rv, areas, vfs, tvfs):
    """Rotation-summed absorbed flux per unit aperture area: kernel and smooth reference."""
    sim = Simulation(config)
    sim.timesteps_per_day = G.SIM_TIMESTEPS
    sim.rotation_period_hours = G.ROTATION_PERIOD_HOURS
    lr = np.radians(lat)
    sim.sunlight_direction = np.array([np.cos(lr), 0.0, np.sin(lr)])
    n = len(rn)

    shape = [Facet(rn[i].copy(), rv[i].copy(), sim.timesteps_per_day, sim.max_days, sim.n_layers, False)
             for i in range(n)]
    td = ThermalData(n, sim.timesteps_per_day, sim.n_layers, sim.max_days, False)
    idx = np.arange(n)
    vis = [np.concatenate([idx[:i], idx[i + 1:]]) for i in range(n)]
    for i, f in enumerate(shape):
        f.visible_facets = vis[i]
    td.set_visible_facets(vis)
    td.set_secondary_radiation_view_factors(vfs)
    td = calculate_insolation(td, shape, sim, config)
    e_kernel = np.sum(td.insolation * areas[:, None]) / 1.0      # aperture area = 1

    # smooth reference: same flat facet as simulate_crater_diurnal_cycle
    flat_n = np.array([1.0, 0.0, 0.0])
    u = np.cross(flat_n, [0.0, 0.0, 1.0]); u /= np.linalg.norm(u)
    v = np.cross(flat_n, u)
    s = 1.0 / np.sqrt(2.0)
    flat = Facet(flat_n, [-s * u - s * v, s * u - s * v, s * v],
                 sim.timesteps_per_day, sim.max_days, sim.n_layers, False)
    flat.visible_facets = np.array([], dtype=np.int64)
    sd = ThermalData(1, sim.timesteps_per_day, sim.n_layers, sim.max_days, False)
    sd.set_visible_facets([np.array([], dtype=np.int64)])
    sd.set_secondary_radiation_view_factors([np.array([])])
    sd = calculate_insolation(sd, [flat], sim, config)
    e_smooth = np.sum(sd.insolation[0])
    dt = sim.rotation_period_s / sim.timesteps_per_day
    return e_kernel * dt, e_smooth * dt


def emitted_energy(lut_t, t_ref, wl, emis_deg, azi_deg, eps, period_s, sim_steps, every_nth):
    """Rotation-summed energy leaving the aperture from LUT R and T_ref, and the smooth facet's."""
    em = np.radians(emis_deg); az = np.radians(azi_deg)
    d_em = np.append(np.diff(em), np.diff(em)[-1]); d_az = np.append(np.diff(az), np.diff(az)[-1])
    cse = (np.cos(em) * np.sin(em) * d_em)[:, None] * d_az[None, :] * 2.0   # as generator
    ref_integral = cse.sum()
    dwl = np.gradient(wl)
    n_t = len(t_ref)
    # Phases the samples were taken at (uneven in files made before Oct 2026), integrated
    # with a periodic trapezoid rule
    if every_nth:
        idx = np.arange(n_t) * (sim_steps // n_t)
    else:
        idx = np.linspace(0, sim_steps - 1, n_t, dtype=int)
    t_s = idx * period_s / sim_steps
    g_k = np.zeros(n_t); g_s = np.zeros(n_t)
    for t in range(n_t):
        T = float(t_ref[t])
        if T <= 0 or not np.isfinite(T):
            continue
        w = G.planck_function(wl, T) * dwl if T > 50 else dwl.copy()
        ang = np.array([np.nansum(lut_t[t, i] * cse) for i in range(len(wl))])  # nansum: e=90 row is NaN
        f = np.sum(w * ang) / (ref_integral * np.sum(w))
        g_s[t] = eps * SIGMA * T**4
        g_k[t] = g_s[t] * f
    t_next = np.append(t_s[1:], t_s[0] + period_s)
    seg = t_next - t_s
    trap = lambda g: np.sum(0.5 * (g + np.roll(g, -1)) * seg)
    return trap(g_k), trap(g_s)


def main(lut_dir):
    config = G.ReferenceConfig()
    config.n_jobs = G.N_JOBS
    rn, rv, areas, vfs, tvfs = build_kernel(config)
    rowsum = np.array([v.sum() for v in vfs])
    print(f'kernel: {len(rn)} sub-facets, sum_j F_ij mean {rowsum.mean():.4f} '
          f'(hemisphere analytic 0.5), N_SCATTERS={G.N_SCATTERS_ENV}')

    files = sorted(glob.glob(os.path.join(lut_dir, '*.h5')))
    with h5py.File(files[0]) as h:
        lats = h['latitude'][:]
        eps = float(h.attrs['reference_emissivity'])
    period_s = G.ROTATION_PERIOD_HOURS * 3600.0

    absorbed = {lat: absorbed_energy(lat, config, rn, rv, areas, vfs, tvfs) for lat in lats}

    rows = []
    for fpath in files:
        with h5py.File(fpath) as h:
            theta = float(h.attrs['theta'])
            wl = h['wavelength'][:]; emi = h['emission'][:]; azi = h['azimuth'][:]
            every_nth = h.attrs.get('lut_time_sampling', '') == 'every_nth'
            for il, lat in enumerate(lats):
                ek, es = emitted_energy(h['lut'][il], h['t_smooth'][il], wl, emi, azi, eps, period_s, G.SIM_TIMESTEPS, every_nth)
                ak, as_ = absorbed[lat]
                rows.append((theta, lat, ek, es, ak, as_))
    rows = np.array(rows)
    theta, lat, ek, es, ak, as_ = rows.T
    ok = as_ > 1e-3 * as_.max()          # lat 90: Sun on the horizon all day, nothing absorbed
    kern = ek / ak - 1
    smooth = es / as_ - 1
    rel = (ek / es) / (ak / as_) - 1      # kernel closure relative to the smooth baseline

    print(f'\n{lut_dir}: {len(files)} theta x {len(lats)} lat; '
          f'{(~ok).sum() // len(files)} latitude(s) with no absorbed energy excluded: {np.unique(lat[~ok])}')
    print(f'absorption gain E_abs,kernel/E_abs,smooth: {np.nanmin(ak[ok] / as_[ok]):.4f}-{np.nanmax(ak[ok] / as_[ok]):.4f}')
    print(f'{"":28s}{"median":>9s}{"max |.|":>9s}')
    for name, x in (('kernel  E_emit/E_abs - 1', kern), ('smooth  E_emit/E_abs - 1', smooth),
                    ('kernel rel. to smooth', rel)):
        print(f'{name:28s}{100 * np.median(x[ok]):+8.2f}%{100 * np.max(np.abs(x[ok])):8.2f}%')
    print('per latitude, max over theta of |kernel E_emit/E_abs - 1|  (absorbed energy rel. to equator):')
    eq = np.mean(as_[lat == 0])
    for la in np.unique(lat[ok]):
        m = ok & (lat == la)
        print(f'  lat {la:5.1f}: {100 * np.abs(kern[m]).max():6.2f}%   E_abs,smooth/E_abs,smooth(0) = {np.mean(as_[m]) / eq:.3f}')
    for lmax in (80.0, 85.0):
        m = ok & (lat <= lmax)
        print(f'max over theta and lat <= {lmax:.1f}: kernel {100 * np.abs(kern[m]).max():.2f}%, '
              f'smooth baseline {100 * np.abs(smooth[m]).max():.2f}%')
    print('per theta, max |kernel E_emit/E_abs - 1| over latitudes:')
    for th in np.unique(theta):
        m = ok & (theta == th)
        worst = np.argmax(np.abs(kern[m]))
        print(f'  theta {th:8.4f}: {100 * np.abs(kern[m]).max():5.2f}% (at lat {lat[m][worst]:.1f}),'
              f' smooth baseline max {100 * np.abs(smooth[m]).max():5.2f}%')


if __name__ == '__main__':
    main(sys.argv[1])
