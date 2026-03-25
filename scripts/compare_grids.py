#!/usr/bin/env python3
"""Compare BTPM and TEMPEST grid parameters"""
import numpy as np

# Shared physical parameters
ks = 0.137254
rhos = 1190.0
cp0 = 750.0
kappa = ks / (rhos * cp0)
day = 15469.2

print("=" * 60)
print("BTPM GRID")
print("=" * 60)

# BTPM skin depth: sqrt(kappa * P / pi)
ls_btpm = np.sqrt(kappa * day / np.pi)
print(f"Thermal diffusivity kappa = {kappa:.6e} m^2/s")
print(f"Skin depth (BTPM definition) ls = {ls_btpm:.6f} m")

# BTPM spatial grid: m=10, n=5, b=20
m = 10; n = 5; b = 20
dz0 = ls_btpm / m
zmax = ls_btpm * b
print(f"Top layer dz0 = ls/m = {dz0:.6f} m")
print(f"Max depth zmax = ls*b = {zmax:.6f} m = {b} skin depths")

# Generate the BTPM grid
dz = dz0
z_val = 0.0
depths = [0.0]
thicknesses = [dz0]
i = 0
while True:
    i += 1
    dz_new = dz * (1 + 1.0/n)
    z_val = z_val + dz_new
    depths.append(z_val)
    thicknesses.append(dz_new)
    dz = dz_new
    if i >= 2 and z_val >= zmax:
        break

print(f"Number of BTPM layers: {len(depths)}")
print(f"Deepest layer depth: {depths[-1]:.6f} m")
print(f"dz range: {thicknesses[0]:.6f} to {thicknesses[-1]:.6f} m")

# BTPM timestep
F = 0.5
dt_btpm = F * rhos * cp0 * thicknesses[0]**2 / ks
print(f"BTPM dt_min = {dt_btpm:.4f} s")
print(f"BTPM steps per day = {day / dt_btpm:.1f}")

print()
print("=" * 60)
print("TEMPEST GRID")
print("=" * 60)

# TEMPEST skin depth: sqrt(k / (rho * cp * omega))
omega = 2 * np.pi / day
ls_tempest = np.sqrt(ks / (rhos * cp0 * omega))
print(f"Skin depth (TEMPEST definition) = {ls_tempest:.6f} m")
print(f"Ratio BTPM/TEMPEST skin depth = {ls_btpm / ls_tempest:.6f}")

n_layers = 100
dz_tempest = 8 * ls_tempest / n_layers
print(f"Layer thickness = 8*ls/n_layers = {dz_tempest:.6f} m")
total_depth = n_layers * dz_tempest
print(f"Total depth = {total_depth:.6f} m = {total_depth / ls_tempest:.1f} TEMPEST skin depths")
print(f"Total depth = {total_depth / ls_btpm:.1f} BTPM skin depths")

# TEMPEST timestep
timesteps_per_day = 360
dt_tempest = day / timesteps_per_day
print(f"dt = {dt_tempest:.4f} s")

# CFL check
const3 = kappa * dt_tempest / dz_tempest**2
print(f"Fourier number (const3) = {const3:.6f} (must be <= 0.5 for stability)")

print()
print("=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"BTPM skin depth:       {ls_btpm:.6f} m")
print(f"TEMPEST skin depth:    {ls_tempest:.6f} m")
print(f"Ratio:                 {ls_btpm / ls_tempest:.6f}")
print(f"Note: ls_btpm = sqrt(kappa*P/pi), ls_tempest = sqrt(kappa/omega) = sqrt(kappa*P/(2*pi))")
print(f"So ls_btpm = ls_tempest * sqrt(2) = {ls_tempest * np.sqrt(2):.6f}")
print()
print(f"BTPM top layer dz:     {thicknesses[0]:.6f} m")
print(f"TEMPEST layer dz:      {dz_tempest:.6f} m")
print(f"Ratio BTPM/TEMPEST:    {thicknesses[0] / dz_tempest:.2f}")
print()
print(f"BTPM total depth:      {depths[-1]:.4f} m ({depths[-1]/ls_btpm:.1f} BTPM skin depths)")
print(f"TEMPEST total depth:   {total_depth:.4f} m ({total_depth/ls_btpm:.1f} BTPM skin depths)")
print()
print(f"BTPM n_layers:         {len(depths)}")
print(f"TEMPEST n_layers:      {n_layers}")
print()
print(f"BTPM dt:               {dt_btpm:.2f} s ({day/dt_btpm:.0f} steps/day)")
print(f"TEMPEST dt:            {dt_tempest:.2f} s ({timesteps_per_day} steps/day)")
