import h5py
import numpy as np

with h5py.File('roughness_lut_spectral_v1.h5', 'r') as f:
    lut = f['lut'][:]
    print(f"LUT shape: {lut.shape}")
    print(f"LUT size: {lut.nbytes / 1024**2:.1f} MB")
    
    # Show grid dimensions
    for key in f.keys():
        if key != 'lut':
            data = f[key][:]
            print(f"{key}: {data.shape} = {data}")
