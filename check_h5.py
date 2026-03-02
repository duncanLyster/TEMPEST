import numpy as np
import h5py

def run_test():
    try:
        f = h5py.File("roughness_lut_spectral_v1_factors.h5", "r")
        print("v1 shape:", f["roughness_factors"].shape)
        print("v1 max:", np.nanmax(f["roughness_factors"]))
        f.close()
    except Exception as e:
        print(e)
run_test()
