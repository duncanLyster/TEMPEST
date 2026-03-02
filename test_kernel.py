import numpy as np

def run_test():
    import h5py
    try:
        f = h5py.File("lut_output/roughness_lut_spectral_v2_theta_1.0.h5", "r")
        print(list(f.keys()))
        f.close()
    except Exception as e:
        print(e)
run_test()
