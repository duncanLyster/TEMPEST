
import numpy as np
import h5py
import matplotlib.pyplot as plt

def planck_function(wavelength_microns, temp_k):
    if temp_k < 1e-9: return 0.0
    c1 = 1.191042e8  
    c2 = 1.4387752e4 
    return c1 / (wavelength_microns**5 * (np.exp(c2 / (wavelength_microns * temp_k)) - 1))

def verify_limb_brightening():
    print("--- Verifying Limb Brightening ---")
    
    # Load LUT
    f = h5py.File("roughness_lut_spectral_v1.h5", "r")
    lut_tensor = f['lut'][:]
    theta_axis = f['theta'][:]
    wave_axis = f['wavelength'][:]
    emi_axis = f['emission'][:] # 0 to 89
    
    # Indices
    # Tensor: (Theta, OpenAngle, Lat, Time, Wave, Emi, Azi)
    # Theta=20.0 (Index 4)
    # OpenAngle=90 (Index 0)
    # Latitude=0.0 (Index 0)
    # Time=Noon. 
    #   Noon is usually index where Sun is overhead.
    #   In generator, Time 0 is Midnight? Or Noon?
    #   Let's check insolation. Max insolation is noon.
    
    # Let's find the time index for Noon at Lat=0
    # Or just use the peak temperature time.
    
    # We don't have the temps here, they were inside generator.
    # But we can look at the LUT corrections to see if they make sense.
    # At Noon (Phase 0), Source and Observer are aligned.
    # So if we look at Emission angle E, we are looking from the Sun direction (if E=0) or we are looking from angle E, but phase is 0?
    # Wait, Phase 0 (Sun behind Observer) means Obs Vector = Sun Vector.
    # So if Emission = E, then Sun Zeniths angle must also be E.
    # This implies we are looking at a point on the surface where the Sun is at Zenith Angle E.
    # This corresponds to "Local Time = Noon" but "Latitude = E" ? 
    # NO.
    
    # Phase 0 Analysis:
    # We look at the WHOLE DISK.
    # Center of disk: Emission=0. Solar Incidence=0. (Sub-solar point).
    # Limb of disk: Emission=90. Solar Incidence=90.
    # So Phase 0 means Incidence Angle = Emission Angle everywhere.
    
    # In the LUT:
    # "Time" axis corresponds to "Sun Phase" or "Rotation Angle".
    # Time runs from 0 to 360 degrees (0 to 24h).
    # Noon (Sub Solar) is likely at Time index such that Sun is at Zenith (or close).
    
    # We need to simulate the "Smooth" temperature at these points.
    # T_smooth(Incidence) ~ T_ss * cos(Incidence)^0.25 (approx, neglecting inertia).
    # With inertia Theta=20, there is lag, but at Noon it's close.
    
    # Let's approximate:
    # We want to compare T_rough vs T_smooth as a function of Angle from Sub-solar point (which is Emission Angle at Phase 0).
    
    # 1. Select a Facet at the Equator (Lat=0).
    # 2. Iterate "Time" to simulate moving away from Noon.
    #    Time=Noon -> Inc=0.
    #    Time=6h -> Inc=90.
    #    So "Time" acts as our "Emission/Incidence" coordinate for Phase 0 check.
    
    # 3. For each Time t (where Inc = E):
    #    L_smooth = Planck(T_smooth(t))
    #    L_rough = L_smooth * Ratio(t, Wave, E, Azi?)
    #    At Phase 0, Azimuth is irrelevant (symmetric)? Or 0?
    #    If Obs and Sun are aligned, Azimuth between them is 0.
    
    # We need T_smooth(t). The LUT doesn't store T_smooth.
    # But generator.py saves 'temperatures.csv' in output/retrieval_analysis/ if we ran it.
    # Or we can just trust the ratios.
    
    # Let's look at the RATIOS in the LUT for Theta=20, Lat=0.
    # For a given spectral band (e.g., 8um):
    # Check Ratio vs Emission Angle (where Solar Zenith = Emission Angle).
    
    # Wait, the LUT axes are:
    # (Theta, Angle, Lat, Time, Wave, Em, Az)
    # We need to pick the "Time" index that corresponds to "Incidence = Emission".
    # If Emission is the VIEW angle.
    # And Time sets the SUN angle.
    # At Phase 0, View Vector = Sun Vector.
    # So Incidence = Emission.
    # We need to find which "Time" corresponds to which "Incidence".
    
    # In generator:
    # sun_vectors[t] computed.
    # We need to reconstruct the sun vectors to map Time -> Incidence.
    
    theta_idx = 4 # Theta=20
    lat_idx = 0 # Lat=0
    wave_idx = 1 # 8.0um
    
    sim_steps = 90
    
    # Reconstruct Sun Vectors for Lat 0
    sun_vectors = []
    sun_declination = 0.0
    for t in range(sim_steps):
        # Time 0 to 90
        # In generator: angle = 2*pi*t / n_steps.
        # This covers 0 to 360 degrees.
        angle = (2 * np.pi * t) / sim_steps
        # x = cos(-angle), y = sin(-angle), z = 0
        # Normal is (1,0,0) (Equator at noon?). No, Normal at Lat 0 is (1,0,0).
        # Wait, in generator:
        # normal = (cos(lat), 0, sin(lat)) -> (1,0,0) for Lat 0.
        # Sun Vector:
        # t=0 -> angle=0 -> x=1, y=0. dot(n, s) = 1. -> NOON.
        # t=sim_steps/2 -> angle=pi -> x=-1. -> MIDNIGHT.
        
        # So Index t maps to Incidence Angle:
        # cos_inc = cos(angle).
        # inc = |angle|.
        
        sun_vectors.append(angle) # radians
        
    print(f"Theta={theta_axis[theta_idx]}, Lat={0.0}, Wave={wave_axis[wave_idx]}um")
    print(f"{'TimeIdx':<8} {'Inc(deg)':<10} {'Emi(deg)':<10} {'Azi':<5} {'Ratio':<10}")
    
    # We want Phase 0. So Obs = Sun. 
    # Incidence = Emission.
    # Azimuth = 0 (Obs aligned with Sun).
    
    # We iterate Emission Angle E from 0 to 80.
    # We find Time T such that Incidence(T) ~ E.
    # We check Ratio(T, Wave, E, Azi=0).
    
    inc_angles = []
    ratios = []
    
    for i_e, emi in enumerate(emi_axis):
        if emi > 85: continue
        
        # Find Time index where Incidence ~ Emi
        # cos(inc) = cos(angle)
        # angle = radians(emi)
        # t = angle * sim_steps / (2*pi)
        
        target_angle = np.radians(emi)
        t_idx = int(target_angle * sim_steps / (2 * np.pi))
        
        # Check Azimuth 0
        azi_idx = 0 # Azimuth 0
        
        ratio = lut_tensor[theta_idx, 0, lat_idx, t_idx, wave_idx, i_e, azi_idx]
        
        inc_angles.append(emi)
        ratios.append(ratio)
        
        print(f"{t_idx:<8} {emi:<10.1f} {emi:<10.1f} {0:<5} {ratio:.4f}")
        
    # Validation
    # If Beaming exists, Ratio should be > 1.0 (Rough > Smooth).
    # If Limb Brightening exists (relative to Smooth), the Ratio should INCREASE as Emission increases?
    # Or at least stay high.
    
    # Smooth decreases as cos(E)^0.25 (approx).
    # Rough decreases slower?
    # If Ratio > 1, Rough is brighter.
    
    # Let's check the trend.
    
    if np.any(np.array(ratios) < 1.0):
        print("FAIL: Some ratios < 1.0 at Phase 0.")
    else:
        print("PASS: Ratios > 1.0 (Beaming).")

    if ratios[-1] > ratios[0]:
        print("Limb Brightening signature found (Ratio increases with Emission).")
    else:
        print("Ratio decreases with Emission (No relative Limb Brightening).")

if __name__ == "__main__":
    verify_limb_brightening()
