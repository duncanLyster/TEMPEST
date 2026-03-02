import math
import numpy as np

def test_dist(phi, R, power):
    thetas = [phi * ((i / R) ** power) for i in range(R + 1)]
    print(f"Power: {power}")
    for i in range(1, R+1):
        theta_prev = thetas[i-1]
        theta_curr = thetas[i]
        seg_len = (theta_curr - theta_prev)
        rad = math.sin(theta_curr)
        print(f"Ring {i}: dTheta={seg_len:.3f}, rad={rad:.3f}, Ratio={rad/seg_len if seg_len>0 else 0:.1f}")

test_dist(math.pi/2, 5, 0.5)
test_dist(math.pi/2, 5, 0.75)
test_dist(math.pi/2, 5, 1.0)
