#!/src/utilities/tumbling_matrices.py
"""
Compute free-precession rotation matrices for an arbitrary ASCII STL shape,
ensuring a rational number of spin and precession cycles.

This is a rough work in progress and TEMPEST does not currently support tumbling bodies.

Dependencies:
    pip install numpy-stl numpy
"""

import argparse
import numpy as np
from stl import mesh
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def rodrigues(u, theta):
    """Return 3×3 rotation matrix rotating by theta around unit-vector u."""
    u = u / np.linalg.norm(u)
    ux, uy, uz = u
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [c + ux*ux*C,    ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s, c + uy*uy*C,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, c + uz*uz*C   ]
    ])


def compute_tumbling_matrices(timesteps, ratio_im_il, ratio_is_il,
                             spinrate, tilt, cone, max_precessions=3, max_spins=100):
    """Compute free-precession rotation matrices based on provided arguments."""

    # --- Moments of inertia ---
    Ilong   = 1.0
    Imiddle = ratio_im_il * Ilong
    Ishort  = ratio_is_il * Ilong
    Ibody   = np.diag([Ilong, Imiddle, Ishort])
    Iinv    = np.linalg.inv(Ibody) # Constant in body frame

    # --- Initial angular momentum vector in the body frame (before cone tilt) ---
    theta_tilt = np.radians(tilt)
    L_dir = np.array([np.sin(theta_tilt), 0.0, np.cos(theta_tilt)])
    Lmag  = Ishort * spinrate # Magnitude definition based on initial spin around short axis
    L_body0 = Lmag * L_dir

    # --- Precession rate and period ratio r = T_prec/T_spin ---
    # WARNING: This formula's applicability to asymmetric rotors might be limited.
    # It approximates the precession frequency when L is near the short axis.
    omega_prec = abs(Lmag) * abs((1.0/Imiddle) - (1.0/Ishort))
    if omega_prec == 0:
        # Handle cases like sphere (I1=I2=I3) or symmetric rotor tilted along symmetry axis
        print("Warning: Calculated precession rate is zero. Using spin rate only.")
        # Avoid division by zero; assign a very small value, motion will be mostly spin.
        omega_prec = 1e-9
    r = spinrate / omega_prec # Ratio of initial spin rate to precession rate

    # --- Rational approximation search (p spins per q precessions) ---
    best = {'p': None, 'q': None, 'err': np.inf, 'r_approx': None} # Initialize r_approx
    found_valid_approximation = False
    for q in range(1, max_precessions + 1):
        p_float = r * q
        p = int(round(p_float))

        # Basic validity check
        if p < 1:
            print(f"  Skipping q={q}: resultant p={p} < 1")
            continue

        # Check against max_spins constraint
        if p > max_spins:
            print(f"  Skipping q={q}: p={p} exceeds max_spins={max_spins}")
            continue

        # This is a valid candidate within constraints
        found_valid_approximation = True
        r_approx = p / q
        err = abs(r_approx - r)

        # Check if this is the best one found so far
        if err < best['err']:
            best.update(p=p, q=q, err=err, r_approx=r_approx)

    # --- Handle case where no valid approximation was found ---
    if not found_valid_approximation:
        # Construct informative error message
        error_message = (
            f"Could not find a suitable rational approximation (p spins / q precessions) "
            f"within the specified limits (max_precessions={max_precessions}, max_spins={max_spins}).\n"
            f"Target ratio r = spinrate / omega_prec = {r:.4f}.\n"
            f"Consider increasing max_spins or max_precessions, or check input parameters "
            f"(tilt, moment ratios, spinrate)."
        )
        # Try to find the *closest* approximation ignoring max_spins, just for reporting
        closest_p_unconstrained = int(round(r * 1)) # Closest for q=1
        if closest_p_unconstrained >= 1:
             error_message += (f"\nFor q=1, the closest integer number of spins is p={closest_p_unconstrained}, "
                               f"which might exceed max_spins.")

        raise ValueError(error_message)


    # --- Report chosen approximation ---
    print(f"Target ratio r = T_prec/T_spin: {r:.6f}")
    print(f"Using omega_prec = |L| * |1/Im - 1/Is| = {omega_prec:.6f}")
    print(f"Best approx within limits: p={best['p']} spins per q={best['q']} precessions") # Clarified output
    print(f"Number of spins per precession cycle (p): {best['p']}")
    print(f" => approximate ratio: {best['r_approx']:.6f}, error = {best['err']:.6e}")

    # --- Time stepping ---
    # Calculate dt based on the desired number of steps per *approximate* precession cycle
    dt = (2 * np.pi / omega_prec) / timesteps
    total_steps = timesteps * best['q']
    print(f"Simulating {best['q']} approximate precession cycles.")
    print(f"Total timesteps = {total_steps}, dt = {dt:.6e}")

    # --- Initialize orientation and storage ---
    orientation = np.eye(3) # Represents transformation from body to space frame
    # Apply initial cone opening (rotation around space Y-axis before simulation starts)
    theta_cone = np.radians(cone)
    R_cone = rodrigues(np.array([0,1,0]), theta_cone)
    orientation = R_cone.dot(orientation) # Initial orientation matrix R(t=0)

    # Calculate constant angular momentum vector in space frame
    # L_space = R(t=0) * L_body(t=0)
    L_space = orientation @ L_body0

    rotations = [orientation.copy()] # Store initial orientation R(t=0)

    # --- Time integration loop ---
    for step in range(total_steps):
        # Calculate current angular momentum and velocity in the *body* frame
        L_body = orientation.T.dot(L_space) # L_body(t) = R(t)^T * L_space
        w_body = Iinv.dot(L_body)           # w_body(t) = I_body^-1 * L_body(t)
        w_norm = np.linalg.norm(w_body)

        # Calculate incremental rotation matrix (rotation in body frame over dt)
        # If w_norm is zero, no rotation occurs.
        deltaR = rodrigues(w_body / w_norm, w_norm * dt) if w_norm > 1e-15 else np.eye(3)

        # Update orientation: R(t+dt) = deltaR(t) * R(t)
        orientation = deltaR.dot(orientation)
        rotations.append(orientation.copy()) # Store R(t = (step+1)*dt)

    # --- Finalize ---
    # Return orientations from t=dt to t=total_steps*dt
    # Exclude the initial t=0 orientation stored before the loop
    rotations = np.stack(rotations[1:])  # shape (total_steps, 3, 3)
    print(f"Computed rotation matrix array with shape {rotations.shape}.")
    return rotations

def animate_rotation(shape_file, rotation_matrices, output_file=None, fps=20, skip_frames=1):
    """
    Animate a 3D shape model rotating according to rotation matrices.
    
    Args:
        shape_file: Path to the STL file
        rotation_matrices: Array of rotation matrices (shape: n_steps x 3 x 3)
        output_file: Path to save animation (if None, displays animation)
        fps: Frames per second for animation
        skip_frames: Number of frames to skip (to reduce file size/rendering time)
    """
    # Load the STL file
    shape_mesh = mesh.Mesh.from_file(shape_file)
    
    # Extract vertices and faces
    vertices = shape_mesh.vectors
    
    # Compute center of mass and normalize size
    center = np.mean(vertices.reshape(-1, 3), axis=0)
    max_range = np.max(np.ptp(vertices.reshape(-1, 3), axis=0))
    
    # Set up figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial collection of triangles
    collection = Poly3DCollection(vertices, alpha=0.7, edgecolor='k', linewidth=0.5)
    collection.set_facecolor('lightgray')
    ax.add_collection3d(collection)
    
    # Set axis limits
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    # Add axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Tumbling Shape Model')
    
    # Draw coordinate axes
    axis_length = max_range * 0.8
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', label='Z')
    
    # Store original vertices
    vertices_original = vertices.copy()

    def update(frame):
        # Use skipped frames to speed up animation
        i = frame * skip_frames
        if i >= len(rotation_matrices):
            i = len(rotation_matrices) - 1
            
        # Apply rotation matrix to original vertices
        R = rotation_matrices[i]
        
        # Rotate all vertices
        rotated_vertices = np.zeros_like(vertices_original)
        for j in range(len(vertices_original)):
            for k in range(3):
                # Center, rotate, and uncenter the vertex
                v = vertices_original[j, k] - center
                rotated_v = R @ v
                rotated_vertices[j, k] = rotated_v + center
        
        # Update the collection
        collection.set_verts(rotated_vertices)
        
        # Update title with frame information
        ax.set_title(f'Tumbling Shape Model (Frame {i+1}/{len(rotation_matrices)})')
        
        return collection,

    # Create animation
    n_frames = len(rotation_matrices) // skip_frames
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    # Either save or display the animation
    if output_file:
        print(f"Saving animation to {output_file}...")
        anim.save(output_file, writer='pillow', fps=fps)
        print("Animation saved!")
    else:
        plt.show()
        
    return anim

if __name__ == "__main__":
    # Parameters
    shape_file = "private/data/shape_models/67P_not_to_scale_low_res.stl"
    timesteps = 1000
    ratio_im_il = 5.22
    ratio_is_il = 5.48
    spinrate = 1.0
    tilt = 5.0
    cone = 40.0
    max_precessions = 3
    max_spins_limit = 10

    print("Computing rotation matrices...")
    rotations = compute_tumbling_matrices(
        timesteps=timesteps,
        ratio_im_il=ratio_im_il,
        ratio_is_il=ratio_is_il,
        spinrate=spinrate,
        tilt=tilt,
        cone=cone,
        max_precessions=max_precessions,
        max_spins=max_spins_limit
    )

    # Create animation
    output_file = "private/data/tumbling_animation.gif"
    animate_rotation(
        shape_file=shape_file,
        rotation_matrices=rotations,
        output_file=None,
        fps=15,
        skip_frames=1  # Skip frames to reduce file size
    )

    # Print length of rotations
    print(f"Length of rotations: {len(rotations)}")
    
    print(f"Animation saved to {output_file}")

