# utils.py

import numpy as np
from numba import jit, float64, int64
import hashlib
from tqdm import tqdm
from typing import Tuple

def calculate_black_body_temp(insolation, emissivity, albedo):
    """
    Calculate the black body temperature based on insolation and surface properties.
    
    Args:
    insolation (float): Incoming solar radiation in W/m^2
    emissivity (float): Surface emissivity
    albedo (float): Surface albedo
    
    Returns:
    float: Black body temperature in Kelvin
    """
    stefan_boltzmann = 5.67e-8  # Stefan-Boltzmann constant in W/(m^2·K^4)
    
    # Calculate temperature using the Stefan-Boltzmann law
    temperature = (insolation / (emissivity * stefan_boltzmann)) ** 0.25
    
    return temperature

def rotate_vector(vec, axis, angle):
    """
    Rotate a vector 'vec' around an 'axis' by 'angle' radians.
    """
    axis = axis / np.linalg.norm(axis)
    vec_rotated = (vec * np.cos(angle) +
                   np.cross(axis, vec) * np.sin(angle) +
                   axis * np.dot(axis, vec) * (1 - np.cos(angle)))
    return vec_rotated

def conditional_print(silent_mode, message):
    if not silent_mode:
        print(message)

def conditional_tqdm(iterable, silent_mode, **kwargs):
    if silent_mode:
        return iterable
    else:
        return tqdm(iterable, **kwargs)

@jit(nopython=True)
def rays_triangles_intersection(
    ray_origin: np.ndarray, ray_directions: np.ndarray, triangles_vertices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sourced from: https://gist.github.com/V0XNIHILI/87c986441d8debc9cd0e9396580e85f4

    Möller–Trumbore intersection algorithm for calculating whether the ray intersects the triangle
    and for which t-value. Based on: https://github.com/kliment/Printrun/blob/master/printrun/stltool.py,
    which is based on:
    http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    Parameters
    ----------
    ray_origin : np.ndarray(3)
        Origin coordinate (x, y, z) from which the ray is fired
    ray_directions : np.ndarray(n, 3)
        Directions (dx, dy, dz) in which the rays are going
    triangle_vertices : np.ndarray(m, 3, 3)
        3D vertices of multiple triangles
    Returns
    -------
    tuple[np.ndarray<bool>(n, m), np.ndarray(n, m)]
        The first array indicates whether or not there was an intersection, the second array
        contains the t-values of the intersections
    """

    output_shape = (len(ray_directions), len(triangles_vertices))

    all_rays_t = np.full(output_shape, np.nan)
    all_rays_intersected = np.zeros(output_shape, dtype=np.bool_)

    v1 = triangles_vertices[:, 0]
    v2 = triangles_vertices[:, 1]
    v3 = triangles_vertices[:, 2]

    eps = 0.000001

    edge1 = v2 - v1
    edge2 = v3 - v1

    for i, ray in enumerate(ray_directions):
        all_t = np.zeros((len(triangles_vertices)))
        intersected = np.full((len(triangles_vertices)), True)

        pvec = np.cross(ray, edge2)

        det = np.sum(edge1 * pvec, axis=1)

        non_intersecting_original_indices = np.absolute(det) < eps

        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        inv_det = 1.0 / det

        tvec = ray_origin - v1

        u = np.sum(tvec * pvec, axis=1) * inv_det

        non_intersecting_original_indices = (u < 0.0) + (u > 1.0)
        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        qvec = np.cross(tvec, edge1)

        v = np.sum(ray * qvec, axis=1) * inv_det

        non_intersecting_original_indices = (v < 0.0) + (u + v > 1.0)

        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        t = (
            np.sum(
                edge2 * qvec,
                axis=1,
            )
            * inv_det
        )

        non_intersecting_original_indices = t < eps
        all_t[non_intersecting_original_indices] = np.nan
        intersected[non_intersecting_original_indices] = False

        intersecting_original_indices = np.invert(non_intersecting_original_indices)
        all_t[intersecting_original_indices] = t[intersecting_original_indices]

        all_rays_t[i] = all_t
        all_rays_intersected[i] = intersected

    return all_rays_intersected, all_rays_t

@jit(nopython=True)
def normalize_vector(v):
    """Normalize a vector."""
    norm = np.sqrt(np.sum(v**2))
    return v / norm if norm != 0 else v

@jit(nopython=True)
def normalize_vectors(vectors):
    """Normalize an array of vectors."""
    norms = np.sqrt(np.sum(vectors**2, axis=1))
    return vectors / norms[:, np.newaxis]

@jit(nopython=True)
def random_points_in_triangle(v0, v1, v2, n):
    r1 = np.sqrt(np.random.rand(n))
    r2 = np.random.rand(n)
    return (1 - r1)[:, np.newaxis] * v0 + (r1 * (1 - r2))[:, np.newaxis] * v1 + (r1 * r2)[:, np.newaxis] * v2

def get_shape_model_hash(shape_model):
    # Create a hash based on the shape model data
    model_data = []
    for facet in shape_model:
        # Flatten the vertices and concatenate with normal and area
        facet_data = np.concatenate([facet.vertices.flatten(), facet.normal, [facet.area]])
        model_data.append(facet_data)
    
    # Convert to a 2D numpy array
    model_data_array = np.array(model_data)
    
    # Create a hash of the array
    return hashlib.sha256(model_data_array.tobytes()).hexdigest()

def get_view_factors_filename(shape_model_hash):
    return f"view_factors/view_factors_{shape_model_hash}.npz"

def get_visible_facets_filename(shape_model_hash):
    return f"visible_facets/visible_facets_{shape_model_hash}.npz"

# Calculate rotation matrix for the body's rotation
@jit(nopython=True)
def calculate_rotation_matrix(axis, theta):
    '''Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.'''
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
