import numpy as np
from numba import jit
from typing import Tuple


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


def calculate_view_factors(subject_vertices, subject_normal, subject_area, test_vertices, test_areas, n_rays):
    '''
    Calculate view factors between subject and test vertices. The view factors are calculated by firing rays from the subject vertices to the test vertices and checking if the rays intersect with the test vertices. The view factor is calculated as the number of rays that intersect with the test vertices divided by the total number of rays fired from the subject vertices.
    '''
    # Generate ray sources. These are random points on the face of the subject facet
    def random_points_in_triangle(v0, v1, v2, n):
        r1 = np.sqrt(np.random.rand(n))
        r2 = np.random.rand(n)
        points = (1 - r1)[:, None] * v0 + (r1 * (1 - r2))[:, None] * v1 + (r1 * r2)[:, None] * v2
        return points
    
    ray_sources = random_points_in_triangle(subject_vertices[0], subject_vertices[1], subject_vertices[2], n_rays)
    
    # Generate random ray directions, then eliminate the ones that are not in the same hemisphere as the normal of the subject facet
    ray_directions = np.random.randn(n_rays, 3)
    ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]  # Normalize directions
    valid_directions = np.dot(ray_directions, subject_normal) > 0
    ray_directions = ray_directions[valid_directions]
    
    # Ensure we have exactly n_rays valid directions
    while len(ray_directions) < n_rays:
        additional_rays_needed = n_rays - len(ray_directions)
        additional_ray_directions = np.random.randn(additional_rays_needed, 3)
        additional_ray_directions /= np.linalg.norm(additional_ray_directions, axis=1)[:, np.newaxis]
        additional_valid_directions = np.dot(additional_ray_directions, subject_normal) > 0
        ray_directions = np.vstack((ray_directions, additional_ray_directions[additional_valid_directions]))
    
    ray_directions = ray_directions[:n_rays]
    
    # Fire n_rays and check for intersection using the rays_triangles_intersection function
    intersections = np.zeros((n_rays, len(test_vertices)), dtype=np.bool_)
    for i in range(n_rays):
        ray_origin = ray_sources[i]
        ray_dir = ray_directions[i]
        intersect, _ = rays_triangles_intersection(ray_origin, ray_dir[np.newaxis, :], test_vertices)
        intersections[i] = intersect[0]
    
    # Calculate view factors
    view_factors = np.sum(intersections, axis=0) / n_rays * subject_area / test_areas
    
    # Use the reciprocal rule to calculate the view factors from the test vertices to the subject vertices
    view_factors = view_factors * (test_areas / subject_area)
    
    return view_factors


# Testing values
subject_vertices = np.array([[-13.9827, -33.83933, 54.99486], [-10.19833, -41.17583, 55.56557], [-7.344019, -39.87791, 56.28316]])
subject_normal = np.array([-0.2260779,  -0.04090781,  0.9732499])
subject_area = 13.281515718706087
test_vertices =np.array([
    [[-12.0, -40.0, 55.0], [-15.0, -33.0, 54.0], [-8.0, -38.0, 57.0]],
    [[-11.0, -42.0, 56.0], [-14.0, -34.0, 55.0], [-6.0, -37.0, 58.0]],
    [[-13.0, -39.0, 57.0], [-16.0, -35.0, 56.0], [-9.0, -36.0, 59.0]]
])
test_areas = np.array([14.510972568252221, 14.510972568252221, 14.510972568252221])
n_rays = 10000

# Calculate view factors
view_factors = calculate_view_factors(subject_vertices, subject_normal, subject_area, test_vertices, test_areas, n_rays)

print(f"View Factors: {view_factors}")