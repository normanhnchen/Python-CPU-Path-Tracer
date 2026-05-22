"""
Path tracing functions for rendering the scene.

Includes intersection tests and the main path tracing loop.
"""


import numpy as np
from numba import njit

from src.pbr import calculate_pbr
import src.math_utils as math_utils
import src.settings as settings


BACKGROUND_COLOR = np.array(settings.background_color)

# Used to return a missed 3d vector.
INF_VEC = np.full(3, np.inf)


@njit(cache=True)
def ray_bound_intersect(ray_origin, ray_dir, bounding_box):
    """
    Axis-aligned bounding box (AABB) ray-intersection test using the slab method.

    Parameters
    ----------
    ray_origin : np.ndarray
        The ray origin coords, shape (3,).
    ray_dir : np.ndarray
        The ray direction vector, shape (3,).
    bounding_box : BoundingBox
        Numba jitclass with the minimum and maximum bounding box coords.

    Returns
    -------
    bool
        True if the ray intersects the bounding box.
    
    Notes
    -----
    See docs/derivations.md section "Ray-Bounding Box Intersection (The Slab Method)" for the derivation.

    References
    ----------
    https://tavianator.com/2022/ray_box_boundary.html
    """

    inv_ray_dir = 1 / (ray_dir + math_utils.EPSILON)

    # Expand the bounding box slightly to handle flat objects that have no thickness.
    # Calculate intersection distances to the bounding box.
    distance_1 = (bounding_box.min - ray_origin - math_utils.EPSILON) * inv_ray_dir
    distance_2 = (bounding_box.max - ray_origin + math_utils.EPSILON) * inv_ray_dir

    # Find enter and exit intersection distances.
    distance_min = np.max(np.minimum(distance_1, distance_2))
    distance_max = np.min(np.maximum(distance_1, distance_2))

    # True if ray intersects bounding box.
    return distance_min < distance_max


@njit(cache=True)
def ray_triangle_intersect(ray_origin, ray_dir, triangle_coords):
    """
    Möller-Trumbore ray-intersection algorithm.
    
    Parameters
    ----------
    ray_origin : np.ndarray
        The ray origin coords, shape (3,).
    ray_dir : np.ndarray
        The ray direction vector, shape (3,).
    triangle_coords : np.ndarray
        Triangle vertex coordinates, shape (3, 3).

    Returns
    -------
    t : float
        Distance from the ray origin to the triangle intersection.
        Np.inf if no intersection occurs.
    normal : np.ndarray
        Surface normal vector of the triangle, shape (3,).
        INF_VEC if no intersection occurs, shape (3,).

    Notes
    -----
    See docs/derivations.md section "Ray-Triangle Intersection (Möller-Trumbore)" for the derivation.

    References
    ----------
    https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates.html
    https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
    https://en.wikipedia.org/wiki/Cramer%27s_rule
    """

    MISSED_RETURN = np.inf, INF_VEC

    vertex_1, vertex_2, vertex_3 = triangle_coords

    # Store common calculations for reuse.
    edge_1 = vertex_2 - vertex_1
    edge_2 = vertex_3 - vertex_1
    t_vec = ray_origin - vertex_1
    p_vec = np.cross(ray_dir, edge_2)
    q_vec = np.cross(t_vec, edge_1)

    det = np.dot(p_vec, edge_1)
    # If the determinant (det) is very close to 0, the ray is parallel to the triangle and doesn't intersect.
    if np.abs(det) < math_utils.EPSILON:
        return MISSED_RETURN
    inv_det = 1 / det

    # While computing, check if the point is outside the triangle and misses.

    # The distance from the ray origin to the triangle.
    t = np.dot(edge_2, q_vec) * inv_det
    if t < 0:
        return MISSED_RETURN

    # Barycentric u-coordinate.
    u = np.dot(t_vec, p_vec) * inv_det
    if u < 0 or u > 1:
        return MISSED_RETURN
    
    # Barycentric v-coordinate.
    v = np.dot(ray_dir, q_vec) * inv_det
    if v < 0 or u + v > 1:
        return MISSED_RETURN
    
    # Surface normal of the triangle.
    normal = math_utils.normalize(np.cross(edge_1, edge_2))

    # Flip if the surface normal is inverted.
    if np.dot(normal, ray_dir) > 0:
        normal *= -1

    return t, normal

 
@njit(cache=True)
def geometry_intersect(triangles, ray_origin, ray_dir):
    """
    Compute the closest ray-triangle intersection for a list of triangles.

    Performs bounding box tests and uses the Möller-Trumbore ray-intersection algorithm to find
    the intersection.

    Parameters
    ----------
    triangles : numba.typed.List(Triangle)
        Numba typed list of Triangle numba jitclasses used for intersection tests.
    ray_origin : np.ndarray
        The ray origin coords, shape (3,).
    ray_dir : np.ndarray
        The ray direction vector, shape (3,).

    Returns
    -------
    is_hit : bool
        If the ray hits a triangle.
    closest_dist : float
        The distance to the closest intersected triangle.
    closest_normal : np.ndarray
        The surface normal of the closest intersected triangle, shape (3,).
    """
    
    closest_dist = np.inf
    closest_normal = INF_VEC
    is_hit = False

    for i in range(len(triangles)):
        triangle = triangles[i]
        if ray_bound_intersect(ray_origin, ray_dir, triangle.bounding_box):
            intersect_dist, normal = ray_triangle_intersect(ray_origin, ray_dir, triangle.coords)
            if intersect_dist < closest_dist:
                closest_dist = intersect_dist
                closest_normal = normal
                is_hit = True

    return is_hit, closest_dist, closest_normal


@njit(cache=True)
def find_closest_object(ray_origin, ray_dir, scene):
    """
    Check if the ray hits an object's triangle and if so, check which triangle.

    Parameters
    ----------
    ray_origin : np.ndarray
        The ray origin coords, shape (3,).
    ray_dir : np.ndarray
        The ray direction vector, shape (3,).
    scene : numba.typed.List(SceneObject)
        Numba typed list of SceneObject numba jitclasses.
        Each SceneObject requires data used for intersection tests.

    Returns
    -------
    closest_hit : np.ndarray
        Coords of the closest intersection, shape (3,).
        INF_VEC if no intersection occurs.
    closest_normal : np.ndarray
        Surface normal vector of the closest intersected triangle, shape (3,).
        INF_VEC if no intersection occurs.
    closest_object_idx : int
        Index of the object in the list scene.
        -1 if no intersection occurs.
    """

    closest_dist = np.inf
    closest_hit = INF_VEC
    closest_normal = INF_VEC
    closest_object_idx = -1

    for i in range(len(scene)):
        scene_object = scene[i]
        if ray_bound_intersect(ray_origin, ray_dir, scene_object.bounding_box):
            is_hit, intersect_dist, normal = geometry_intersect(scene_object.triangles, ray_origin, ray_dir)

            if is_hit and intersect_dist < closest_dist:
                closest_dist = intersect_dist
                closest_normal = normal
                closest_object_idx = i
                # Find hit point using ray's formula: P(t) = O + tD.
                closest_hit = ray_origin + closest_dist * ray_dir
    
    return closest_hit, closest_normal, closest_object_idx


@njit(cache=True)
def path_trace(scene, ray_origin, ray_dir, bounces):
    """
    Perform path tracing for a single ray in the scene.

    Traces the ray through the scene, bouncing up to bounces times and accumulating light with
    PBR calculations.

    Parameters
    ----------
    scene : numba.typed.List(SceneObject)
        Numba typed list of SceneObject numba jitclasses.
        Each SceneObject requires data used for intersection tests and PBR calculations.
    ray_origin : np.ndarray
        The ray origin coords, shape (3,).
    ray_dir : np.ndarray
        The ray direction vector, shape (3,).
    bounces : int
        How many times a ray should bounce in the scene.

    Returns
    -------
    np.ndarray
        Total accumulated RGB light for a ray, shape (3,).
    
    References
    ----------
    https://www.youtube.com/watch?v=Qz0KTGYJtUk
    """

    ray_color = np.ones(3)
    accum_light = np.zeros(3)

    # Add 1 because the first ray doesn't count as a bounce.
    for _ in range(bounces+1):
        hit_point, normal, closest_object_idx = find_closest_object(ray_origin, ray_dir, scene)

        if closest_object_idx >= 0:
            closest_object = scene[closest_object_idx]
            # Move ray origin slightly above the surface to prevent self-intersection.
            bounce_ray_origin = hit_point + normal * math_utils.EPSILON
            ray_origin = bounce_ray_origin

            ray_dir, ray_color, accum_light = calculate_pbr(ray_dir, normal, closest_object, ray_color, accum_light)
        
        else:
            # Ray escaped the scene; add the background light.
            accum_light += ray_color * BACKGROUND_COLOR
            break
    
    return accum_light
