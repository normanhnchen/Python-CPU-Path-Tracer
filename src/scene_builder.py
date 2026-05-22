"""
Parses .obj/.mtl files and outputs usable Python data for the path tracer.

Note: model must be triangulated or else faces will not render.
"""


import numpy as np
import numba
from numba import njit
from numba.experimental import jitclass

import src.paths as paths


material_spec = [
    ("base_color", numba.float64[:]),
    ("emissive_strength", numba.float64),
    ("roughness", numba.float64),
    ("metalness", numba.float64),
]


@jitclass(material_spec)
class Material:
    """
    A numba jitclass used to represent an object's PBR material.

    Attributes
    ----------
    base_color : np.ndarray
        The material's base color, shape (3,), dtype float64.
    emissive_strength : float
        The material's emissive strength, dtype float64.
    roughness : float
        The material's roughness, dtype float64.
    metalness : float
        The material's metalness, dtype float64.

    Notes
    -----
    **All arguments must use their respective types for jitclass compilation.**
    """

    def __init__(self, base_color, emissive_strength, roughness, metalness):
        self.base_color = base_color
        self.emissive_strength = emissive_strength
        self.roughness = roughness
        self.metalness = metalness


bounding_box_spec = [
    ("min", numba.float64[:]),
    ("max", numba.float64[:])
]


@jitclass(bounding_box_spec)
class BoundingBox:
    """
    A numba jitclass used to represent a geometry's bounding box.

    Attributes
    ----------
    min : np.ndarray
        The bounding box minimum values, shape (3,), dtype float64.
    max : np.ndarray
        The bounding box maximum values, shape (3,), dtype float64.

    Notes
    -----
    **All arguments must use their respective types for jitclass compilation.**
    """

    def __init__(self, bounds_min, bounds_max):
        self.min = bounds_min
        self.max = bounds_max


triangle_spec = [
    ("coords", numba.float64[:, :]),
    ("bounding_box", BoundingBox.class_type.instance_type)
]


@jitclass(triangle_spec)
class Triangle:
    """
    A numba jitclass used to represent a triangular plane in 3D space.

    Attributes
    ----------
    coords : np.ndarray
        The 3 vertex coords of the triangle, shape (3, 3), dtype float64.
    bounding_box : BoundingBox
        The triangle's bounding box.

    Notes
    -----
    **All arguments must use their respective types for jitclass compilation.**
    """

    def __init__(self, coords, bounding_box):
        self.coords = coords
        self.bounding_box = bounding_box


scene_object_spec = [
    ("triangles", numba.types.ListType(Triangle.class_type.instance_type)),
    ("surface_area", numba.float64),
    ("bounding_box", BoundingBox.class_type.instance_type),
    ("material", Material.class_type.instance_type)
]


@jitclass(scene_object_spec)
class SceneObject:
    """
    A numba jitclass used to store all data of a scene object.

    Attributes
    ----------
    triangles : numba.typed.List[Triangle]
        A numba list of all triangles in the object.
    surface_area : float
        The total surface area of the object.
    bounding_box : BoundingBox
        The bounding box of the object.
    material : Material
        The object's material.

    Notes
    -----
    **All arguments must use their respective types for jitclass compilation.**
    """

    def __init__(self, triangles, surface_area, bounding_box, material):
        self.triangles = triangles
        self.surface_area = surface_area
        self.bounding_box = bounding_box
        self.material = material


def create_scene_object(triangles, surface_area, material):
    """
    Create a SceneObject with the given data.

    Parameters
    -----------
    triangles : list
        A list of all triangles in the object.
    surface_area : float
        The total surface area of the object.
    material : Material
        The object's material.
    
    Returns
    --------
    SceneObject:
        A SceneObject with all required data.
    """

    all_vertices = np.vstack(triangles)

    object_bounds_min = np.min(all_vertices, axis=0)
    object_bounds_max = np.max(all_vertices, axis=0)
    object_bounding_box = BoundingBox(object_bounds_min, object_bounds_max)

    all_triangles = []
    for triangle in triangles:
        triangle_bounds_min = np.min(triangle, axis=0)
        triangle_bounds_max = np.max(triangle, axis=0)
        triangle_bounding_box = BoundingBox(triangle_bounds_min, triangle_bounds_max)

        all_triangles.append(Triangle(triangle, triangle_bounding_box))
    
    all_triangles = numba.typed.List(all_triangles)
    
    return SceneObject(all_triangles, surface_area, object_bounding_box, material)


@njit(cache=True)
def calculate_triangle_area(triangle_coords):
    """
    Find the area of a triangular plane in 3d space.

    Parameters
    ----------
    triangle_coords : np.ndarray
        The vertex coords of a triangle, shape (3, 3).

    Returns
    -------
    float
        The area of the triangular plane.
    
    Notes
    -----
    Refer to docs/derivations.md section "Triangle Area Calculation" for the derivation.
    """
    
    vertex_1, vertex_2, vertex_3 = triangle_coords

    edge_1 = vertex_2 - vertex_1
    edge_2 = vertex_3 - vertex_1

    return np.linalg.norm(np.cross(edge_1, edge_2)) / 2


def parse_obj(obj_file_path, mtl_file_path):
    """
    Parse the .obj file into usable python data.

    Parameters
    ----------
    obj_file_path : str
        The path of the .obj file.
    mtl_file_path : str
        The path of the .mtl file.
        
    Returns
    -------
    list
        A scene that contains all of the SceneObjects.

    References
    ----------
    https://en.wikipedia.org/wiki/Wavefront_.obj_file
    """

    vertex_coords = []
    triangles = []
    surface_area = np.float64(0)
    scene_objects = []

    # Default material if no .mtl file found.
    if not mtl_file_path:
        material = parse_mtl(None, "")
    else:
        # Default material if no material found.
        material = parse_mtl(mtl_file_path, "")

    with open(obj_file_path) as file:
        for line in file:
            line = line.strip()
            # Create new material.
            if line.startswith("usemtl"):
                if triangles:
                    scene_object = create_scene_object(triangles, surface_area, material)
                    scene_objects.append(scene_object)
                    # Reset for next object.
                    triangles = []
                    surface_area = np.float64(0)
                
                material_name = line[7:]
                if not len(material_name):
                    material_name = ""
                material = parse_mtl(mtl_file_path, material_name)

            # End of object reached; save and reset before switching material/object.
            if line.startswith("o"):
                if triangles:
                    scene_object = create_scene_object(triangles, surface_area, material)
                    scene_objects.append(scene_object)
                    # Reset for next object.
                    triangles = []
                    surface_area = np.float64(0)
            
            elif line.startswith("v "):
                # Vertex data in the form v x y z ...
                coords = list(map(float, line[2:].split()))
                vertex_coords.append(np.array(coords))
            
            elif line.startswith("f"):
                # Face data in the form f vertex_index/texture_index/normal_index ...
                triangle_indices = [data.split("/")[0] for data in line[2:].split()]

                triangle_coords = []
                for vertex_i in triangle_indices:
                    # Obj file indices start at 1.
                    coord_i = int(vertex_i) - 1
                    triangle_coords.append(vertex_coords[coord_i])

                triangle_coords = np.array(triangle_coords)
                surface_area += calculate_triangle_area(triangle_coords)
                triangles.append(triangle_coords)
    
    # Append last object (not previously stored).
    if triangles:
        scene_object = create_scene_object(triangles, surface_area, material)
        scene_objects.append(scene_object)
        # Reset for next object.
        triangles = []
        surface_area = np.float64(0)
    
    return scene_objects


def parse_mtl(file_path, material_name):
    """
    Parse the .mtl file into usable python data.

    Parameters
    ----------
    file_path : str
        The path of the .mtl file.
    material_name : str
        The name of the material, None if the .mtl does not exist.
    
    Returns
    -------
    Material
        A numba jitclass that holds all required material data.

    References
    ----------
    https://en.wikipedia.org/wiki/Wavefront_.obj_file
    """

    found_material = False
    
    # Default material values.
    base_color = np.full(3, 0.8)
    emissive_strength = 0
    roughness = 0.8
    metalness = 0

    # Return default material if no .mtl file found.
    if not file_path:
        return Material(base_color, emissive_strength, roughness, metalness)

    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if line.startswith("newmtl"):
                if found_material:
                    return Material(base_color, emissive_strength, roughness, metalness)
                
                elif line == f"newmtl {material_name}":
                    found_material = True
                
            if found_material:
                if line.startswith("Ns"):
                    # Graph roughness to Ns (shininess) from Blender exports.
                    # https://www.desmos.com/calculator/8ma6dkpwdo
                    shininess = float(line.split()[1])
                    roughness = -np.sqrt(shininess / 1000) + 1
                    roughness = np.clip(roughness, 0, 1)
                
                elif line.startswith("Ka"):
                    # From testing, if illum model is 3, metalness = Ka (ambient reflection).
                    metalness = float(line[3:].split()[0])
                
                elif line.startswith("Kd"):
                    base_color = np.array(list(map(float, line[3:].split())))
                
                elif line.startswith("Ke"):
                    # Ke = emission strength * emissive color (base color).
                    emission = list(map(float, line[3:].split()))
                    # Offset color to prevent divison by 0.
                    emissive_strength = emission[0] / (base_color[0] + 1e-8)
                
                # Illumination model: tells the renderer how to render the material.
                elif line.startswith("illum"):
                    illum_model = int(line.split()[1])
                    # Based on testing, illum model 3 means Ka is the metalness value.
                    if illum_model != 3:
                        metalness = 0

    return Material(base_color, emissive_strength, roughness, metalness)


obj_files_path = paths.OBJ_FILES_DIR

scene = []
for obj_file in obj_files_path.iterdir():
    if obj_file.suffix == ".obj":
        obj_file_path = str(obj_file)

        mtl_file_path = obj_file.with_suffix(".mtl")

        # Check if the .mtl file doesn't exist.
        if not mtl_file_path.is_file():
            mtl_file_path = None

        obj_scene_objects = parse_obj(obj_file_path, mtl_file_path)
        scene.extend(obj_scene_objects)

if not scene:
    print("OBJ_FILES ISSUE: No .obj file found\n")

scene = numba.typed.List(scene)
