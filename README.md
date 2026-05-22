# A Path Tracer Project

Created by Norman Chen.

This is a CPU-based path tracer written in Python.

## Project Reflection

[docs/personal_reflection.md](docs/personal_reflection.md)

## Features

- Numpy-based vector math and optimizations
- Möller–Trumbore ray–triangle intersection algorithm
- Basic PBR workflow (metalness-roughness)
- .obj scene builder
- Numba njit and jitclass compiling
- CPU python multithreading with Numba parallelization
- Pygame comparison view between noisy and denoised renders

## How To Run

- Add obj and mtl files to main/obj_files as the scene (**model must be triangulated**)
- Configure camera positioning and rotation to your liking  
- Edit path tracing settings in **settings.py**  
- Run the file from the root directory (the parent of main/)
- Run **run.py** where you can choose between render the scene or viewing a comparison slider

## IMPORTANT CONSIDERATIONS

**Please don't expect a perfect path tracer!** Currently, there might be a few lighting bugs here  
and there. However, it should work mostly correct.

**There are many limitations and a lot of features are not implemented.** Notably, textures, PBR  
settings such as transparency, displacement, etc. The renderer only supports .obj models,  
manual camera positioning and rotating, and only supports one emissive because of no irradiance  
caching.

**The speed of the path tracer** depends on the amount of CPU threads and the CPU speed.  
Additionally, geometric complexity (based on the number of objects, triangles, and materials)  
decreases the performance. 

**The quality of the render** output greatly depends on the screen resolution and the number of  
samples. However, increasing the samples a lot will drastically increase the render time. Scenes  
with enclosed areas will have higher quality because rays have a lower chance of scattering  
into the background.

## Hardware Requirements

Intel Open Image Denoise hardware requirements can be found at:  
https://www.openimagedenoise.org/

Numba hardware requirements can be found at:  
https://numba.pydata.org/numba-doc/dev/user/installing.html

## Possible Future Implementations

- Proper folder structure management
- Settings GUI
- Correct light sampling
- Textures
- Smooth shading using vertex normals
- GPU parallelization
- Transparent/translucent materials
- Irradiance caching
- Depth of field and blur control
- Skyboxes
- Ray marching
- Bounding Volume Hiearchy (BVH)
- More pygame window customizations (resizable window, GUIs, etc.)

## References

https://learnopengl.com/Advanced-Lighting/Gamma-Correction  
https://tavianator.com/2022/ray_box_boundary.html  
https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates.html  
https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html  
https://www.youtube.com/watch?v=Qz0KTGYJtUk  
https://learnopengl.com/PBR/Lighting  
https://www.bluebill.net/2021/vector_reflection.html  
https://graphicscompendium.com/raytracing/11-fresnel-beer  
https://pema.dev/obsidian/math/light-transport/cosine-weighted-sampling.html  
https://en.wikipedia.org/wiki/Wavefront_.obj_file  
https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html
https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
https://en.wikipedia.org/wiki/Cross_product#Geometric_meaning
https://gabrielgambetta.com/computer-graphics-from-scratch/02-basic-raytracing.html
https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html