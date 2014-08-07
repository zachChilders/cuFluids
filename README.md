cuFluids
========

3D fluid solver built with OpenGL and CUDA.

It uses Smooth Particle Hydrodynamics to get real time fluid simulations.

It consists of two major parts:  the OpenGL renderer and the CUDA solver.

Renderer
========
The renderer consists of a Particle class being passed through the OpenGL pipeline.
Each particle can be instanced using a geometry shader, and then passed back into the pipeline 
using a Transform Feedback Buffer.  Velocities are applied via the vertex shader, which means as 
much work as is possible is done on the GPU.  This will hook nicely into the CUDA solver, because CUDA
will export a velocity volume as a 3D texture.  This texture can mesh nicely into the shader pipeline, 
allowing real time feedback.
