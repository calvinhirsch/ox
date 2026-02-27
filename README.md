# Ox
A real-time ray tracing voxel rendering engine.

## Demo

https://github.com/user-attachments/assets/47502fb0-4986-42f0-bd8b-558e2a95b0fc

#### Real time ray tracing (dynamic lighting, reflections, etc.)

https://github.com/user-attachments/assets/0390327a-53e8-4fed-9bb9-9d85f7481e84


#### Large render distance
This is 23^3 top level chunks (of size 64^3), or 2240^3 = 11.2m area

https://github.com/user-attachments/assets/2a6174d3-801e-4ca7-8993-5e554f715acc


#### Level of Detail (LOD) system

https://github.com/user-attachments/assets/6c6f1aa4-19bb-47cb-9e09-679762f11b43


## Setup

Rust: [](https://rustup.rs/)
Vulkan: [](https://github.com/vulkano-rs/vulkano/blob/master/README.md#setup-and-troubleshooting)

To run: `cd example_game` ; `cargo run`

## Getting started

see WALKTHROUGH.md

## Implementation overview

Uses custom ray tracing in a compute shader (i.e., does not use hardware ray tracing) (see shaders/raytrace.comp)

There is a list of voxel type definitions that can easily be modified in Rust that define how
each voxel looks (material properties, whether it emits light, etc.).

Voxels are stored densely on the GPU (no sparse data structures, no meshing) in two formats:
 1. A bitmap (0 for air, 1 for filled in)
 2. A list of IDs (which are u8s) representing which voxel type is present in each spot

There is a level of detail (LOD) system where unit voxels in further away chunks are combined together and rendered as larger voxels.

A chunk loading system runs in parallel with main thread execution and allows you to define your own generation and persisting of chunks.

See walkthrough for more details.

## Limitations
 - Relies on storing all voxel data densely in GPU memory (no meshing, sparse data structures, etc.)
 - Currently, lots of the parameters that can be freely changed in the Rust/CPU code are hardcoded in the GLSL/shader code. You will need to change both if you do something different from example_game.
