# Ox
A real-time ray tracing voxel rendering engine.

## Setup

Requires Rust & Vulkan

To run: `cd example_game` ; `cargo run`

## Implementation

Uses custom ray tracing in a compute shader (i.e., does not use hardware ray tracing) (see shaders/raytrace.comp)

There is a list of voxel type definitions that can easily be modified in Rust that define how
each voxel looks (material properties, whether it emits light, etc.).

Voxels are stored densely on the GPU (no sparse data structures, no meshing) in two formats:
 1. A bitmap (0 for air, 1 for filled in)
 2. A list of IDs (which are u8s) representing which voxel type is present in each spot

There is a level of detail (LOD) system where unit voxels in further away chunks are combined together and rendered as larger voxels.

## Limitations
 - Relies on storing all voxel data densely in GPU memory (no meshing, sparse data structures, etc.)
 - Currently, lots of the parameters that can be freely changed in the Rust/CPU code are hardcoded in the GLSL/shader code. You will need to change both if you do something different from example_game.
