mod gpu_defs;
mod lod;
mod grid;
mod layer_contents;

use cgmath::Vector3;
use gpu_defs::{ChunkVoxelIDs, ChunkBitmask};
use crate::voxel_type::VoxelTypeEnum;
use crate::world::camera::Camera;
use crate::world::mem_grid::{PhysicalMemoryGrid, VirtualMemoryGrid};



struct RenderingLayers {
    lods: Vec<Vec<RenderingLayer>>,
}