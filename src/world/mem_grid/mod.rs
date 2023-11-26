use cgmath::{Array, Vector3};
use crate::voxel_type::VoxelTypeEnum;
use crate::world::mem_grid::voxel::gpu_defs::ChunkVoxelIDs;
use crate::world::{TLCPos};

pub mod voxel;
pub mod layer;
mod utils;
mod layer_set;


pub trait PhysicalMemoryGrid<V> {
    fn shift_offsets(&mut self, shift: Vector3<i64>);

    fn size(&self) -> usize;

    fn as_virtual(&mut self) -> Self::Virtual { self.as_virtual_for_size(self.size()) }

    fn as_virtual_for_size(&mut self, grid_size: usize) -> V;
}
