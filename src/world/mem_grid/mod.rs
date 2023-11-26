use cgmath::{Array, Vector3};
use crate::voxel_type::VoxelTypeEnum;
use crate::world::mem_grid::rendering::gpu_defs::ChunkVoxelIDs;
use crate::world::{TLCPos};

pub mod rendering;
mod layer;
mod utils;
mod layer_set;


pub trait PhysicalMemoryGrid<VE: VoxelTypeEnum, V: VirtualMemoryGrid<VE>> {
    fn shift_offsets(&mut self, shift: Vector3<i64>);

    fn size(&self) -> usize;

    fn as_virtual(&mut self) -> Self::Virtual;

    fn as_virtual_for_size(&mut self, grid_size: usize) -> V;
}


pub trait VirtualMemoryGrid<VE: VoxelTypeEnum> {
    fn load_or_generate_tlc(&self, voxel_output: &mut ChunkVoxelIDs, tlc: TLCPos<i64>);

    fn reload_all(&mut self);
}

