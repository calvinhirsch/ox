use cgmath::{Array, Point3, Vector3};
use crate::voxel_type::VoxelTypeEnum;
use crate::world::camera::Camera;
use crate::world::mem_grid::rendering::gpu_defs::ChunkVoxelIDs;
use crate::world::{TLCPos, VoxelPos};

mod rendering;
mod layer;
mod utils;
mod layer_set;
mod layer_contents;


pub trait PhysicalMemoryGrid<VE: VoxelTypeEnum> {
    type Virtual: VirtualMemoryGrid<VE>;

    fn shift_offsets(&mut self, shift: Vector3<i64>);

    fn size(&self) -> usize;

    fn to_virtual(self) -> Self::Virtual;

    fn to_virtual_for_size(self, grid_size: usize) -> Self::Virtual;
}


pub trait VirtualMemoryGrid<VE: VoxelTypeEnum> {
    type Physical: PhysicalMemoryGrid<VE>;

    fn load_or_generate_tlc(&self, voxel_output: &mut ChunkVoxelIDs, tlc: TLCPos<i64>);

    fn reload_all(&mut self);

    fn set_voxel(&mut self, position: VoxelPos<usize>, voxel_type: VE, tlc_size: usize) -> Option<()>;

    fn lock(self) -> Option<Self::Physical>;
}

