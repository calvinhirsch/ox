use vulkano::command_buffer::BufferCopy;
use crate::renderer::component::DataComponent;
use crate::renderer::component::voxel::data::VoxelTypeIDs;

pub mod data;
pub mod lod;


pub struct VoxelData {

}

pub struct LODUpdateRegions {
    pub bitmask_updated_regions: Vec<BufferCopy>,
    pub voxel_id_updated_regions: Option<Vec<BufferCopy>>,
}

impl VoxelData {
    pub fn set_updated_regions(&mut self, regions: Vec<Vec<Option<LODUpdateRegions>>>) {

    }
}