use cgmath::Point3;
use crate::voxel_type::VoxelTypeEnum;

pub trait MemoryGridLayerContents<VE: VoxelTypeEnum> {
    fn set_voxel(chunk_neighborhood: [Self; 3*3*3], chunk_pos: Point3<i64>, voxel_type: VE) -> Option<()>;
}