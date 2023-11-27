use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;
use super::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::buffers::{BufferScheme, ConstantBuffer, DualBuffer};
use crate::renderer::component::{DataComponent, DataComponentSet};


pub struct VoxelLODUpdateRegions {
    pub bitmask_updated_regions: Vec<BufferCopy>,
    pub voxel_id_updated_regions: Option<Vec<BufferCopy>>,
}

pub struct RendererVoxelLOD {
    pub bitmask_buffers: DataComponent<DualBuffer<VoxelBitmask>>,
    pub voxel_type_id_buffers: Option<DataComponent<DualBuffer<VoxelTypeIDs>>>,
    pub update_regions: VoxelLODUpdateRegions,
}
impl RendererVoxelLOD {
    pub fn new<BMI: Iterator<Item=VoxelBitmask>, VII: Iterator<Item=VoxelTypeIDs>>(
        bitmask_iter: BMI,
        voxel_id_iter: Option<VII>,
        bitmask_binding: u32,
        voxel_id_binding: Option<u32>,
        memory_allocator: Arc<dyn MemoryAllocator>,
    ) -> Self {
        RendererVoxelLOD {
            bitmask_buffers: DataComponent {
                buffer_scheme: DualBuffer::new(bitmask_iter, Arc::clone(&memory_allocator)),
                binding: bitmask_binding,
            },
            voxel_type_id_buffers: match voxel_id_iter {
                None => None,
                Some(iter) => Some(
                    DataComponent {
                        buffer_scheme: DualBuffer::new(iter, memory_allocator),
                        binding: voxel_id_binding.unwrap(),
                    }
                ),
            },
            update_regions: VoxelLODUpdateRegions {
                bitmask_updated_regions: vec![],
                voxel_id_updated_regions: match voxel_id_binding {
                    None => None,
                    Some(_) => vec![]
                }
            },
        }
    }
}

impl DataComponentSet for RendererVoxelLOD {
    fn list_dynamic_components(&self) -> Vec<&DataComponent<DualBuffer<dyn BufferContents>>> {
        match &self.voxel_type_id_buffers {
            None => vec![&self.bitmask_buffers],
            Some(comp) => vec![&self.bitmask_buffers, comp],
        }
    }

    fn list_constant_components(&self) -> Vec<&DataComponent<ConstantBuffer<dyn BufferContents>>> {
        vec![]
    }
}