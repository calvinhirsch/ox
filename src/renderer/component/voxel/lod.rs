use std::sync::Arc;
use syn::Data;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;
use super::data::{BlockBitmask, VoxelTypeIDs};
use crate::renderer::buffers::DualBuffer;
use crate::renderer::component::DataComponent;


pub struct RenderingLOD {
    pub bitmask_buffers: DataComponent<DualBuffer<BlockBitmask>>,
    pub voxel_type_id_buffers: Option<DataComponent<DualBuffer<VoxelTypeIDs>>>,
    pub bitmask_updated_regions: Vec<BufferCopy>,
    pub voxel_type_id_updated_regions: Option<Vec<BufferCopy>>,
}
impl RenderingLOD {
    pub fn new<BMI: Iterator<Item = BlockBitmask>, VII: Iterator<Item = VoxelTypeIDs>>(
        bitmask_iter: BMI,
        voxel_id_iter: Option<VII>,
        bitmask_binding: u32,
        voxel_id_binding: Option<u32>,
        memory_allocator: Arc<dyn MemoryAllocator>,
    ) -> Self {
        RenderingLOD {
            bitmask_buffers: DataComponent {
                buffer_scheme: DualBuffer::new(bitmask_iter, Arc::clone(&memory_allocator)),
                binding: bitmask_binding,
            } ,
            voxel_type_id_buffers: match voxel_id_iter {
                None => None,
                Some(iter) => Some(
                    DataComponent {
                        buffer_scheme: DualBuffer::new(iter, memory_allocator),
                        binding: voxel_id_binding.unwrap(),
                    }
                ),
            },
            bitmask_updated_regions: vec![],
            voxel_type_id_updated_regions: match voxel_id_binding { None => None, Some(_) => vec![] }
        }
    }
}