use super::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::buffers::{ConstantDeviceLocalBuffer, DualBuffer};
use crate::renderer::component::{DataComponent, DataComponentSet};
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;

pub struct VoxelLODUpdate<'a> {
    pub bitmask: &'a Vec<VoxelBitmask>,
    pub voxel_type_ids: Option<&'a Vec<VoxelTypeIDs>>,
    pub bitmask_updated_regions: Vec<BufferCopy>,
    pub voxel_id_updated_regions: Option<Vec<BufferCopy>>,
}

pub struct RendererVoxelLOD {
    pub bitmask_buffers: DataComponent<DualBuffer<VoxelBitmask>>,
    pub voxel_type_id_buffers: Option<DataComponent<DualBuffer<VoxelTypeIDs>>>,
}
impl RendererVoxelLOD {
    pub fn new<BMI: Iterator<Item = VoxelBitmask>, VII: Iterator<Item = VoxelTypeIDs>>(
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
                Some(iter) => Some(DataComponent {
                    buffer_scheme: DualBuffer::new(iter, memory_allocator),
                    binding: voxel_id_binding.unwrap(),
                }),
            },
        }
    }

    pub fn update_staging_buffers(&mut self, update: VoxelLODUpdate) {
        self.bitmask_buffers.buffer_scheme.update_staging_buffer(update.bitmask, update.bitmask_updated_regions);
        match &mut self.voxel_type_id_buffers {
            None => {},
            Some(bs) => {
                bs.update_staging_buffer(update.voxel_type_ids.unwrap(), update.voxel_id_updated_regions.unwrap());
            }
        };
    }
}

impl DataComponentSet for RendererVoxelLOD {
    fn list_dynamic_components(&self) -> Vec<&DataComponent<DualBuffer<dyn BufferContents>>> {
        match &self.voxel_type_id_buffers {
            None => vec![&self.bitmask_buffers],
            Some(comp) => vec![&self.bitmask_buffers, comp],
        }
    }

    fn list_constant_components(&self) -> Vec<&DataComponent<ConstantDeviceLocalBuffer<dyn BufferContents>>> {
        vec![]
    }
}
