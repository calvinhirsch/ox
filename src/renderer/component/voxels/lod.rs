use super::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::component::{DataComponent, DataComponentSet};
use std::sync::Arc;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, BufferCopy};
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::buffers::dual::{DualBuffer, DualBufferWithDynamicCopyRegions};

pub struct VoxelLODUpdate<'a> {
    pub bitmask: &'a Vec<VoxelBitmask>,
    pub voxel_type_ids: Option<&'a Vec<VoxelTypeIDs>>,
    pub bitmask_updated_regions: Vec<BufferCopy>,
    pub voxel_id_updated_regions: Option<Vec<BufferCopy>>,
}

pub struct RendererVoxelLOD {
    pub bitmask_buffers: DataComponent<DualBufferWithDynamicCopyRegions<VoxelBitmask>>,
    pub voxel_type_id_buffers: Option<DataComponent<DualBufferWithDynamicCopyRegions<VoxelTypeIDs>>>,
}
impl RendererVoxelLOD {
    pub fn new<BMI: ExactSizeIterator<Item = VoxelBitmask>, VII: ExactSizeIterator<Item = VoxelTypeIDs>>(
        bitmask_iter: BMI,
        voxel_id_iter: Option<VII>,
        bitmask_binding: u32,
        voxel_id_binding: Option<u32>,
        memory_allocator: Arc<dyn MemoryAllocator>,
    ) -> Self {
        RendererVoxelLOD {
            bitmask_buffers: DataComponent {
                buffer_scheme: DualBuffer::from_iter(bitmask_iter, Arc::clone(&memory_allocator), false).with_copy_regions(),
                binding: bitmask_binding,
            },
            voxel_type_id_buffers: match voxel_id_iter {
                None => None,
                Some(iter) => Some(DataComponent {
                    buffer_scheme: DualBuffer::from_iter(iter, memory_allocator, false).with_copy_regions(),
                    binding: voxel_id_binding.unwrap(),
                }),
            },
        }
    }

    pub fn update_staging_buffers(&mut self, update: VoxelLODUpdate) {
        self.bitmask_buffers.buffer_scheme.update_staging_buffer(update.bitmask, update.bitmask_updated_regions);
        match &mut self.voxel_type_id_buffers {
            None => {},
            Some(vids) => {
                vids.buffer_scheme.update_staging_buffer(update.voxel_type_ids.unwrap(), update.voxel_id_updated_regions.unwrap());
            }
        };
    }
}

impl DataComponentSet for RendererVoxelLOD {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        todo!()
    }

    fn record_repeated_transfer<L, A: CommandBufferAllocator>(&self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        todo!()
    }

    fn record_transfer_jit<L, A: CommandBufferAllocator>(&mut self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        todo!()
    }
}
