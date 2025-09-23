use super::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::buffers::{
    dual::{DualBuffer, DualBufferWithDynamicCopyRegions},
    BufferScheme,
};
use crate::renderer::component::{DataComponent, DataComponentSet};
use std::sync::Arc;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, BufferCopy};
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::memory::allocator::MemoryAllocator;

#[derive(Debug, Clone)]
pub struct VoxelIDUpdate<'a> {
    pub ids: &'a [VoxelTypeIDs],
    pub updated_region: BufferCopy,
}

#[derive(Debug, Clone)]
pub struct VoxelLODUpdate<'a> {
    pub bitmask: &'a [VoxelBitmask],
    pub bitmask_updated_region: BufferCopy,
    pub id_update: Option<VoxelIDUpdate<'a>>,
}

#[derive(Debug)]
pub struct RendererVoxelLOD {
    pub bitmask_buffers: DataComponent<DualBufferWithDynamicCopyRegions<VoxelBitmask>>,
    pub id_buffers: Option<DataComponent<DualBufferWithDynamicCopyRegions<VoxelTypeIDs>>>,
}

impl RendererVoxelLOD {
    pub fn new<
        BMI: ExactSizeIterator<Item = VoxelBitmask>,
        VII: ExactSizeIterator<Item = VoxelTypeIDs>,
    >(
        bitmask_iter: BMI,
        voxel_id_iter: Option<VII>,
        bitmask_binding: u32,
        voxel_id_binding: Option<u32>,
        memory_allocator: Arc<dyn MemoryAllocator>,
    ) -> Self {
        RendererVoxelLOD {
            bitmask_buffers: DataComponent {
                buffer_scheme: DualBuffer::from_iter(
                    bitmask_iter,
                    Arc::clone(&memory_allocator),
                    false,
                )
                .with_copy_regions(),
                binding: bitmask_binding,
            },
            id_buffers: voxel_id_iter.map(|iter| DataComponent {
                buffer_scheme: DualBuffer::from_iter(iter, memory_allocator, false)
                    .with_copy_regions(),
                binding: voxel_id_binding.unwrap(),
            }),
        }
    }

    pub fn update_staging_buffers_and_prep_copy(&mut self, update: VoxelLODUpdate) {
        self.bitmask_buffers
            .buffer_scheme
            .update_staging_buffer_and_prep_copy(update.bitmask, update.bitmask_updated_region);
        match &mut self.id_buffers {
            None => {}
            Some(vids) => {
                let id_update = update
                    .id_update
                    .expect("Renderer did not receive ID update for LOD that has voxel IDs.");
                vids.buffer_scheme
                    .update_staging_buffer_and_prep_copy(id_update.ids, id_update.updated_region);
            }
        };
    }
}

impl DataComponentSet for RendererVoxelLOD {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        self.bitmask_buffers.bind(descriptor_writes);
        if let Some(comp) = &self.id_buffers {
            comp.bind(descriptor_writes);
        }
    }

    fn record_repeated_buffer_transfer<L, A: CommandBufferAllocator>(
        &self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        self.bitmask_buffers
            .buffer_scheme
            .record_repeated_transfer(builder);
        if let Some(comp) = &self.id_buffers {
            comp.buffer_scheme.record_repeated_transfer(builder);
        }
    }

    fn record_buffer_transfer_jit<L, A: CommandBufferAllocator>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        self.bitmask_buffers
            .buffer_scheme
            .record_transfer_jit(builder);
        if let Some(comp) = &mut self.id_buffers {
            comp.buffer_scheme.record_transfer_jit(builder);
        }
    }
}
