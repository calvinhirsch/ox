use crate::renderer::component::voxels::lod::{RendererVoxelLOD, VoxelLODUpdate};
use crate::renderer::component::DataComponentSet;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::WriteDescriptorSet;

pub mod data;
pub mod lod;

pub struct VoxelData<const N: usize> {
    lods: [RendererVoxelLOD; N],
}

impl<const N: usize> VoxelData<N> {
    pub fn new(lods: [RendererVoxelLOD; N]) -> Self {
        VoxelData { lods }
    }

    pub fn update_staging_buffers_and_prep_copy(&mut self, updates: [Vec<VoxelLODUpdate>; N]) {
        for (lod, updates) in self.lods.iter_mut().zip(updates.into_iter()) {
            for update in updates {
                lod.update_staging_buffers_and_prep_copy(update);
            }
        }
    }
}

impl<const N: usize> DataComponentSet for VoxelData<N> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        for lod in self.lods.iter() {
            lod.bind(descriptor_writes);
        }
    }

    fn record_repeated_buffer_transfer<L, A: CommandBufferAllocator>(
        &self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        for lod in self.lods.iter() {
            lod.record_repeated_buffer_transfer(builder);
        }
    }

    fn record_buffer_transfer_jit<L, A: CommandBufferAllocator>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        for lod in self.lods.iter_mut() {
            lod.record_buffer_transfer_jit(builder);
        }
    }
}
