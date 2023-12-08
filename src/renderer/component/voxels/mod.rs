use crate::renderer::component::voxels::lod::{RendererVoxelLOD, VoxelLODUpdate};
use crate::renderer::component::{DataComponentSet};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::WriteDescriptorSet;

pub mod data;
pub mod lod;

pub struct VoxelData {
    lods: Vec<Vec<Option<RendererVoxelLOD>>>,
}

impl VoxelData {
    pub fn new(lods: Vec<Vec<Option<RendererVoxelLOD>>>) -> Self {
        VoxelData { lods }
    }

    pub fn update_staging_buffers(&mut self, updates: Vec<Vec<Option<Vec<VoxelLODUpdate>>>>) {
        for (lvl_updates, lvl) in updates.into_iter().zip(self.lods.iter_mut()) {
            for (lod_updates_o, lod_o) in lvl_updates.into_iter().zip(lvl.iter_mut()) {
                match (lod_updates_o, lod_o) {
                    (Some(lod_updates), Some(lod)) => {
                        for update in lod_updates {
                            lod.update_staging_buffers(update);
                        }
                    }
                    (None, None) => {}
                    _ => panic!(),
                }
            }
        }
    }
}

impl DataComponentSet for VoxelData {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        for lod in self.lods.iter().flatten().flatten() {
            lod.bind(descriptor_writes);
        }
    }

    fn record_repeated_buffer_transfer<L, A: CommandBufferAllocator>(&self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        for lod in self.lods.iter().flatten().flatten() {
            lod.record_repeated_buffer_transfer(builder);
        }
    }

    fn record_buffer_transfer_jit<L, A: CommandBufferAllocator>(&mut self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        for lod in self.lods.iter_mut().flatten().flatten() {
            lod.record_buffer_transfer_jit(builder);
        }
    }
}
