use crate::renderer::component::voxels::lod::{RendererVoxelLOD, VoxelLODUpdate};
use crate::renderer::component::DataComponentSet;
use data::VoxelBitmask;
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
        for (lod_i, (lod, updates)) in self.lods.iter_mut().zip(updates.into_iter()).enumerate() {
            dbg!(lod_i);
            if lod_i == 1 {
                for b in lod
                    .bitmask_buffers
                    .buffer_scheme
                    .staging
                    .write()
                    .unwrap()
                    .iter_mut()
                {
                    b.mask = u128::MAX;
                }
            }
            let temp_cloned_updates__ = updates.clone();
            for update in updates {
                lod.update_staging_buffers_and_prep_copy(update);
            }
            if lod_i == 1 {
                for upd in temp_cloned_updates__ {
                    for region in upd.bitmask_updated_regions {
                        let dst_offset = region.dst_offset as usize / size_of::<VoxelBitmask>();
                        let size = 1.max((region.size as usize) / size_of::<VoxelBitmask>());
                        dbg!(region.dst_offset, region.size);
                        println!(
                            "Staging buffer full: {}",
                            &lod.bitmask_buffers.buffer_scheme.staging.read().unwrap()
                                [dst_offset..dst_offset + size]
                                .iter()
                                .all(|m| m.mask == u128::MAX)
                        );
                    }
                }
                println!("Staging buffer % full: {}", {
                    let read = lod.bitmask_buffers.buffer_scheme.staging.read().unwrap();
                    read.iter()
                        .map(|x| (x.mask == u128::MAX) as u32)
                        .sum::<u32>() as f32
                        * 100.
                        / read.len() as f32
                });
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
