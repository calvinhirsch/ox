use crate::renderer::buffers::BufferScheme;
use derive_new::new;
use getset::Getters;
use smallvec::SmallVec;
use std::cmp::max;
use std::mem;
use std::mem::size_of;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, BufferCopy, CopyBufferInfo};
use vulkano::descriptor_set::WriteDescriptorSet;

/// Dual buffer scheme where different regions are copied each frame
#[derive(new, Debug, Getters)]
pub struct DualBufferWithDynamicCopyRegions<T: BufferContents> {
    staging: Subbuffer<[T]>,
    device_local: Subbuffer<[T]>,
    #[get = "pub"]
    copy_regions: Vec<BufferCopy>,
}

impl<T: BufferContents> BufferScheme for DualBufferWithDynamicCopyRegions<T> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>, binding: u32) {
        descriptor_writes.push(WriteDescriptorSet::buffer(
            binding,
            self.device_local.clone(),
        ))
    }

    fn record_repeated_transfer<L, A: CommandBufferAllocator>(
        &self,
        _: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        // do no repeated transfer--only jit
    }

    fn record_transfer_jit<L, A: CommandBufferAllocator>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        let copy_regions = mem::take(&mut self.copy_regions);
        if copy_regions.len() > 0 {
            builder
                .copy_buffer(CopyBufferInfo {
                    regions: SmallVec::from(copy_regions),
                    ..CopyBufferInfo::buffers(self.staging.clone(), self.device_local.clone())
                })
                .unwrap();
        }
    }
}

impl<T: BufferContents + Copy + std::fmt::Debug> DualBufferWithDynamicCopyRegions<T> {
    /// Update staging buffers from `src` based on `regions` and add `regions` to `self.copy_regions`
    /// so that those regions are later transferred to the GPU.
    pub fn update_staging_buffer_and_prep_copy(
        &mut self,
        src: &[T],
        regions: impl IntoIterator<Item = BufferCopy>, // regions to copy from src to staging buffer
    ) {
        // Regions here are in bytes, so we need to rescale them to be indices
        let mut bitmask_write = self.staging.write().unwrap();
        for region in regions {
            // copy from src to staging buffer
            let src_offset = region.src_offset as usize / size_of::<T>();
            let dst_offset = region.dst_offset as usize / size_of::<T>();
            let size = max(1, (region.size as usize) / size_of::<T>());
            bitmask_write[dst_offset..dst_offset + size]
                .copy_from_slice(&src[src_offset..src_offset + size]);

            // queue copy from staging buffer to GPU
            self.copy_regions.push(BufferCopy {
                src_offset: region.dst_offset as u64,
                dst_offset: region.dst_offset as u64, // staging buffer should be same as device local buffer
                size: region.size as u64,
                ..Default::default()
            });
        }
    }
}
