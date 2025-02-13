use crate::renderer::buffers::BufferScheme;
use derive_new::new;
use smallvec::SmallVec;
use std::cmp::max;
use std::mem;
use std::mem::size_of;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, BufferCopy, CopyBufferInfo};
use vulkano::descriptor_set::WriteDescriptorSet;

/// Dual buffer scheme where different regions are copied each frame
#[derive(new, Debug)]
pub struct DualBufferWithDynamicCopyRegions<T: BufferContents> {
    pub staging: Subbuffer<[T]>, // TODO: PUB IS TEMP, REMOVE IT
    device_local: Subbuffer<[T]>,
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
        builder
            .copy_buffer(CopyBufferInfo {
                regions: SmallVec::from(copy_regions),
                ..CopyBufferInfo::buffers(self.staging.clone(), self.device_local.clone())
            })
            .unwrap();
    }
}

impl<T: BufferContents + Copy + std::fmt::Debug> DualBufferWithDynamicCopyRegions<T> {
    /// Update staging buffers from `src` based on `regions` and add `regions` to `self.copy_regions`
    /// so that those regions are later transferred to the GPU.
    pub fn update_staging_buffer_and_prep_copy(
        &mut self,
        src: &[T],
        regions: impl IntoIterator<Item = BufferCopy>,
    ) {
        // Regions here are in bytes, so we need to rescale them to be indices
        let mut bitmask_write = self.staging.write().unwrap();
        for region in regions {
            let src_offset = region.src_offset as usize / size_of::<T>();
            let dst_offset = region.dst_offset as usize / size_of::<T>();
            let size = max(1, (region.size as usize) / size_of::<T>());
            dbg!(src_offset, dst_offset, size);
            bitmask_write[dst_offset..dst_offset + size]
                .copy_from_slice(&src[src_offset..src_offset + size]);
            // dbg!((region.src_offset, region.dst_offset, region.size));
            // self.copy_regions.push(BufferCopy {
            //     src_offset: src_offset as u64,
            //     dst_offset: dst_offset as u64,
            //     size: region.size,
            //     ..Default::default()
            // });
            println!();
            self.copy_regions.push(region);
        }
    }
}
