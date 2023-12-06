use std::cmp::max;
use std::mem;
use std::mem::size_of;
use derive_new::new;
use smallvec::SmallVec;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, BufferCopy, CopyBufferInfo};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::renderer::buffers::{BufferScheme};


/// Dual buffer scheme where different regions are copied each frame
#[derive(new)]
pub struct DualBufferWithDynamicCopyRegions<T: BufferContents> {
    staging: Subbuffer<[T]>,
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

    fn record_repeated_transfer<L, A: CommandBufferAllocator>(&self, _: &mut AutoCommandBufferBuilder<L, A>) {}

    fn record_transfer_jit<L, A: CommandBufferAllocator>(&mut self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        let copy_regions = mem::take(&mut self.copy_regions);
        builder.copy_buffer(
            CopyBufferInfo {
                regions: SmallVec::from(copy_regions),
                ..CopyBufferInfo::buffers(self.staging.clone(), self.device_local.clone())
            }
        ).unwrap();
    }
}

impl<T: BufferContents + Copy> DualBufferWithDynamicCopyRegions<T> {
    pub fn update_staging_buffer(&mut self, src: &[T], regions: impl IntoIterator<Item = BufferCopy>) {
        let mut bitmask_write = self.staging.write().unwrap();
        let rate = 1 / size_of::<T>();
        for region in regions {
            let start = region.src_offset as usize * rate;
            bitmask_write.copy_from_slice(&src[start..max(start+2, start+region.size as usize*rate)]);
        }
    }
}