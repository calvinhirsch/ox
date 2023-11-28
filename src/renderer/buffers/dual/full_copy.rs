use derive_new::new;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::renderer::buffers::{BufferScheme, DynamicBufferScheme};


/// Dual buffer scheme where the whole staging buffer is copied to the device local buffer every frame
#[derive(new)]
pub struct DualBufferWithFullCopy<T: BufferContents> {
    staging: Subbuffer<T>,
    device_local: Subbuffer<T>,
}


impl<T: BufferContents> BufferScheme for DualBufferWithFullCopy<T> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>, binding: u32) {
        descriptor_writes.push(WriteDescriptorSet::buffer(
            binding,
            self.device_local.clone(),
        ))
    }
}

impl<T: BufferContents> DynamicBufferScheme for DualBufferWithFullCopy<T> {
    fn record_repeated_transfer<L, A: CommandBufferAllocator>(&self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.staging.clone(),
                self.device_local.clone(),
            ))
            .unwrap();
    }

    fn record_transfer_jit<L, A: CommandBufferAllocator>(&self, _: &mut AutoCommandBufferBuilder<L, A>) {}
}

impl<T: BufferContents> DualBufferWithFullCopy<T> {
    pub fn update_staging_buffer(&mut self, src: &[T]) {
        let mut bitmask_write = self.staging.write().unwrap();
        bitmask_write.copy_from_slice(src);
    }
}