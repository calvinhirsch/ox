use derive_new::new;
use vulkano::buffer::{Subbuffer};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::renderer::buffers::{BufferScheme};


/// Buffer scheme with only a device local buffer (does not need to be updated continuously)
#[derive(new)]
pub struct ConstantDeviceLocalBuffer<T: ?Sized> {
    device_local: Subbuffer<T>,
}


impl<T: ?Sized> BufferScheme for ConstantDeviceLocalBuffer<T> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>, binding: u32) {
        descriptor_writes.push(WriteDescriptorSet::buffer(
            binding,
            self.device_local.clone(),
        ))
    }

    fn record_repeated_transfer<L, A: CommandBufferAllocator>(&self, _: &mut AutoCommandBufferBuilder<L, A>) { }

    fn record_transfer_jit<L, A: CommandBufferAllocator>(&mut self, _: &mut AutoCommandBufferBuilder<L, A>) { }
}
