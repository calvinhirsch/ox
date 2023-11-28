use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder};
use vulkano::descriptor_set::WriteDescriptorSet;

pub mod dual;


pub trait BufferScheme {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>, binding: u32);
}

pub trait DynamicBufferScheme {
    fn record_repeated_transfer<L, A: CommandBufferAllocator>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    );

    fn record_transfer_jit<L, A: CommandBufferAllocator>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    );
}