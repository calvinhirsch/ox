use crate::renderer::buffers::BufferScheme;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::WriteDescriptorSet;

pub mod camera;
pub mod materials;
pub mod ubo;
pub mod voxels;

pub trait DataComponentSet {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>);

    fn record_repeated_buffer_transfer<L, A: CommandBufferAllocator>(
        &self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    );

    fn record_buffer_transfer_jit<L, A: CommandBufferAllocator>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    );
}

#[derive(Debug)]
pub struct DataComponent<B: BufferScheme> {
    pub buffer_scheme: B,
    pub binding: u32,
}

impl<B: BufferScheme> DataComponent<B> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        self.buffer_scheme.bind(descriptor_writes, self.binding);
    }
}

impl<B: BufferScheme> DataComponentSet for DataComponent<B> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        self.buffer_scheme.bind(descriptor_writes, self.binding);
    }

    fn record_repeated_buffer_transfer<L, A: CommandBufferAllocator>(
        &self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        self.buffer_scheme.record_repeated_transfer(builder);
    }

    fn record_buffer_transfer_jit<L, A: CommandBufferAllocator>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        self.buffer_scheme.record_transfer_jit(builder);
    }
}
