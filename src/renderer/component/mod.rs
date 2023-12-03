use crate::renderer::buffers::{BufferScheme};
use vulkano::command_buffer::allocator::{CommandBufferAllocator};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::WriteDescriptorSet;

pub mod camera;
pub mod ubo;
pub mod voxels;
pub mod materials;


pub trait DataComponentSet {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>);

    fn record_repeated_transfer<L, A: CommandBufferAllocator>(
        &self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    );

    fn record_transfer_jit<L, A: CommandBufferAllocator>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    );
}


pub struct DataComponent<B: BufferScheme> {
    pub buffer_scheme: B,
    pub binding: u32,
}


impl<B: BufferScheme> DataComponent<B> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        self.buffer_scheme.bind(descriptor_writes, self.binding);
    }
}

pub trait DataComponentWrapper {
    type B: BufferScheme;
    fn comp(&self) -> &DataComponent<Self::B>;
    fn comp_mut(&mut self) -> &mut DataComponent<Self::B>;
}

impl<B: BufferScheme, W: DataComponentWrapper<B = B>> DataComponentSet for W {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        self.comp().buffer_scheme.bind(descriptor_writes, self.comp().binding);
    }

    fn record_repeated_transfer<L, A: CommandBufferAllocator>(&self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        self.comp().buffer_scheme.record_repeated_transfer(builder);
    }

    fn record_transfer_jit<L, A: CommandBufferAllocator>(&mut self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        self.comp_mut().buffer_scheme.record_transfer_jit(builder);
    }
}