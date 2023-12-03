use crate::renderer::component::{DataComponent, DataComponentSet};
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::buffers::dual::{DualBuffer, DualBufferWithFullCopy};

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Ubo {
    pub sun_dir: [f32; 3],
    pub time: u32,
}

pub struct RendererUBO {
    comp: DataComponent<DualBufferWithFullCopy<Ubo>>,
}

impl RendererUBO {
    pub fn new(value: Ubo, memory_allocator: Arc<dyn MemoryAllocator>, binding: u32) -> Self {
        RendererUBO {
            comp: DataComponent {
                buffer_scheme: DualBuffer::from_data(
                    value,
                    memory_allocator,
                    true
                ).with_full_copy(),
                binding,
            },
        }
    }
}

impl DataComponentSet for RendererUBO {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        todo!()
    }

    fn record_repeated_transfer<L, A: CommandBufferAllocator>(&self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        todo!()
    }

    fn record_transfer_jit<L, A: CommandBufferAllocator>(&mut self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        todo!()
    }
}
