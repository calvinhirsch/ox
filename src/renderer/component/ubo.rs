use crate::renderer::component::{DataComponent, DataComponentWrapper};
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::buffers::dual::{DualBuffer, DualBufferWithFullCopy};

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Ubo {
    pub sun_dir: [f32; 3],
    pub time: u32,
    pub start_tlc: [i32; 3],  // ENHANCEMENT: These should really be i64, but glsl uses 32 bit ints
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

impl DataComponentWrapper for RendererUBO {
    type B = DualBufferWithFullCopy<Ubo>;
    fn comp(&self) -> &DataComponent<Self::B> { &self.comp }
    fn comp_mut(&mut self) -> &mut DataComponent<Self::B> { &mut self.comp }
}
