use crate::renderer::buffers::{ConstantDeviceLocalBuffer, DualBuffer};
use crate::renderer::component::{DataComponent, DataComponentSet};
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::memory::allocator::MemoryAllocator;

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Ubo {
    pub sun_dir: [f32; 3],
    pub time: u32,
}

pub struct RendererUBO {
    comp: DataComponent<DualBuffer<Ubo>>,
}

impl RendererUBO {
    pub fn new(value: Ubo, memory_allocator: Arc<dyn MemoryAllocator>, binding: u32) -> Self {
        RendererUBO {
            comp: DataComponent {
                buffer_scheme: DualBuffer::from_data(value, memory_allocator, true),
                binding,
            },
        }
    }
}

impl DataComponentSet for RendererUBO {
    fn list_dynamic_components(&self) -> Vec<&DataComponent<DualBuffer<dyn BufferContents>>> {
        vec![&self.comp]
    }

    fn list_constant_components(&self) -> Vec<&DataComponent<ConstantDeviceLocalBuffer<dyn BufferContents>>> {
        vec![]
    }
}
