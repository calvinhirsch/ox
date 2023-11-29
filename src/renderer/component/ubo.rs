use crate::renderer::buffers::{ConstantDeviceLocalBuffer, DualBuffer, DynamicBufferScheme};
use crate::renderer::component::{DataComponent, DataComponentSet};
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::buffers::dual::{ConstantDeviceLocalBuffer, DualBuffer};

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
    fn dynamic_components_mut(&mut self) -> Vec<&mut DataComponent<dyn DynamicBufferScheme>> {
        vec![&mut self.comp]
    }

    fn constant_components_mut(&self) -> Vec<&mut DataComponent<ConstantDeviceLocalBuffer<dyn BufferContents>>> {
        vec![]
    }
}
