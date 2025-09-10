use crate::renderer::buffers::dual::{ConstantDeviceLocalBuffer, DualBuffer};
use crate::renderer::component::DataComponent;
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::memory::allocator::MemoryAllocator;

#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
pub struct Material {
    pub color: [f32; 3],
    pub _pad1: f32,
    pub specular_color: [f32; 3],
    pub _pad2: f32,
    pub emission_color: [f32; 3],
    pub emission_strength: f32,
    pub specular_prob_perpendicular: f32,
    pub specular_prob_parallel: f32,
    pub _pad3: [f32; 2],
}

impl Default for Material {
    fn default() -> Self {
        Material {
            color: [0., 0., 0.],
            specular_color: [0., 0., 0.],
            emission_color: [0., 0., 0.],
            emission_strength: 0.,
            specular_prob_perpendicular: 0.,
            specular_prob_parallel: 0.,
            _pad1: 0.,
            _pad2: 0.,
            _pad3: [0., 0.],
        }
    }
}

pub type MaterialList = DataComponent<ConstantDeviceLocalBuffer<[Material]>>;

impl MaterialList {
    pub fn new<L, A: CommandBufferAllocator>(
        materials: &[Material],
        memory_allocator: Arc<dyn MemoryAllocator>,
        binding: u32,
        one_time_transfer_builder: &mut AutoCommandBufferBuilder<L, A>,
    ) -> MaterialList {
        DataComponent {
            buffer_scheme: DualBuffer::from_iter(
                materials.iter().copied(),
                memory_allocator,
                false,
            )
            .without_staging_buffer(one_time_transfer_builder),
            binding,
        }
    }
}
