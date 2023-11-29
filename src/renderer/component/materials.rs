use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::buffers::dual::{ConstantDeviceLocalBuffer, DualBuffer};
use crate::renderer::component::DataComponent;


#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Material {
    pub color: [f32; 3],
    _pad1: f32,
    pub specular_color: [f32; 3],
    _pad2: f32,
    pub emission_color: [f32; 3],
    pub emission_strength: f32,
    pub specular_prob_perpendicular: f32,
    pub specular_prob_parallel: f32,
    _pad3: [f32; 2],
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


pub struct MaterialList {
    comp: DataComponent<ConstantDeviceLocalBuffer<Material>>
}


impl MaterialList {
    pub fn new<L, A: CommandBufferAllocator>(
        materials: &Vec<Material>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        binding: u32,
        one_time_transfer_builder: &mut AutoCommandBufferBuilder<L, A>,
    ) -> MaterialList {
        MaterialList {
            comp: DataComponent {
                buffer_scheme: ConstantDeviceLocalBuffer::from_dual_buffer(
                    one_time_transfer_builder,
                    DualBuffer::from_iter(materials.iter().copied(), memory_allocator, false),
                ),
                binding,
            }
        }
    }
}