use vulkano::buffer::BufferContents;
use crate::renderer::buffers::DualBuffer;
use crate::renderer::camera::RendererCamera;

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Ubo {
    sun_dir: [f32; 3],
    time: u32,
}

pub struct RendererUBO {
    buffers: DualBuffer<Ubo>,
}

impl RendererUBO {
    pub fn new() -> Self {
        RendererUBO
    }
}