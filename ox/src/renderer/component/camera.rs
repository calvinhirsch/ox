use crate::renderer::buffers::dual::{DualBuffer, DualBufferWithFullCopy};
use crate::renderer::component::DataComponent;
use crate::world::camera::Camera;
use crate::world::VoxelPos;
use cgmath::{Angle, Array, Point3, Rad, Vector3};
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::memory::allocator::MemoryAllocator;

pub type RendererCamera = DataComponent<DualBufferWithFullCopy<CameraUBO>>;
impl RendererCamera {
    pub fn new(binding: u32, allocator: Arc<dyn MemoryAllocator>) -> Self {
        DataComponent {
            buffer_scheme: DualBuffer::from_data(CameraUBO::new_blank(), allocator, true)
                .with_full_copy(),
            binding,
        }
    }

    pub fn update_staging_buffer(&mut self, camera: &Camera) {
        let mut w = self.buffer_scheme.write_staging();
        w.update(camera, VoxelPos(Point3::<f32>::from_value(0.)));
    }
}

/// Uniform buffer object containing camera info that gets passed to the GPU
#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct CameraUBO {
    eye: [f32; 3],
    _pad1: f32,
    viewport_center: [f32; 3],
    _pad2: f32,
    right_dir: [f32; 3], // should be normalized
    _pad3: f32,
    up_dir: [f32; 3], // should be normalized
    _pad4: f32,
}

impl CameraUBO {
    pub fn new_blank() -> Self {
        CameraUBO {
            eye: [0.0, 0.0, 0.0],
            viewport_center: [0.0, 0.0, 0.0],
            right_dir: [0.0, 0.0, 0.0],
            up_dir: [0.0, 0.0, 0.0],
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        }
    }

    pub fn new(camera: &Camera, origin: VoxelPos<f32>) -> Self {
        let mut s = CameraUBO::new_blank();
        s.update(camera, origin);
        s
    }

    pub fn update(&mut self, camera: &Camera, origin: VoxelPos<f32>) {
        let avg_res = (camera.resolution.0 + camera.resolution.1) as f32 / 2.;
        let avg_viewport_dim = camera.viewport_dist * (camera.avg_fov / 2.0).tan();
        let viewport_half_dims = (
            avg_viewport_dim * camera.resolution.0 as f32 / avg_res,
            avg_viewport_dim * camera.resolution.1 as f32 / avg_res,
        );

        self.eye = (camera.position.0 - origin.0).try_into().unwrap();

        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let (_, pitch_cos) = camera.pitch.sin_cos();

        self.viewport_center = camera.viewport_center().into();

        self.right_dir = Vector3 {
            x: -yaw_sin * viewport_half_dims.0,
            y: 0.,
            z: -yaw_cos * viewport_half_dims.0,
        }
        .try_into()
        .unwrap();

        self.up_dir = Vector3 {
            x: yaw_cos
                * (if camera.pitch > Rad(0.) { 1. } else { -1. })
                * (1. - pitch_cos)
                * viewport_half_dims.1,
            y: pitch_cos * viewport_half_dims.1,
            z: yaw_sin
                * (if camera.pitch > Rad(0.) { -1. } else { 1. })
                * (1. - pitch_cos)
                * viewport_half_dims.1,
        }
        .try_into()
        .unwrap();
    }
}
