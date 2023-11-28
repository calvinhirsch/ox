use std::time::Duration;
use super::Camera;
pub mod winit;


pub trait CameraController {
    fn apply(&mut self, camera: &mut Camera, dt: Duration);
}
