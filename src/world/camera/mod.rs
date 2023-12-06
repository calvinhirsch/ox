use cgmath::{Point3, Rad};
use crate::world::VoxelPos;

pub mod controller;


#[derive(Debug)]
pub struct Camera {
    pub position: VoxelPos<f32>,
    pub yaw: Rad<f32>,   // radians
    pub pitch: Rad<f32>, // radians
    pub viewport_dist: f32,
    pub resolution: (u32, u32), // width, height
    pub avg_fov: Rad<f32>,      // average of x-fov and y-fov
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: VoxelPos(Point3 {
                x: 0.,
                y: 0.,
                z: 0.,
            }),
            yaw: Rad(0.),
            pitch: Rad(0.),
            viewport_dist: 0.1,
            resolution: (0, 0),
            avg_fov: Rad(90.),
        }
    }
}

impl Camera {
    pub fn pos(&self) -> &VoxelPos<f32> {
        &self.position
    }
}