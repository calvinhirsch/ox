use crate::world::VoxelPos;
use cgmath::{Angle, Point3, Rad, Vector3};

pub mod controller;

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: VoxelPos<f32>, // position relative to the memory grid; 0,0,0 is the bottom corner of the memory grid
    pub yaw: Rad<f32>,           // radians
    pub pitch: Rad<f32>,         // radians
    pub viewport_dist: f32,
    pub resolution: (u32, u32), // width, height
    pub avg_fov: Rad<f32>,      // average of x-fov and y-fov
}

impl Camera {
    /// Create a camera at the center position of the center top level chunk.
    pub fn new(tlc_size: usize, mem_grid_size: usize) -> Camera {
        // Camera position is relative to the memory grid.
        let offset = ((mem_grid_size / 2 - 1) * tlc_size + tlc_size / 2) as f32;
        Camera {
            position: VoxelPos(Point3 {
                x: offset,
                y: offset,
                z: offset,
            }),
            yaw: Rad(0.),
            pitch: Rad(0.),
            viewport_dist: 0.1,
            resolution: (0, 0),
            avg_fov: Rad(90.),
        }
    }

    pub fn pos(&self) -> &VoxelPos<f32> {
        &self.position
    }

    pub fn viewport_center(&self) -> Point3<f32> {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        (self.position.0
            + Vector3 {
                x: yaw_cos * pitch_cos * self.viewport_dist,
                y: -pitch_sin * self.viewport_dist,
                z: -yaw_sin * pitch_cos * self.viewport_dist,
            })
        .try_into()
        .unwrap()
    }
}
