use cgmath::{Angle, InnerSpace, Point3, Rad};

pub mod controller;


#[derive(Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,   // radians
    pub pitch: Rad<f32>, // radians
    pub viewport_dist: f32,
    pub resolution: (u32, u32), // width, height
    pub avg_fov: Rad<f32>,      // average of x-fov and y-fov
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: Point3 {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            yaw: 0.into(),
            pitch: 0.into(),
            viewport_dist: 0.1,
            resolution: (0, 0),
            avg_fov: 90.into(),
        }
    }
}

