use std::f32::consts::PI;
use std::time::Duration;
use cgmath::{Angle, InnerSpace, Point3, Rad, Vector3};
use vulkano::buffer::BufferContents;
use winit::event::{ElementState, VirtualKeyCode};


#[derive(Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,  // radians
    pub pitch: Rad<f32>,  // radians
    pub viewport_dist: f32,
    pub resolution: (u32, u32),  // width, height
    pub avg_fov: Rad<f32>,  // average of x-fov and y-fov
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: Point3 { x: 0., y: 0., z: 0. },
            yaw: 0.into(),
            pitch: 0.into(),
            viewport_dist: 0.1,
            resolution: (0, 0),
            avg_fov: 90.into(),
        }
    }
}

impl Camera {
    pub fn apply_controller_updates(&mut self, controller: &mut CameraController, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = self.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, -yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, -yaw_cos).normalize();
        self.position += forward * (controller.amount_forward - controller.amount_backward) * controller.speed * dt;
        self.position += right * (controller.amount_right - controller.amount_left) * controller.speed * dt;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        self.position.y += (controller.amount_up - controller.amount_down) * controller.speed * dt;

        // Rotate
        self.yaw += Rad(controller.rotate_horizontal) * controller.sensitivity * dt;
        self.pitch += Rad(controller.rotate_vertical) * controller.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non cardinal direction.
        controller.rotate_horizontal = 0.0;
        controller.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if self.pitch < -Rad(SAFE_FRAC_PI_2) {
            self.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if self.pitch > Rad(SAFE_FRAC_PI_2) {
            self.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}


// https://sotrh.github.io/learn-wgpu/intermediate/tutorial12-camera/#cleaning-up-lib-rs

#[derive(Debug)]
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    speed: f32,
    sensitivity: f32,
}

const SAFE_FRAC_PI_2: f32 = PI / 2.0 - 0.0001;

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool{
        let amount = if state == ElementState::Pressed { 1.0 } else { 0.0 };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.amount_right = amount;
                true
            }
            VirtualKeyCode::Space => {
                self.amount_up = amount;
                true
            }
            VirtualKeyCode::LShift => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }
}