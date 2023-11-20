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

/// Uniform buffer object containing camera info that gets passed to the GPU
#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct CameraUBO {
    eye: [f32; 3],
    _pad1: f32,
    viewport_center: [f32; 3],
    _pad2: f32,
    right_dir: [f32; 3],  // should be normalized
    _pad3: f32,
    up_dir: [f32; 3],  // should be normalized
    _pad4: f32,
}

impl CameraUBO {
    pub fn new(camera: &Camera, origin: Point3<f32>) -> Self {
        let mut s = CameraUBO {
            eye: [0.0, 0.0, 0.0],
            viewport_center: [0.0, 0.0, 0.0],
            right_dir: [0.0, 0.0, 0.0],
            up_dir: [0.0, 0.0, 0.0],
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        s.update(camera, origin);
        s
    }

    pub fn update(&mut self, camera: &Camera, origin: Point3<f32>) {
        let avg_res = (camera.resolution.0 + camera.resolution.1) as f32 / 2.;
        let avg_viewport_dim = camera.viewport_dist * (camera.avg_fov / 2.0).tan();
        let viewport_half_dims = (
            avg_viewport_dim * camera.resolution.0 as f32 / avg_res,
            avg_viewport_dim * camera.resolution.1 as f32 / avg_res,
        );

        self.eye = (camera.position - origin).try_into().unwrap();

        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = camera.pitch.sin_cos();

        self.viewport_center = (camera.position - origin + Vector3 {
            x: yaw_cos * pitch_cos * camera.viewport_dist,
            y: -pitch_sin * camera.viewport_dist,
            z: -yaw_sin * pitch_cos * camera.viewport_dist,
        }).try_into().unwrap();

        self.right_dir = Vector3 {
            x: -yaw_sin * viewport_half_dims.0,
            y: 0.,
            z: -yaw_cos * viewport_half_dims.0,
        }.try_into().unwrap();

        self.up_dir = Vector3 {
            x: yaw_cos * (if camera.pitch > Rad(0.) {1.} else {-1.}) * (1. - pitch_cos) * viewport_half_dims.1,
            y: pitch_cos * viewport_half_dims.1,
            z: yaw_sin * (if camera.pitch > Rad(0.) {-1.} else {1.}) * (1. - pitch_cos) * viewport_half_dims.1,
        }.try_into().unwrap();
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