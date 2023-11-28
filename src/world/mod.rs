use cgmath::{Array, Point3, Vector3};
use num_traits::real::Real;
use num_traits::{Pow, PrimInt, Zero};
use std::any::Any;
use std::time::Duration;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::BufferCopy;

pub mod mem_grid;
use mem_grid::PhysicalMemoryGrid;

pub mod camera;

use camera::{Camera, controller::CameraController};
use crate::world::mem_grid::{AsVirtualMemoryGrid, VirtualMemoryGrid};

/// Position in units of top level chunks
#[derive(Clone, Copy)]
pub struct TLCPos<T>(pub Point3<T>);

/// Vector in units of top level chunks
#[derive(Clone, Copy)]
pub struct TLCVector<T>(pub Vector3<T>);

/// Position in units of 1 (i.e. LOD 0 voxels)
#[derive(Clone, Copy)]
pub struct VoxelPos<T>(pub Point3<T>);

/// Vector in units of 1 (i.e. LOD 0 voxels)
#[derive(Clone, Copy)]
pub struct VoxelVector<T>(pub Vector3<T>);

struct WorldMetadata {
    tlc_size: usize,
}

pub struct World<MG: PhysicalMemoryGrid> {
    pub mem_grid: MG,
    pub camera: Camera,
    metadata: WorldMetadata,
}

pub struct WorldEditor<'a, C> {
    pub virtual_mem_grid: VirtualMemoryGrid<C>,
    pub camera: &'a mut Camera,
}
impl<'a, C> WorldEditor<'a, C> {
    fn new<MG: AsVirtualMemoryGrid<C>>(world: &mut World<MG>) -> Self {
        WorldEditor {
            virtual_mem_grid: world.mem_grid.as_virtual(),
            camera: &mut world.camera,
        }
    }
}

impl<MG: PhysicalMemoryGrid> World<MG> {
    pub fn new(mem_grid: MG, camera: Camera, tlc_size: usize) -> Option<World<MG>> {
        Some(World {
            mem_grid,
            camera,
            metadata: WorldMetadata { tlc_size },
        })
    }
    pub fn move_camera(&mut self, camera_controller: &mut impl CameraController, dt: Duration) {
        camera_controller.apply(&mut self.camera, dt);

        let move_vector = (self.camera.position / (self.metadata.tlc_size as f32))
            .map(|a| a.floor() as i64)
            - Point3::<i64>::from_value(((self.mem_grid.size() - 1) / 2) as i64);

        if !move_vector.is_zero() {
            self.camera.position -= (move_vector * self.metadata.tlc_size as i64)
                .cast::<f32>()
                .unwrap();
            self.mem_grid.shift_offsets(move_vector);
        }
    }
}

impl<C, MG: AsVirtualMemoryGrid<C>> World<MG> {
    pub fn start_editing(&mut self) -> WorldEditor<C> {
        WorldEditor::new(self)
    }
}
