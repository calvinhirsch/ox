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
mod loader;

use camera::{Camera, controller::CameraController};
use crate::world::mem_grid::{AsVirtualMemoryGrid, ChunkLoader, VirtualMemoryGrid};
use crate::world::mem_grid::layer::MemoryGridLayerMetadata;

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


pub struct WorldMetadata {
    tlc_size: usize,
    tlc_load_dist_thresh: u32,
    tlc_unload_dist_thresh: u32,
    buffer_chunk_states: [BufferChunkState; 3],
}

pub struct World<MG> {
    pub mem_grid: MG,
    camera: Camera,
    metadata: WorldMetadata,
}

pub enum BufferChunkState {
    Unloaded = 0,
    LoadedUpper = 1,
    LoadedLower = 2,
}


impl<MG> World<MG> {
    pub fn new(
        mem_grid: MG,
        camera: Camera,
        tlc_size: usize,
        tlc_load_dist_thresh: u32,
        tlc_unload_dist_thresh: u32,
    ) -> World<MG> {
        World {
            mem_grid,
            camera,
            metadata: WorldMetadata {
                tlc_size,
                tlc_load_dist_thresh,
                tlc_unload_dist_thresh,
                buffer_chunk_states: [BufferChunkState::Unloaded; 3]
            },
        }
    }
}

impl<MG: PhysicalMemoryGrid> World<MG> {
    pub fn move_camera(&mut self, camera_controller: &mut impl CameraController, dt: Duration) {
        camera_controller.apply(&mut self.camera, dt);

        let move_vector = (self.camera.position / (self.tlc_size as f32))
            .map(|a| a.floor() as i64)
            - Point3::<i64>::from_value(((self.mem_grid.size() - 1) / 2) as i64);

        if !move_vector.is_zero() {
            self.camera.position -= (move_vector * self.tlc_size as i64)
                .cast::<f32>()
                .unwrap();
        }

        for c in vec![0, 1, 2] {
            if move_vector[c] != 0 {
                if move_vector[c].abs() > 1 {
                    self.metadata.buffer_chunk_states[c] = BufferChunkState::Unloaded;
                } else {
                    if shift[c] == -1 && self.metadata.tlc_size - center_chunk_cam_pos[c] > self.metadata.
                }
            }
        }

        self.mem_grid.shift_offsets(TLCVector(move_vector), VoxelPos(self.camera.position));
    }
}

impl<C, MG: AsVirtualMemoryGrid<C>> World<MG> {
    pub fn unlock(self) -> World<VirtualMemoryGrid<C>> {
        VirtualMemoryGrid {
            mem_grid: self.mem_grid.as_virtual(),

        }
    }
}
