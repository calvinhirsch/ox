use crate::loader::BorrowedChunk;
use cgmath::{Array, EuclideanSpace, Point3, Vector3};
use getset::Getters;
use mem_grid::{MemGridShift, ShiftGridAxis, ShiftGridAxisVal};
use num_traits::Zero;
use std::time::Duration;

pub mod camera;
pub mod mem_grid;

use crate::loader::ChunkLoader;
use crate::world::mem_grid::{EditMemoryGridChunk, MemoryGrid, MemoryGridLoadChunks};
use camera::{controller::CameraController, Camera};

/// Position in units of top level chunks
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TlcPos<T>(pub Point3<T>);

/// Vector in units of top level chunks
#[derive(Clone, Copy, Debug)]
pub struct TlcVector<T>(pub Vector3<T>);

/// Position in units of 1 (i.e. LOD 0 voxels)
#[derive(Clone, Copy, Debug)]
pub struct VoxelPos<T>(pub Point3<T>);

/// Vector in units of 1 (i.e. LOD 0 voxels)
#[derive(Clone, Copy, Debug)]
pub struct VoxelVector<T>(pub Vector3<T>);

#[derive(Getters, Debug)]
pub struct WorldMetadata {
    #[get = "pub"]
    tlc_size: usize,
    #[get = "pub"]
    tlc_load_dist_thresh: u32,
    // State of the buffer chunks in each axis
    #[get = "pub"]
    buffer_chunk_states: [BufferChunkState; 3],
}

#[derive(Getters, Debug)]
pub struct World<MG> {
    pub mem_grid: MG,
    #[get = "pub"]
    camera: Camera,
    #[get = "pub"]
    metadata: WorldMetadata,
}

/// Whether the buffer chunks for a specific axis are unloaded, have the upper (larger coordinate)
/// chunks loaded, or the lower (smaller coordinate) chunks loaded.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum BufferChunkState {
    Unloaded = 0,
    LoadedUpper = 1,
    LoadedLower = 2,
}

impl<MG: MemoryGrid> World<MG> {
    pub fn new(mem_grid: MG, camera: Camera, tlc_size: usize, tlc_load_dist_thresh: u32) -> Self {
        World {
            mem_grid,
            camera,
            metadata: WorldMetadata {
                tlc_size,
                tlc_load_dist_thresh,
                buffer_chunk_states: [BufferChunkState::Unloaded; 3],
            },
        }
    }

    pub fn set_camera_res(&mut self, width: u32, height: u32) {
        self.camera.resolution = (width, height);
    }

    /// Given a chunk position (in global chunk coordinates), determine that chunk's position
    /// in the virtual memory grid. This involves checking the state of the buffer chunks to
    /// see where chunks on the edge of the loaded area might end up in the vgrid.
    pub fn chunk_vgrid_pos(&self, global_tlc_pos: TlcPos<i64>) -> Option<TlcVector<usize>> {
        let mut i = 0;
        if let Point3 {
            x: Some(x),
            y: Some(y),
            z: Some(z),
        } = (global_tlc_pos.0 - self.mem_grid.start_tlc().0.to_vec()).map(|a| {
            let state = self.metadata.buffer_chunk_states[i];
            i += 1;
            if a < 0 {
                if a == -1 && state == BufferChunkState::LoadedLower {
                    Some(self.mem_grid.size() - 1)
                } else {
                    None
                }
            } else if a >= self.mem_grid.size() as i64 - 1 {
                if a == self.mem_grid.size() as i64 - 1 && state == BufferChunkState::LoadedUpper {
                    Some(self.mem_grid.size() - 1)
                } else {
                    None
                }
            } else {
                Some(a as usize)
            }
        }) {
            Some(TlcVector(Vector3 { x, y, z }))
        } else {
            None
        }
    }
}

impl<QI: Eq, MG: MemoryGrid + MemoryGridLoadChunks<ChunkLoadQueueItemData = QI>> World<MG> {
    pub fn queue_load_all<BC>(&mut self, loader: &mut ChunkLoader<QI, BC>)
    where
        BC: BorrowedChunk<MemoryGrid = MG>,
    {
        for chunk in self.mem_grid.queue_load_all() {
            let prio = self.mem_grid.chunk_loading_priority(chunk.pos);
            loader.enqueue(chunk, prio);
        }
    }

    pub fn move_camera<BC>(
        &mut self,
        camera_controller: &mut impl CameraController,
        dt: Duration,
        loader: &mut ChunkLoader<QI, BC>,
    ) where
        BC: BorrowedChunk<MemoryGrid = MG>,
    {
        camera_controller.apply(&mut self.camera, dt);

        // Delta in units of top level chunks; 0 if still in the same TLC
        let tlc_delta = (self.camera.position.0 / (self.metadata.tlc_size as f32))
            .map(|a| a.floor() as i64)
            - Point3::<i64>::from_value(((self.mem_grid.size() - 2) / 2) as i64);

        // When we move to a different top level chunk, we have to adjust the camera position since it is relative to the current memory grid.
        if !tlc_delta.is_zero() {
            self.camera.position = VoxelPos(
                self.camera.position.0
                    - (tlc_delta * self.metadata.tlc_size as i64)
                        .cast::<f32>()
                        .unwrap(),
            );
        }

        let center_chunk_cam_pos = self.camera.position.0
            - Vector3::from_value(
                self.metadata.tlc_size as f32 * (self.mem_grid.size() - 2) as f32 / 2.,
            );

        // Shift memory grid and handle buffer chunks
        MemGridShift::new([0, 1, 2].map(|ax| {
            let within_upper_load_thresh = self.metadata.tlc_size as f32 - center_chunk_cam_pos[ax]
                < self.metadata.tlc_load_dist_thresh as f32;
            let within_lower_load_thresh =
                center_chunk_cam_pos[ax] < self.metadata.tlc_load_dist_thresh as f32;
            let prev_buffer_chunk_state = self.metadata.buffer_chunk_states[ax];

            if tlc_delta[ax] == 0 {
                if within_upper_load_thresh {
                    self.metadata.buffer_chunk_states[ax] = BufferChunkState::LoadedUpper;
                    match prev_buffer_chunk_state {
                        BufferChunkState::LoadedUpper => {
                            ShiftGridAxis::MaintainUpperLoadedBufferChunks
                        }
                        _ => ShiftGridAxis::LoadUpperBufferChunks,
                    }
                } else if within_lower_load_thresh {
                    self.metadata.buffer_chunk_states[ax] = BufferChunkState::LoadedLower;
                    match prev_buffer_chunk_state {
                        BufferChunkState::LoadedLower => {
                            ShiftGridAxis::MaintainLowerLoadedBufferChunks
                        }
                        _ => ShiftGridAxis::LoadLowerBufferChunks,
                    }
                } else {
                    // maintain whatever previous buffer chunk state was
                    match prev_buffer_chunk_state {
                        BufferChunkState::LoadedLower => {
                            ShiftGridAxis::MaintainLowerLoadedBufferChunks
                        }
                        BufferChunkState::LoadedUpper => {
                            ShiftGridAxis::MaintainUpperLoadedBufferChunks
                        }
                        BufferChunkState::Unloaded => ShiftGridAxis::DoNothing,
                    }
                }
            } else {
                // If we moved TLCs, that means the trailing buffer chunks are loaded because we just moved off of them--set them to represent this
                // (this value won't be used until next frame)
                self.metadata.buffer_chunk_states[ax] = {
                    if tlc_delta[ax] > 0 {
                        BufferChunkState::LoadedLower
                    } else {
                        BufferChunkState::LoadedUpper
                    }
                };

                ShiftGridAxis::Shift(ShiftGridAxisVal::new(
                    tlc_delta[ax] as i32,
                    prev_buffer_chunk_state
                        == (if tlc_delta[ax] > 0 {
                            BufferChunkState::LoadedUpper
                        } else {
                            BufferChunkState::LoadedLower
                        }),
                ))
            }
        }))
        .map(|shift| {
            for chunk in self.mem_grid.shift(&shift) {
                let priority = self.mem_grid.chunk_loading_priority(chunk.pos);
                loader.enqueue(chunk, priority);
            }
        });
    }
}

impl<MG: MemoryGrid> World<MG> {
    pub fn edit_chunk<M>(
        &mut self,
        global_tlc_pos: TlcPos<i64>,
    ) -> Option<<MG as EditMemoryGridChunk<M>>::ChunkEditor<'_>>
    where
        MG: EditMemoryGridChunk<M>,
    {
        self.mem_grid
            .edit_chunk(global_tlc_pos, self.metadata().buffer_chunk_states)
    }
}
