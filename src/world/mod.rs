use cgmath::{Array, EuclideanSpace, Point3, Vector3};
use loader::{BorrowChunkForLoading, BorrowedChunk};
use mem_grid::MemoryGridEditorChunk;
use num_traits::Zero;
use std::marker::PhantomData;
use std::mem;
use std::time::Duration;

pub mod camera;
pub mod loader;
pub mod mem_grid;

use crate::world::loader::{ChunkLoadQueueItem, ChunkLoader, LoadChunk};
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor};
use camera::{controller::CameraController, Camera};

/// Position in units of top level chunks
#[derive(Clone, Copy, Debug)]
pub struct TLCPos<T>(pub Point3<T>);

/// Vector in units of top level chunks
#[derive(Clone, Copy, Debug)]
pub struct TLCVector<T>(pub Vector3<T>);

/// Position in units of 1 (i.e. LOD 0 voxels)
#[derive(Clone, Copy, Debug)]
pub struct VoxelPos<T>(pub Point3<T>);

/// Vector in units of 1 (i.e. LOD 0 voxels)
#[derive(Clone, Copy, Debug)]
pub struct VoxelVector<T>(pub Vector3<T>);

pub struct WorldMetadata {
    tlc_size: usize,
    tlc_load_dist_thresh: u32,
    // State of the buffer chunks in each axis
    buffer_chunk_states: [BufferChunkState; 3],
}

pub struct World<QI, BC: BorrowedChunk, MD, MG: MemoryGrid<ChunkLoadQueueItemData = QI>> {
    metadata_type: PhantomData<MD>,

    pub mem_grid: MG,
    chunk_loader: ChunkLoader<MG::ChunkLoadQueueItemData, MD, BC>,
    chunks_to_load: Vec<ChunkLoadQueueItem<MG::ChunkLoadQueueItemData>>,
    camera: Camera,
    metadata: WorldMetadata,
}

pub struct WorldEditor<'a, CE, MD> {
    pub mem_grid: MemoryGridEditor<CE, MD>,
    pub metadata: &'a WorldMetadata,
}

/// Whether the buffer chunks for a specific axis are unloaded, have the upper (larger coordinate)
/// chunks loaded, or the lower (smaller coordinate) chunks loaded.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum BufferChunkState {
    Unloaded = 0,
    LoadedUpper = 1,
    LoadedLower = 2,
}

impl<QI, BC: BorrowedChunk, MD, MG: MemoryGrid<ChunkLoadQueueItemData = QI>> World<QI, BC, MD, MG> {
    pub fn new(
        mem_grid: MG,
        chunk_loader: ChunkLoader<MG::ChunkLoadQueueItemData, MD, BC>,
        camera: Camera,
        tlc_size: usize,
        tlc_load_dist_thresh: u32,
    ) -> Self {
        World {
            metadata_type: PhantomData,
            mem_grid,
            chunk_loader,
            chunks_to_load: vec![],
            camera,
            metadata: WorldMetadata {
                tlc_size,
                tlc_load_dist_thresh,
                buffer_chunk_states: [BufferChunkState::Unloaded; 3],
            },
        }
    }

    pub fn queue_load_all(&mut self) {
        self.chunks_to_load.extend(self.mem_grid.queue_load_all())
    }

    pub fn borrow_camera(&self) -> &Camera {
        &self.camera
    }

    pub fn move_camera(&mut self, camera_controller: &mut impl CameraController, dt: Duration) {
        fn pt_lt(pt: Point3<f32>, val: f32) -> bool {
            pt.x < val && pt.y < val && pt.z < val
        }

        let last_pos = self.camera.pos().0;
        camera_controller.apply(&mut self.camera, dt);
        let cam_delta = self.camera.pos().0 - last_pos;

        // Delta in units of top level chunks; 0 if still in the same TLC
        let tlc_delta = (self.camera.position.0 / (self.metadata.tlc_size as f32))
            .map(|a| a.floor() as i64)
            - Point3::<i64>::from_value(((self.mem_grid.size() - 2) / 2) as i64);

        if tlc_delta.is_zero() {
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

        // For each axis, set self.metadata.buffer_chunk_states, load_buffer, and load_in_from_edge
        let mut load_buffer: [bool; 3] = [false; 3];
        let mut load_in_from_edge = TLCVector(Vector3 { x: 0, y: 0, z: 0 });
        for a in [0, 1, 2] {
            let within_upper_load_thresh = pt_lt(
                Point3::from_value(self.metadata.tlc_size as f32) - center_chunk_cam_pos.to_vec(),
                self.metadata.tlc_load_dist_thresh as f32,
            );
            let within_lower_load_thresh = pt_lt(
                center_chunk_cam_pos,
                self.metadata.tlc_load_dist_thresh as f32,
            );
            let buffer_chunk_state = self.metadata.buffer_chunk_states[a];

            if tlc_delta[a] == 0 {
                if cam_delta[a] > 0. {
                    load_buffer[a] = within_upper_load_thresh
                        && buffer_chunk_state != BufferChunkState::LoadedUpper;
                    self.metadata.buffer_chunk_states[a] = BufferChunkState::LoadedUpper;
                } else {
                    load_buffer[a] = within_lower_load_thresh
                        && buffer_chunk_state != BufferChunkState::LoadedLower;
                    self.metadata.buffer_chunk_states[a] = BufferChunkState::LoadedLower;
                }
            } else {
                // Load number of chunks in this direction equal to the number we traveled, minus
                // one if we had already loaded one of them in this direction
                let loaded_one_already = buffer_chunk_state
                    == (if tlc_delta[a] > 0 {
                        BufferChunkState::LoadedUpper
                    } else {
                        BufferChunkState::LoadedLower
                    });
                load_in_from_edge.0[a] = tlc_delta[a] - loaded_one_already as i64;

                (self.metadata.buffer_chunk_states[a], load_buffer[a]) = {
                    if tlc_delta[a] > 0 {
                        if within_upper_load_thresh {
                            (BufferChunkState::LoadedUpper, true)
                        } else {
                            (BufferChunkState::LoadedLower, false)
                        }
                    } else if within_lower_load_thresh {
                        (BufferChunkState::LoadedLower, true)
                    } else {
                        (BufferChunkState::LoadedUpper, false)
                    }
                };
            }
        }

        self.chunks_to_load.extend(self.mem_grid.shift(
            TLCVector(tlc_delta.cast::<i32>().unwrap()),
            TLCVector(load_in_from_edge.0.cast::<i32>().unwrap()),
            load_buffer,
        ));
    }

    pub fn set_camera_res(&mut self, width: u32, height: u32) {
        self.camera.resolution = (width, height);
    }
}

impl<
        QI: Clone + Send + 'static,
        BC: BorrowedChunk + LoadChunk<QI, MD> + 'static,
        MD: Clone + Send + 'static,
        MG: MemoryGrid<ChunkLoadQueueItemData = QI>,
    > World<QI, BC, MD, MG>
{
    pub fn edit<
        'a,
        CE: BorrowChunkForLoading<BC> + MemoryGridEditorChunk<'a, MG, MD>,
        F: FnOnce(WorldEditor<CE, MD>),
    >(
        &'a mut self,
        edit_f: F,
    ) {
        // Sync with chunk loader and queue new chunks to load
        let start_tlc = self.mem_grid.start_tlc();
        let chunks_to_load = mem::take(&mut self.chunks_to_load);
        let mut mem_grid_editor = CE::edit_grid(&mut self.mem_grid);
        self.chunk_loader.sync(
            start_tlc,
            &mut mem_grid_editor,
            chunks_to_load,
            &self.metadata.buffer_chunk_states,
        );

        edit_f(WorldEditor {
            mem_grid: mem_grid_editor,
            metadata: &self.metadata,
        })
    }
}
