use cgmath::{Array, Point3, Vector3};
use getset::Getters;
use loader::{BorrowChunkForLoading, BorrowedChunk};
use mem_grid::{MemGridShift, MemoryGridEditorChunk, ShiftGridAxis, ShiftGridAxisVal};
use num_traits::Zero;
use std::marker::PhantomData;
use std::mem;
use std::time::Duration;

pub mod camera;
pub mod loader;
pub mod mem_grid;

use crate::world::loader::{ChunkLoadQueueItem, ChunkLoader};
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor};
use camera::{controller::CameraController, Camera};

/// Position in units of top level chunks
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Debug)]
pub struct WorldMetadata {
    tlc_size: usize,
    tlc_load_dist_thresh: u32,
    // State of the buffer chunks in each axis
    buffer_chunk_states: [BufferChunkState; 3],
}

#[derive(Getters, Debug)]
pub struct World<QI: Eq, BC: BorrowedChunk, MD, MG: MemoryGrid<ChunkLoadQueueItemData = QI>> {
    metadata_type: PhantomData<MD>,

    pub mem_grid: MG,
    #[get = "pub"]
    chunk_loader: ChunkLoader<MG::ChunkLoadQueueItemData, MD, BC>,
    chunks_to_load: Vec<ChunkLoadQueueItem<MG::ChunkLoadQueueItemData>>,
    #[get = "pub"]
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

impl<QI: Eq, BC: BorrowedChunk, MD, MG: MemoryGrid<ChunkLoadQueueItemData = QI>>
    World<QI, BC, MD, MG>
{
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

    pub fn move_camera(&mut self, camera_controller: &mut impl CameraController, dt: Duration) {
        let last_pos = self.camera.pos().0;
        camera_controller.apply(&mut self.camera, dt);

        println!(
            "last pos: {:?}, curr pos: {:?}",
            last_pos,
            self.camera.pos().0
        );

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
        .map(|shift| self.chunks_to_load.extend(self.mem_grid.shift(&shift)));
    }

    pub fn set_camera_res(&mut self, width: u32, height: u32) {
        self.camera.resolution = (width, height);
    }
}

impl<
        QI: Clone + Send + Eq + std::fmt::Debug + 'static,
        BC: BorrowedChunk + 'static,
        MD: Clone + Send + 'static,
        MG: MemoryGrid<ChunkLoadQueueItemData = QI>,
    > World<QI, BC, MD, MG>
{
    pub fn edit<
        'a,
        CE: BorrowChunkForLoading<BC> + MemoryGridEditorChunk<'a, MG, MD>,
        EF: FnOnce(WorldEditor<CE, MD>),
        LF: Fn(&mut BC, ChunkLoadQueueItem<QI>, MD) + Sync,
    >(
        &'a mut self,
        edit_f: EF,
        load_f: &'static LF,
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
            load_f,
        );

        edit_f(WorldEditor {
            mem_grid: mem_grid_editor,
            metadata: &self.metadata,
        })
    }
}
