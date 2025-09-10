use cgmath::{Array, Point3, Vector3};
use std::marker::PhantomData;
use std::time::Duration;
use num_traits::Zero;

pub mod mem_grid;

pub mod camera;
pub mod loader;

use camera::{Camera, controller::CameraController};
use crate::world::loader::{ChunkLoader, ChunkLoadQueueItem, LoadChunk};
use crate::world::mem_grid::{FromVirtual, MemoryGridMetadata, PhysicalMemoryGrid, ToVirtual, VirtualMemoryGridStruct};

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
    buffer_chunk_states: [BufferChunkState; 3],
}

pub struct World<MG, PMG, C, VMD>
where PMG: ToVirtual<C, VMD> + FromVirtual<C, VMD> + PhysicalMemoryGrid,
      C: LoadChunk<PMG::ChunkLoadQueueItemData> + Send + 'static,
      VMD: MemoryGridMetadata {
    // physical_type: PhantomData<PMG>,
    chunk_data_type: PhantomData<C>,
    virtual_grid_metadata: PhantomData<VMD>,

    pub mem_grid: MG,
    chunk_loader: ChunkLoader<PMG::ChunkLoadQueueItemData, C>,
    chunks_to_load: Vec<ChunkLoadQueueItem<PMG::ChunkLoadQueueItemData>>,
    camera: Camera,
    metadata: WorldMetadata,
}

#[derive(Copy, Clone, PartialEq)]
pub enum BufferChunkState {
    Unloaded = 0,
    LoadedUpper = 1,
    LoadedLower = 2,
}


impl<PMG, C, VMD> World<PMG, PMG, C, VMD>
where PMG: ToVirtual<C, VMD> + FromVirtual<C, VMD>,
      C: LoadChunk<PMG::ChunkLoadQueueItemData> + Send,
      VMD: MemoryGridMetadata{
    pub fn new(
        mem_grid: PMG,
        chunk_loader: ChunkLoader<PMG::ChunkLoadQueueItemData, C>,
        camera: Camera,
        tlc_size: usize,
        tlc_load_dist_thresh: u32,
    ) -> World<PMG, PMG, C, VMD> {
        World {
            // physical_type: PhantomData,
            chunk_data_type: PhantomData,
            virtual_grid_metadata: PhantomData,
            mem_grid,
            chunk_loader,
            chunks_to_load: vec![],
            camera,
            metadata: WorldMetadata {
                tlc_size,
                tlc_load_dist_thresh,
                buffer_chunk_states: [BufferChunkState::Unloaded; 3]
            },
        }
    }

    pub fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<PMG::ChunkLoadQueueItemData>> {
        self.mem_grid.queue_load_all()
    }

    pub fn borrow_camera(&self) -> &Camera { &self.camera }

    pub fn move_camera(
        &mut self,
        camera_controller: &mut impl CameraController,
        dt: Duration
    ) {
        fn pt_gr(pt: Point3<f32>, val: f32) -> bool {
            pt.x > val && pt.y > val && pt.z > val
        }
        fn pt_lt(pt: Point3<f32>, val: f32) -> bool {
            pt.x < val && pt.y < val && pt.z < val
        }

        let last_pos = self.camera.pos().0;
        camera_controller.apply(&mut self.camera, dt);
        let cam_delta = self.camera.pos().0 - last_pos;

        let move_vector = (self.camera.position.0 / (self.metadata.tlc_size as f32))
            .map(|a| a.floor() as i64)
            - Point3::<i64>::from_value(((self.mem_grid.size() - 1) / 2) as i64);

        if !move_vector.is_zero() {
            self.camera.position = VoxelPos(
                self.camera.position.0 - (move_vector * self.metadata.tlc_size as i64)
                    .cast::<f32>()
                    .unwrap()
            );
        }

        let center_chunk_cam_pos = self.camera.position.0 - Vector3::from_value(
            self.metadata.tlc_size as f32 * (self.mem_grid.size() - 1) as f32 / 2.
        );

        // For each axis, set self.metadata.buffer_chunk_states, load_buffer, and load_in_from_edge
        let mut load_buffer: [bool; 3] = [false; 3];
        let mut load_in_from_edge = TLCVector(Vector3 { x: 0, y: 0, z: 0 });
        for a in [0, 1, 2] {
            let within_upper_load_thresh = pt_gr(
                center_chunk_cam_pos - Vector3::from_value(self.metadata.tlc_size as f32),
                self.metadata.tlc_load_dist_thresh as f32
            );
            let within_lower_load_thresh = pt_lt(center_chunk_cam_pos, self.metadata.tlc_load_dist_thresh as f32);
            let buffer_chunk_state = self.metadata.buffer_chunk_states[a];

            if move_vector[a] == 0 {
                if cam_delta[a] > 0. {
                    load_buffer[a] = within_upper_load_thresh && buffer_chunk_state != BufferChunkState::LoadedUpper;
                }
                else {
                    load_buffer[a] = within_lower_load_thresh && buffer_chunk_state != BufferChunkState::LoadedLower;
                }
            }
            else {
                // Load number of chunks in this direction equal to the number we traveled, minus
                // one if we had already loaded one of them in this direction
                let loaded_one_already = buffer_chunk_state == (
                    if move_vector[a] > 0 { BufferChunkState::LoadedUpper }
                    else { BufferChunkState::LoadedLower }
                );
                load_in_from_edge.0[a] = move_vector[a] - loaded_one_already as i64;

                (self.metadata.buffer_chunk_states[a], load_buffer[a]) = {
                    if move_vector[a] > 0 {
                        if within_upper_load_thresh { (BufferChunkState::LoadedUpper, true) }
                        else { (BufferChunkState::LoadedLower, false) }
                    }
                    else if within_lower_load_thresh { (BufferChunkState::LoadedLower, true) }
                    else { (BufferChunkState::LoadedUpper, false) }
                };
            }
        }

        self.chunks_to_load.extend(
            self.mem_grid.shift(
                TLCVector(move_vector.cast::<i32>().unwrap()),
                TLCVector(load_in_from_edge.0.cast::<i32>().unwrap()),
                load_buffer
            )
        );
    }

    pub fn set_camera_res(&mut self, width: u32, height: u32) {
        self.camera.resolution = (width, height);
    }

    pub fn unlock(self) -> World<VirtualMemoryGridStruct<C, VMD>, PMG, C, VMD> {
        let start_tlc = self.mem_grid.start_tlc();

        let mut world = World {
            // physical_type: Default::default(),
            chunk_data_type: Default::default(),
            virtual_grid_metadata: Default::default(),
            mem_grid: self.mem_grid.to_virtual(),

            chunk_loader: self.chunk_loader,
            chunks_to_load: vec![],
            camera: Default::default(),
            metadata: self.metadata,
        };

        world.chunk_loader.sync(start_tlc, &mut world.mem_grid, self.chunks_to_load);

        world
    }
}

impl<C, VMD, PMG> World<VirtualMemoryGridStruct<C, VMD>, PMG, C, VMD>
where PMG: ToVirtual<C, VMD> + FromVirtual<C, VMD>,
      C: LoadChunk<PMG::ChunkLoadQueueItemData> + Send,
      VMD: MemoryGridMetadata {
    pub fn lock(self) -> World<PMG, PMG, C, VMD> {
        World {
            // physical_type: Default::default(),
            chunk_data_type: Default::default(),
            virtual_grid_metadata: Default::default(),
            mem_grid: PMG::from_virtual(self.mem_grid),

            chunk_loader: self.chunk_loader,
            chunks_to_load: self.chunks_to_load,
            camera: Default::default(),
            metadata: self.metadata,
        }
    }

    pub fn chunks(&self) -> &Vec<Option<C>> { &self.mem_grid.chunks }
    pub fn chunks_mut(&mut self) -> &mut Vec<Option<C>> { &mut self.mem_grid.chunks }
}

