use std::any::Any;
use std::time::Duration;
use cgmath::{Array, Point3, Vector3};
use num_traits::{Pow, PrimInt};
use num_traits::real::Real;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{BufferCopy};

pub mod mem_grid;
use mem_grid::{
    physical_grid::MemoryGrid,
    virtual_grid::VirtualMemoryGrid,
    rendering::RenderingLayerSet,
};

mod camera;
use camera::{Camera, CameraController};
use super::voxel_type::VoxelTypeEnum;
use crate::renderer::Renderer;
use crate::world::mem_grid::physical_grid::MemoryGridLayerSet;
use crate::world::mem_grid::rendering::{RenderingLayerSetCreateParams, RenderingLayerSetMetadata};


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


struct WorldMetadata {}

struct World<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> {
    mem_grid: VirtualMemoryGrid<VE, DL>,
    camera: Camera,
    meta: WorldMetadata,
}

struct LockedWorld<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> {
    mem_grid: MemoryGrid<VE, DL>,
    camera: Camera,
    meta: WorldMetadata,
}

struct WorldCreateParams<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> {
    pub curr_tlc: TLCPos<i64>,
    pub chunk_size: usize,
    pub n_chunk_lvls: usize,
    pub n_lods: usize,
    pub gen_func: fn(VoxelPos<i64>) -> VE,
    pub layer_params: DL::LayerSetCreateParams,
    pub rendering_layer_params: RenderingLayerSetCreateParams,  // layers should be n_chunk_lvls x n_lods
}

impl<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> World<VE, DL> {
    pub fn new<const N_CHUNK_LVLS: usize, const N_LODS: usize> (
        params: WorldCreateParams<VE, DL>,
    ) -> Option<World<VE, DL>> {
        let mut mg = MemoryGrid::<VE, DL>::new(params)?.to_virtual();
        mg.reload_all();  // ENHANCEMENT: not an efficient way to load
        Some(World {
            mem_grid: mg,
            camera: Camera::default(),
            meta: WorldMetadata { },
        })
    }

    pub fn set_voxel(&mut self, position: Point3<usize>, voxel_type: VE) -> Option<()> {
        self.mem_grid.set_voxel(position, voxel_type)
    }

    /// Start the transfer of the updates to the world to the GPU. This function copies the data to
    /// the staging buffers, waits for the GPU to be done reading from the device local buffers, and
    /// then creates and submits a command buffer to transfer the updates from the staging buffers
    /// to the device local buffers.
    ///
    /// This should be called once all updates to the world for a frame are complete.
    pub fn start_transfer_to_gpu(self, renderer: &mut Renderer) -> Option<LockedWorld<VE, DL>> {
        // Transfer to staging buffers
        let (mem_grid, transfer_regions) = self.mem_grid.consolidate_and_start_transfer()?;
        let world = LockedWorld { mem_grid, meta: self.meta, camera: self.camera };

        // Start transfer pass from staging to device local buffers
        renderer.start_transfer_to_local_buffers(transfer_regions);

        Some(world)
    }
}

impl<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> LockedWorld<VE, DL> {

    pub fn move_camera(&mut self, camera_controller: &mut CameraController, dt: Duration) {
        self.camera.apply_controller_updates(camera_controller, dt);
        self.mem_grid.move_grid(&mut self.camera);
    }

    pub fn unlock(self) -> World<VE, DL> {
        World {
            mem_grid: self.mem_grid.to_virtual(),
            meta: self.meta,
            camera: self.camera,
        }
    }
}
