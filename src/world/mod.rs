use std::any::Any;
use std::time::Duration;
use cgmath::{Array, Point3, Vector3};
use num_traits::{Pow, PrimInt, Zero};
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


struct WorldMetadata {
    tlc_size: usize,
    chunk_size: usize,
}

struct World<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> {
    mem_grid: VirtualMemoryGrid<VE, DL>,
    camera: Camera,
    metadata: WorldMetadata,
}

struct LockedWorld<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> {
    mem_grid: MemoryGrid<VE, DL>,
    camera: Camera,
    metadata: WorldMetadata,
}

impl<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> World<VE, DL> {
    pub fn new (
        mem_grid: MemoryGrid,
        camera: Camera,
    ) -> Option<World<VE, DL>> {
        Some(LockedWorld(mem_grid, camera, WorldMetadata { tlc_size: mem_grid.tlc_size(), chunk_size: mem_grid. }))
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
        let world = LockedWorld { mem_grid, metadata: self.metadata, camera: self.camera };

        // Start transfer pass from staging to device local buffers
        renderer.start_transfer_to_local_buffers(transfer_regions);

        Some(world)
    }
}

impl<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> LockedWorld<VE, DL> {
    fn move_camera_for_size(camera: &mut Camera, tlc_size: usize, grid_size: usize) -> Vector3<i64> {
        let move_vector = (camera.position / (tlc_size as f32))
            .map(|a| a.floor() as i64)
            - Point3::<i64>::from_value(((grid_size-1)/2) as i64);

        if !move_vector.is_zero() {
            camera.position -= (move_vector * tlc_size as i64)
                .cast::<f32>().unwrap();
        }

        move_vector
    }

    pub fn move_camera(&mut self, camera_controller: &mut CameraController) {
        let move_vector = Self::move_camera_for_size(&mut self.camera, self.metadata.tlc_size, self.metadata.size);
        self.shift_offsets(move_vector);
    }

    pub fn unlock(self) -> World<VE, DL> {
        World {
            mem_grid: self.mem_grid.to_virtual(),
            metadata: self.metadata,
            camera: self.camera,
        }
    }
}
