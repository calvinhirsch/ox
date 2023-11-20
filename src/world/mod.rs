use std::any::Any;
use std::time::Duration;
use cgmath::{Array, Point3, Vector3};
use num_traits::{Pow, PrimInt};
use num_traits::real::Real;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{BufferCopy};

mod mem_grid;
use mem_grid::{VirtualMemoryGrid, MemoryGrid};

mod camera;
use camera::{Camera, CameraController};
use super::voxel_type::VoxelTypeEnum;
use crate::renderer::Renderer;


/// Position in units of top level chunks
#[derive(Clone, Copy)]
struct TLCPos<T>(Point3<T>);

/// Vector in units of top level chunks
#[derive(Clone, Copy)]
struct TLCVector<T>(Vector3<T>);

/// Position in units of 1 (i.e. LOD 0 voxels)
#[derive(Clone, Copy)]
struct VoxelPos<T>(Point3<T>);

/// Vector in units of 1 (i.e. LOD 0 voxels)
#[derive(Clone, Copy)]
struct VoxelVector<T>(Vector3<T>);


struct WorldMetadata {}

struct World<VE: VoxelTypeEnum> {
    mem_grid: VirtualMemoryGrid<VE>,
    camera: Camera,
    meta: WorldMetadata,
}

struct LockedWorld<VE: VoxelTypeEnum> {
    mem_grid: MemoryGrid<VE>,
    camera: Camera,
    meta: WorldMetadata,
}

struct WorldCreateParams<const N_CHUNK_LVLS: usize, const N_LODS: usize, VE: VoxelTypeEnum> {
    curr_tlc: TLCPos<i64>,
    chunk_size: usize,
    gen_func: fn(VoxelPos<i64>) -> VE,
    render_area_sizes: [[Option<usize>; N_LODS]; N_CHUNK_LVLS],
    load_thresh_dist: usize,
    lod_block_fill_thresh: u8,
}
impl<VE: VoxelTypeEnum> Default for WorldCreateParams<2, 3, VE> {
    fn default() -> Self {
        WorldCreateParams {
            curr_tlc: TLCPos(Point3 { x: 0, y: 0, z: 0 }),
            chunk_size: 8,
            gen_func: |VoxelPos(p)| {
                if (p.y as f64) < 20. + 10. * (p.x as f64).pow(4.0f64/5.0f64).cos() { VE::first().unwrap() }
                else { VE::last().unwrap() }
            },
            render_area_sizes: [[Some(3), Some(7), Some(15)], [Some(31), None, None]],
            load_thresh_dist: 8,
            lod_block_fill_thresh: 3,
        }
    }
}

impl<VE: VoxelTypeEnum> World<VE> {
    pub fn set_voxel(&mut self, position: Point3<usize>, voxel_type: VE) -> Option<()> {
        self.mem_grid.set_voxel(position, voxel_type)
    }

    /// Start the transfer of the updates to the world to the GPU. This function copies the data to
    /// the staging buffers, waits for the GPU to be done reading from the device local buffers, and
    /// then creates and submits a command buffer to transfer the updates from the staging buffers
    /// to the device local buffers.
    ///
    /// This should be called once all updates to the world for a frame are complete.
    pub fn start_transfer_to_gpu(self, renderer: &mut Renderer) -> Option<LockedWorld<VE>> {
        // Transfer to staging buffers
        let (mem_grid, transfer_regions) = self.mem_grid.consolidate_and_start_transfer()?;
        let world = LockedWorld { mem_grid, meta: self.meta, camera: self.camera };

        // Start transfer pass from staging to device local buffers
        renderer.start_transfer_to_local_buffers(transfer_regions);

        Some(world)
    }
}

impl<VE: VoxelTypeEnum> LockedWorld<VE> {
    pub fn new<const N_CHUNK_LVLS: usize, const N_LODS: usize> (
        params: WorldCreateParams<N_CHUNK_LVLS, N_LODS, VE>
    ) -> Option<LockedWorld<VE>> {
        Some(LockedWorld {
            mem_grid: MemoryGrid::<VE>::new::<N_CHUNK_LVLS, N_LODS>(params)?,
            camera: Camera::default(),
            meta: WorldMetadata { },
        })
    }

    pub fn move_camera(&mut self, camera_controller: &mut CameraController, dt: Duration) {
        self.camera.apply_controller_updates(camera_controller, dt);
        self.mem_grid.move_grid(&mut self.camera);
    }

    pub fn unlock(self) -> World<VE> {
        World {
            mem_grid: self.mem_grid.to_virtual(),
            meta: self.meta,
            camera: self.camera,
        }
    }
}
