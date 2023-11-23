use cgmath::{Array, EuclideanSpace, Point3, Vector3};
use num_traits::Zero;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::camera::Camera;
use crate::world::mem_grid::{PhysicalMemoryGrid, VirtualMemoryGrid};
use crate::world::mem_grid::utils::{amod, pos_for_index, pos_index};
use crate::world::{TLCPos, TLCVector, VoxelPos};

use super::layer_contents::MemoryGridLayerContents;

#[derive(Clone)]
pub struct MemoryGridLayerMetadata<E: Sized> {
    size: usize, // Size of grid in number of TLCs, 1 more than the rendered area size in each dimension for pre-loading
    offsets: Vector3<usize>,
    loaded_upper_chunks: bool,
    extra: E,
}

pub struct MemoryGridLayer<D: MemoryGridLayerContents<VE>, MD: Sized, VE: VoxelTypeEnum> {
    metadata: MemoryGridLayerMetadata<MD>,
    chunks: Vec<D>,
}

// TODO: might be able to remove VirtualMemoryGrid trait and just use this for everything
pub struct VirtualMemoryGridForLayer<D: MemoryGridLayerContents<VE>, MD: Sized, VE: VoxelTypeEnum> {
    metadata: MemoryGridLayerMetadata<MD>,
    chunks: Vec<Option<D>>,
}


impl<D: MemoryGridLayerContents<VE>, MD: Sized, VE: VoxelTypeEnum> MemoryGridLayer<D, MD, VE> {
    pub fn new_raw(metadata: MemoryGridLayerMetadata<MD>, chunks: Vec<D>) -> Self {
        MemoryGridLayer { metadata, chunks }
    }
}
impl<D: MemoryGridLayerContents<VE>, MD: Sized, VE: VoxelTypeEnum> PhysicalMemoryGrid<VE> for MemoryGridLayer<D, MD, VE> {
    type Virtual = VirtualMemoryGridForLayer<D, MD, VE>;

    fn shift_offsets(&mut self, shift: Vector3<i64>) {
        if !shift.is_zero() {
            todo!();
        }
    }

    fn to_virtual(self) -> Self::Virtual {
        self.to_virtual_for_size(self.metadata.size-1)
    }

    fn to_virtual_for_size(self, grid_size: usize) -> Self::Virtual {
        let mut grid = vec![None; (grid_size).pow(3)];

        for (chunk_i, chunk) in self.chunks.into_iter().enumerate() {
            // If this fff is smaller than full grid, add padding to virtual position so it
            // is centered
            let virtual_pos = self.metadata.virtual_grid_pos_for_index(chunk_i, grid_size).0;
            grid[pos_index(virtual_pos, grid_size-1)] = chunk;
        }

        VirtualMemoryGridForLayer {
            metadata: self.metadata,
            chunks: grid,
        }
    }
}

impl<D: MemoryGridLayerContents<VE>, MD: Sized, VE: VoxelTypeEnum> VirtualMemoryGridForLayer<D, MD, VE> {
    pub fn destroy(self) -> (MemoryGridLayerMetadata<MD>, Vec<D>) {
        (self.metadata, self.chunks)
    }
}
impl<D: MemoryGridLayerContents<VE>, MD: Sized, VE: VoxelTypeEnum> VirtualMemoryGrid<VE> for VirtualMemoryGridForLayer<D, MD, VE> {
    type Physical = MemoryGridLayer<D, MD, VE>;

    fn load_or_generate_tlc(&self, voxel_output: &mut ChunkVoxelIDs, tlc: TLCPos<i64>) {
        todo!()
    }

    fn reload_all(&mut self) {
        todo!()
    }

    fn set_voxel(&mut self, position: VoxelPos<usize>, voxel_type: VE, tlc_size: usize) -> Option<()> {
        for (i, chunk_o) in self.chunks.iter().enumerate() {
            match chunk_o {
                None => {},
                Some(chunk) => {
                    let chunk_pos = position.0.cast::<i64>().unwrap() - pos_for_index(i, self.metadata.size).cast::<i64>().unwrap() * tlc_size as i64;
                    chunk.set_voxel(VoxelPos(chunk_pos), voxel_type);
                }
            }
        }

        Some(())
    }

    fn lock(self) -> Option<Self::Physical> {
        Some(MemoryGridLayer {
            chunks: self.chunks,
            metadata: self.metadata,
        })
    }
}


impl<E: Sized> MemoryGridLayerMetadata<E> {
    pub fn calc_offsets(start_tlc: TLCPos<i64>, size: usize) -> Vector3<usize> {
        amod(start_tlc.0.to_vec(), size)
    }

    pub fn grid_pos_for_virtual_grid_pos(&self, tlc_pos: TLCVector<usize>, vgrid_size: usize) -> TLCVector<usize> {
        let local_vgrid_pos = tlc_pos - Vector3::from_value(
            if vgrid_size > self.size { (vgrid_size - (self.size - 1)) / 2 }
            else { 0 }
        );
        TLCVector((local_vgrid_pos.0 + self.offsets) % self.size)
    }

    pub fn index_for_virtual_grid_pos(&self, pos: TLCVector<usize>, vgrid_size: usize) -> usize {
        pos_index(self.grid_pos_for_virtual_grid_pos(pos, vgrid_size).0, self.size)
    }

    pub fn virtual_grid_pos_for_grid_pos(&self, pos: TLCVector<usize>, vgrid_size: usize) -> TLCVector<usize> {
        let local_vgrid_pos = amod(pos.0.cast::<i64>().unwrap() - self.offsets.cast::<i64>().unwrap(), self.size);
        TLCVector(
            local_vgrid_pos + Vector3::<usize>::from_value(
                if self.size < vgrid_size {
                    (vgrid_size - self.size) / 2
                }
                else { 0 }
            )
        )
    }

    pub fn virtual_grid_pos_for_index(&self, index: usize, vgrid_size: usize) -> TLCVector<usize> {
        self.virtual_grid_pos_for_grid_pos(TLCVector(pos_for_index(index, self.size)), vgrid_size)
    }
}