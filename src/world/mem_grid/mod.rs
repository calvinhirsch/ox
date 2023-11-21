use cgmath::{Array, EuclideanSpace, Vector3};

use crate::voxel_type::VoxelTypeEnum;
use crate::world::{
    TLCVector, TLCPos,
};

pub mod utils;
use utils::{pos_index, amod, pos_for_index};
use crate::world::mem_grid::physical_grid::MemoryGridLayerExtraMetadata;

pub mod rendering;

pub mod physical_grid;
pub mod virtual_grid;

mod world_data;


#[derive(Clone)]
pub struct MemoryGridMetadata {
    pub size: usize,
    chunk_size: usize,
    tlc_size: usize,
    load_thresh_dist: usize,
    n_chunk_lvls: usize,
    n_lods: usize,
    start_tlc: TLCPos<i64>,
    lod_block_fill_thresh: u8,  // up to 8
}

#[derive(Clone)]
pub struct MemoryGridLayerMetadata<P: Sized, E: MemoryGridLayerExtraMetadata<P>> {
    size: usize, // Size of grid in number of TLCs, 1 more than the rendered area size in each dimension for pre-loading
    offsets: Vector3<usize>,
    loaded_upper_chunks: bool,
    extra: E,
}

pub struct MemoryGridLayerCreateParams<E: Sized> {
    size: usize,
    extra: E,
}

impl<P: Sized, E: MemoryGridLayerExtraMetadata<P>> MemoryGridLayerMetadata<P, E> {
    pub fn new(params: MemoryGridLayerCreateParams<E>, grid_meta: &MemoryGridMetadata) -> MemoryGridLayerMetadata<P, E> {
        MemoryGridLayerMetadata {
            size: params.size,
            offsets: MemoryGridLayerMetadata::calc_offsets(grid_meta.start_tlc, params.size),
            loaded_upper_chunks: false,
            extra: E::new(params),
        }
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
    
    pub fn calc_offsets(start_tlc: TLCPos<i64>, size: usize) -> Vector3<usize> {
        amod(start_tlc.0.to_vec(), size)
    }
}

pub trait NewChunkData<P: Sized> {
    fn new(tlc_size: usize, params: &MemoryGridLayerCreateParams<P>) -> Self;
}

/// Data for a single top level chunk for a single layer of a memory grid
pub trait MemoryGridLayerChunkData<VE: VoxelTypeEnum> {
    fn set_voxel(&mut self, index: usize, voxel_type: VE) -> Option<()>;
}
