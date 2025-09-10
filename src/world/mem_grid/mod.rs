use std::ops::Deref;
use derive_new::new;
use crate::world::{TLCPos, TLCVector};
use crate::world::loader::ChunkLoadQueueItem;
use crate::world::mem_grid::utils::index_for_pos;
use crate::world::mem_grid::voxel::grid::GlobalVoxelPos;

pub mod layer;
mod layer_set;
pub mod utils;
pub mod voxel;


#[derive(new, Clone)]
pub struct PhysicalMemoryGridStruct<D, MD: MemoryGridMetadata> {
    pub data: D,
    pub metadata: MD,
}

#[derive(new, Clone)]
pub struct VirtualMemoryGridStruct<C: Placeholder, MD: MemoryGridMetadata> {
    pub chunks: Vec<Option<C>>,
    pub metadata: MD
}

pub trait MemoryGridMetadata {
    fn size(&self) -> usize;
    fn start_tlc(&self) -> TLCPos<i64>;
}


impl<D, MD: MemoryGridMetadata> PhysicalMemoryGridStruct<D, MD> {
    pub fn deconstruct(self) -> (D, MD) { (self.data, self.metadata) }
    pub fn size(&self) -> usize { self.metadata.size() }
    pub fn start_tlc(&self) -> TLCPos<i64> { self.metadata.start_tlc() }
}

impl<C: Placeholder, MD: MemoryGridMetadata> VirtualMemoryGridStruct<C, MD> {
    pub fn deconstruct(self) -> (Vec<Option<C>>, MD) { (self.chunks, self.metadata) }
    pub fn size(&self) -> usize { self.metadata.size() }
    pub fn start_tlc(&self) -> TLCPos<i64> { self.metadata.start_tlc() }

    pub fn chunk_index(&self, global_tlc_pos: TLCPos<i64>) -> Option<usize> {
        Some(
            index_for_pos(
                (global_tlc_pos.0 - self.start_tlc().0).cast::<usize>()?,
                self.size(),
            )
        )
    }

    pub fn chunk_for(&self, global_pos: GlobalVoxelPos) -> Option<&C> {
        self.chunks.get(self.chunk_index(global_pos.tlc)?)?.as_ref()
    }

    pub fn chunk_for_mut(&mut self, global_pos: GlobalVoxelPos) -> Option<&mut C> {
        let idx = self.chunk_index(global_pos.tlc)?;
        self.chunks.get_mut(idx)?.as_mut()
    }
}


pub trait PhysicalMemoryGrid: Deref<Target = PhysicalMemoryGridStruct<Self::Data, Self::Metadata>> + Sized {
    type Data;
    type Metadata: MemoryGridMetadata;
    type ChunkLoadQueueItemData: Clone + Send + 'static;
    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
}


pub trait ToVirtual<C: Placeholder, MD: MemoryGridMetadata>: PhysicalMemoryGrid {
    fn to_virtual(self) -> VirtualMemoryGridStruct<C, MD> {
        let size = self.size();
        self.to_virtual_for_size(size)
    }

    fn to_virtual_for_size(self, grid_size: usize) -> VirtualMemoryGridStruct<C, MD>;
}

pub trait FromVirtual<C: Placeholder, MD: MemoryGridMetadata>: PhysicalMemoryGrid {
    fn from_virtual(virtual_grid: VirtualMemoryGridStruct<C, MD>) -> Self {
        let size = virtual_grid.size();
        Self::from_virtual_for_size(virtual_grid, size)
    }

    fn from_virtual_for_size(virtual_grid: VirtualMemoryGridStruct<C, MD>, vgrid_size: usize) -> Self;
}


pub trait Placeholder {
    fn placeholder(&self) -> Self;
}
