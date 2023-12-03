use std::ops::Deref;
use derive_new::new;
use crate::world::{TLCVector};
use crate::world::loader::ChunkLoadQueueItem;

pub mod layer;
mod layer_set;
pub mod utils;
pub mod voxel;

#[derive(new)]
pub struct PhysicalMemoryGridStruct<D, MD: MemoryGridMetadata> {
    pub data: D,
    pub metadata: MD,
}

#[derive(new)]
pub struct VirtualMemoryGridStruct<C, MD: MemoryGridMetadata> {
    pub chunks: Vec<Option<C>>,
    pub metadata: MD
}

pub trait MemoryGridMetadata {
    fn size(&self) -> usize;
}


impl<D, MD: MemoryGridMetadata> PhysicalMemoryGridStruct<D, MD> {
    pub fn deconstruct(self) -> (D, MD) { (self.data, self.metadata) }
    pub fn size(&self) -> usize { self.metadata.size() }
}

impl<C, MD: MemoryGridMetadata> VirtualMemoryGridStruct<C, MD> {
    pub fn deconstruct(self) -> (Vec<Option<C>>, MD) { (self.chunks, self.metadata) }
    pub fn size(&self) -> usize { self.metadata.size() }
}


pub trait PhysicalMemoryGrid: Deref<Target = PhysicalMemoryGridStruct<Self::Data, Self::Metadata>> + Sized {
    type Data;
    type Metadata: MemoryGridMetadata;
    type ChunkLoadQueueItemData;
    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
}


pub trait ToVirtual<C, MD: MemoryGridMetadata>: PhysicalMemoryGrid {
    fn to_virtual(self) -> VirtualMemoryGridStruct<C, MD> {
        self.to_virtual_for_size(self.size())
    }

    fn to_virtual_for_size(self, grid_size: usize) -> VirtualMemoryGridStruct<C, MD>;
}

pub trait FromVirtual<C, MD: MemoryGridMetadata>: PhysicalMemoryGrid {
    fn from_virtual(virtual_grid: VirtualMemoryGridStruct<C, MD>) -> Self {
        Self::from_virtual_for_size(virtual_grid, virtual_grid.metadata.size())
    }

    fn from_virtual_for_size(virtual_grid: VirtualMemoryGridStruct<C, MD>, vgrid_size: usize) -> Self;
}