use std::ops::Deref;
use derive_new::new;
use crate::world::{TLCVector};
use crate::world::mem_grid::layer::{MemoryGridLayerData, MemoryGridLayerMetadata, PhysicalMemoryGridLayer};

pub mod layer;
mod layer_set;
mod utils;
pub mod voxel;

#[derive(new)]
pub struct PhysicalMemoryGridStruct<D, MD: MemoryGridMetadata> {
    data: D,
    metadata: MD,
}

#[derive(new)]
pub struct VirtualMemoryGridStruct<C, MD: MemoryGridMetadata> {
    data: Vec<Option<C>>,
    metadata: MD
}

pub trait MemoryGridMetadata {
    fn size(&self) -> usize;
}


impl<D, MD: MemoryGridMetadata> PhysicalMemoryGridStruct<D, MD> {
    pub fn deconstruct(self) -> (D, MD) { (self.data, self.metadata) }
}

impl<C, MD: MemoryGridMetadata> VirtualMemoryGridStruct<C, MD> {
    pub fn deconstruct(self) -> (Vec<Option<C>>, MD) { (self.data, self.metadata) }
}

impl<D, MD: MemoryGridMetadata> PhysicalMemoryGridLayer<D, MD> {
    pub fn size(&self) -> usize { self.metadata.size() }
}
impl<C, MD: MemoryGridMetadata> VirtualMemoryGridStruct<C, MD> {
    fn size(&self) -> usize { self.metadata.size() }
}


pub trait PhysicalMemoryGrid<C, MD: MemoryGridMetadata>: Deref<Target = PhysicalMemoryGridStruct<C, MD>> {
    type ChunkLoadQueue;
    fn shift(&mut self, shift: TLCVector<i32>, load: TLCVector<i32>) -> Self::ChunkLoadQueue;
}


pub trait ToVirtual<C, MD: MemoryGridMetadata> {
    fn to_virtual(self) -> VirtualMemoryGridStruct<Option<C>, MD> {
        self.to_virtual_for_size(self.size())
    }

    fn to_virtual_for_size(self, grid_size: usize) -> VirtualMemoryGridStruct<Option<C>, MD>;
}

pub trait FromVirtual<C, MD: MemoryGridMetadata>: Sized {
    fn from_virtual(virtual_grid: VirtualMemoryGridStruct<C, MD>) -> Self {
        Self::from_virtual_for_size(virtual_grid, virtual_grid.metadata.size())
    }

    fn from_virtual_for_size(virtual_grid: VirtualMemoryGridStruct<C, MD>, grid_size: usize) -> Self;
}
