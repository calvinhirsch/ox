use std::marker::PhantomData;
use derive_new::new;
use getset::Getters;
use crate::world::{TLCPos, TLCVector};
use crate::world::loader::{ChunkLoadQueueItem, ChunkLoadQueueItemData};
use crate::world::mem_grid::utils::index_for_pos;
use crate::world::mem_grid::voxel::grid::GlobalVoxelPos;

pub mod layer;
mod layer_set;
pub mod utils;
pub mod voxel;


#[derive(new, Clone, Getters)]
pub struct MemoryGridEditor<'a, CE, MD> {
    #[new(default)]
    pub lifetime: PhantomData<&'a CE>,
    pub chunks: Vec<Option<CE>>,
    pub size: usize,
    pub start_tlc: TLCPos<i64>,
    #[get = "pub"]
    metadata: MD,
}


impl<'a, CE, MD> MemoryGridEditor<'a, CE, MD> {
    pub fn chunk_index_in(global_tlc_pos: TLCPos<i64>, grid_start_tlc: TLCPos<i64>, grid_size: usize) -> Option<usize> {
        Some(
            index_for_pos(
                (global_tlc_pos.0 - grid_start_tlc.0).cast::<usize>()?,
                grid_size,
            )
        )
    }

    pub fn chunk_index(&self, global_tlc_pos: TLCPos<i64>) -> Option<usize> {
        Self::chunk_index_in(global_tlc_pos, self.start_tlc, self.size)
    }

    pub fn chunk(&self, global_pos: GlobalVoxelPos) -> Option<&CE> {
        let idx = self.chunk_index(global_pos.tlc)?;
        self.chunks.get(idx)?.as_ref()
    }

    pub fn chunk_mut(&mut self, global_pos: GlobalVoxelPos) -> Option<&mut CE> {
        let idx = self.chunk_index(global_pos.tlc)?;
        self.chunks.get_mut(idx)?.as_mut()
    }
}


pub trait MemoryGrid {
    type ChunkLoadQueueItemData: ChunkLoadQueueItemData + 'static;
    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
    fn size(&self) -> usize;
    fn start_tlc(&self) -> TLCPos<i64>;
}

pub trait EditMemoryGrid<'a, CE, MD>: MemoryGrid {
    fn edit(&'a mut self) -> MemoryGridEditor<CE, MD> {
        self.edit_for_size(self.size())
    }
    fn edit_for_size(&'a mut self, grid_size: usize) -> MemoryGridEditor<CE, MD>;
}


// pub trait Placeholder {
//     /// Generate a placeholder that can be swapped in for self
//     fn placeholder(&self) -> Self;
// }
pub trait ChunkCapsule<'a, E: ChunkEditor<'a>> {
    fn edit(&'a mut self) -> E;
    fn move_into_chunk(self, dest_chunk_editor: &mut E);
}

pub trait ChunkEditor<'a>: Sized {
    type Capsule: ChunkCapsule<'a, Self>;

    /// Generate placeholders for all fields of self, swap them in, and return the original data in a capsule.
    fn replace_with_placeholder(&mut self) -> Self::Capsule;
}
