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
pub struct MemoryGridEditor<CE, MD> {
    pub chunks: Vec<Option<CE>>,
    pub size: usize,
    pub start_tlc: TLCPos<i64>,
    #[get = "pub"]
    metadata: MD,
}
pub trait MemoryGridEditorTrait<CE, MD> {
    fn this(&self) -> &MemoryGridEditor<CE, MD>;
    fn this_mut(&mut self) -> &mut MemoryGridEditor<CE, MD>;
}
impl<CE, MD> MemoryGridEditorTrait<CE, MD> for MemoryGridEditor<CE, MD> {
    fn this(&self) -> &MemoryGridEditor<CE, MD> { self }
    fn this_mut(&mut self) -> &mut MemoryGridEditor<CE, MD> { self }
}


impl<CE, MD> MemoryGridEditor<CE, MD> {
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

pub trait NewMemoryGridEditor<'a, MG: MemoryGrid>: Sized {
    fn for_grid(mem_grid: &'a mut MG) -> Self {
        let size = mem_grid.size();
        Self::for_grid_with_size(mem_grid, size)
    }
    fn for_grid_with_size(mem_grid: &'a mut MG, grid_size: usize) -> Self;
}


pub trait MemoryGrid {
    type ChunkLoadQueueItemData: ChunkLoadQueueItemData + 'static;
    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
    fn size(&self) -> usize;
    fn start_tlc(&self) -> TLCPos<i64>;
}

// pub trait EditMemoryGrid<CE, MD>: MemoryGrid {
//     fn edit(&mut self) -> MemoryGridEditor<CE, MD> {
//         self.edit_for_size(self.size())
//     }
//     fn edit_for_size(&mut self, grid_size: usize) -> MemoryGridEditor<CE, MD>;
// }


pub trait ChunkEditor<'a>: Sized {
    type Capsule: 'static;

    /// Called when chunk is initially queued for loading. Should set some internal state that makes it clear this data
    /// is no longer valid (e.g. in VoxelLOD set loaded = false).
    fn on_queued_for_loading(&mut self);

    fn new_from_capsule(capsule: &'a mut Self::Capsule) -> Self;

    /// Generate placeholders for all fields of self, swap them in, and return the original data in a capsule.
    fn replace_with_placeholder(&mut self) -> Self::Capsule;

    /// Given a capsule, move the data from that capsule into the chunk this editor is for
    fn replace_with_capsule(&mut self, capsule: Self::Capsule);

    /// Called before calling replace_with_placeholder, should check that there are no placeholder values.
    fn ok_to_replace_with_placeholder(&self) -> bool;

    /// Called before calling replace_with_capsule with a loaded capsule, should ensure that all data fields are
    /// placeholders, otherwise something has gone wrong (only used in debug mode).
    #[cfg(debug_assertions)]
    fn ok_to_replace_with_capsule(&self) -> bool;
}


pub trait Placeholder: Sized {
    fn replace_with_placeholder(&mut self) -> Self;
    fn is_placeholder(&self) -> bool;
}