use crate::world::loader::ChunkLoadQueueItem;
use crate::world::mem_grid::utils::index_for_pos;
use crate::world::mem_grid::voxel::grid::GlobalVoxelPos;
use crate::world::{TLCPos, TLCVector};
use derive_new::new;
use getset::Getters;

pub mod layer;
mod layer_set;
pub mod utils;
pub mod voxel;

#[derive(new, Clone, Getters)]
pub struct MemoryGridEditor<CE, MD> {
    pub chunks: Vec<CE>,
    pub size: usize,
    pub start_tlc: TLCPos<i64>,
    #[get = "pub"]
    metadata: MD,
}

impl<CE, MD> MemoryGridEditor<CE, MD> {
    pub fn chunk_index_in(
        global_tlc_pos: TLCPos<i64>,
        grid_start_tlc: TLCPos<i64>,
        grid_size: usize,
    ) -> Option<usize> {
        Some(index_for_pos(
            (global_tlc_pos.0 - grid_start_tlc.0).cast::<usize>()?,
            grid_size,
        ))
    }

    pub fn chunk_index(&self, global_tlc_pos: TLCPos<i64>) -> Option<usize> {
        Self::chunk_index_in(global_tlc_pos, self.start_tlc, self.size)
    }

    pub fn chunk(&self, global_pos: GlobalVoxelPos) -> Result<&CE, ()> {
        let idx = self.chunk_index(global_pos.tlc).ok_or(())?;
        Ok(self.chunks.get(idx).unwrap())
    }

    pub fn chunk_mut(&mut self, global_pos: GlobalVoxelPos) -> Result<&mut CE, ()> {
        let idx = self.chunk_index(global_pos.tlc).ok_or(())?;
        Ok(self.chunks.get_mut(idx).unwrap())
    }
}

pub trait MemoryGrid {
    type ChunkLoadQueueItemData;

    /// Queue all chunks in memory grid to be loaded. Does not queue buffer chunks or change their state.
    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
    fn shift(
        &mut self,
        shift: TLCVector<i32>,
        load_in_from_edge: TLCVector<i32>,
        load_buffer: [bool; 3],
    ) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
    fn size(&self) -> usize;
    fn start_tlc(&self) -> TLCPos<i64>;
}

pub trait MemoryGridEditorChunk<'a, MG: MemoryGrid, MD>: Sized {
    fn edit_grid(mem_grid: &'a mut MG) -> MemoryGridEditor<Self, MD> {
        let size = mem_grid.size();
        Self::edit_grid_with_size(mem_grid, size)
    }
    fn edit_grid_with_size(mem_grid: &'a mut MG, grid_size: usize) -> MemoryGridEditor<Self, MD>;
}

pub trait ChunkEditor<C, MD> {
    fn edit(chunk: C, metadata: MD) -> Self;
}
