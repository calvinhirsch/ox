use crate::loader::{ChunkLoadQueueItem, LayerChunk};
use crate::world::mem_grid::utils::{amod, cubed, index_for_pos};
use crate::world::mem_grid::{EditMemoryGridChunk, MemoryGrid, MemoryGridLoadChunks};
use crate::world::{BufferChunkState, TlcPos, TlcVector};
use cgmath::{EuclideanSpace, Point3, Vector3};
use getset::{Getters, MutGetters};

use super::MemGridShift;

#[derive(Clone, Debug, Getters)]
pub struct MemoryGridLayerMetadata<MD> {
    #[getset(get = "pub")]
    start_tlc: TlcPos<i64>,
    #[getset(get = "pub")]
    size: usize, // grid size (or render area size + 1)
    #[getset(get = "pub")]
    offsets: TlcVector<usize>,
    #[getset(get = "pub")]
    extra: MD,
}

/// Memory grid layers are layers of the memory grid that each can have their own size (AKA load/render distance).
/// Whole values of type `C` will be pulled out and loaded, so `C` should not contain different peices of data that
/// you want to be loaded independently; all data in a single instance of `C` will be loaded together.
#[derive(Debug, Getters, MutGetters)]
pub struct MemoryGridLayer<C, MD = (), S = ()> {
    /// data for each chunk
    #[getset(get = "pub", get_mut = "pub")]
    chunks: Vec<LayerChunk<C>>,
    /// metadata, not exposed for mutation during editing
    #[getset(get = "pub")]
    metadata: MemoryGridLayerMetadata<MD>,
    /// global mutable state, exposed for mutation during editing
    #[getset(get = "pub", get_mut = "pub")]
    state: S,
}

impl<C, MD, S> MemoryGridLayer<C, MD, S> {
    pub fn new(
        chunks: Vec<LayerChunk<C>>,
        start_tlc: TlcPos<i64>,
        size: usize,
        extra_metadata: MD,
        state: S,
    ) -> Self {
        debug_assert!(chunks.len() == cubed(size));
        MemoryGridLayer {
            chunks,
            metadata: MemoryGridLayerMetadata {
                start_tlc,
                size,
                offsets: Self::calc_offsets_for(start_tlc, size),
                extra: extra_metadata,
            },
            state,
        }
    }

    pub fn chunks_and_state_mut(&mut self) -> (&mut Vec<LayerChunk<C>>, &mut S) {
        (&mut self.chunks, &mut self.state)
    }

    pub fn calc_offsets_for(start_tlc: TlcPos<i64>, size: usize) -> TlcVector<usize> {
        TlcVector(amod(start_tlc.0, size).to_vec())
    }

    pub fn calc_offsets(&self) -> TlcVector<usize> {
        Self::calc_offsets_for(self.metadata().start_tlc, self.metadata().size)
    }

    /// Given a chunk position in this layer's virtual grid, return the physical grid position.
    /// (basically, just apply the current offsets)
    pub fn grid_pos_for_vgrid_pos(&self, vgrid_pos: TlcVector<usize>) -> TlcVector<usize> {
        TlcVector((vgrid_pos.0.map(|a| a as usize) + self.metadata.offsets.0) % self.metadata.size)
    }

    pub fn index_for_grid_pos(&self, grid_pos: TlcVector<usize>) -> usize {
        index_for_pos(Point3::from_vec(grid_pos.0.map(|a| a as u32)), self.size())
    }

    pub fn index_for_vgrid_pos(&self, vgrid_pos: TlcVector<usize>) -> usize {
        self.index_for_grid_pos(self.grid_pos_for_vgrid_pos(vgrid_pos))
    }

    pub fn chunk_vgrid_pos(
        &self,
        pos: TlcPos<i64>,
        buffer_chunk_states: [BufferChunkState; 3],
    ) -> Option<TlcVector<usize>> {
        // dbg!((pos, self.start_tlc(), self.size()));
        let mut i = 0;
        if let Point3 {
            x: Some(x),
            y: Some(y),
            z: Some(z),
        } = (pos.0 - self.start_tlc().0.to_vec()).map(|a| {
            let state = buffer_chunk_states[i];
            i += 1;
            if a < 0 {
                if a == -1 && state == BufferChunkState::LoadedLower {
                    Some(self.size() - 1)
                } else {
                    None
                }
            } else if a >= self.size() as i64 - 1 {
                if a == self.size() as i64 - 1 && state == BufferChunkState::LoadedUpper {
                    Some(self.size() - 1)
                } else {
                    None
                }
            } else {
                Some(a as usize)
            }
        }) {
            // dbg!(("vgrid_pos", x, y, z));
            Some(TlcVector(Vector3 { x, y, z }))
        } else {
            // dbg!("failed");
            None
        }
    }
}

impl<C, MD, S> MemoryGridLoadChunks for MemoryGridLayer<C, MD, S> {
    type ChunkLoadQueueItemData = ();

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        let start_tlc = self.metadata().start_tlc.0;
        let size = self.metadata().size;

        println!("{:?}  {}", start_tlc, size);

        (0..size as i64 - 1)
            .flat_map(|x| {
                (0..size as i64 - 1).flat_map(move |y| {
                    (0..size as i64 - 1).map(move |z| ChunkLoadQueueItem {
                        pos: TlcPos(start_tlc + Vector3 { x, y, z }),
                        data: (),
                    })
                })
            })
            .collect()
    }

    fn shift(
        &mut self,
        shift: &MemGridShift,
    ) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        // Apply the shift to start_tlc
        self.metadata.start_tlc.0 += shift.offset_delta().cast::<i64>().unwrap();

        // Apply the shift to grid offset
        self.metadata.offsets = self.calc_offsets();

        // Queue all the chunks that need to be loaded based on the shift
        shift.collect_chunks_to_load(self.metadata().size, self.metadata().start_tlc, |pos| {
            ChunkLoadQueueItem { pos, data: () }
        })
    }
}

impl<C, MD, S> MemoryGrid for MemoryGridLayer<C, MD, S> {
    fn size(&self) -> usize {
        self.metadata().size
    }
    fn start_tlc(&self) -> TlcPos<i64> {
        self.metadata().start_tlc
    }
}

#[derive(Debug)]
pub struct DefaultLayerChunkEditor<'a, C, MD = (), S = ()> {
    pub chunk: &'a mut LayerChunk<C>,
    pub chunk_idx: usize,
    pub metadata: &'a MemoryGridLayerMetadata<MD>,
    pub layer_state: &'a mut S,
}

impl<C, MD, S> EditMemoryGridChunk for MemoryGridLayer<C, MD, S> {
    type ChunkEditor<'a> = DefaultLayerChunkEditor<'a, C, MD, S>
        where
            Self: 'a;

    fn edit_chunk(
        &mut self,
        pos: TlcPos<i64>,
        buffer_chunk_states: [BufferChunkState; 3],
    ) -> Option<Self::ChunkEditor<'_>> {
        let vgrid_pos = self.chunk_vgrid_pos(pos, buffer_chunk_states)?;
        let chunk_idx = self.index_for_vgrid_pos(vgrid_pos);
        Some(DefaultLayerChunkEditor {
            chunk: self.chunks.get_mut(chunk_idx)?,
            chunk_idx,
            metadata: &self.metadata,
            layer_state: &mut self.state,
        })
    }
}
