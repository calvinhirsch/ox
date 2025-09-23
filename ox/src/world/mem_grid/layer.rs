use crate::world::loader::{ChunkLoadQueueItem, LayerChunk};
use crate::world::mem_grid::utils::{amod, cubed, index_for_pos};
use crate::world::mem_grid::{MemoryGrid, MemoryGridChunkEditor};
use crate::world::{TlcPos, TlcVector};
use cgmath::{Array, EuclideanSpace, Point3, Vector3};
use getset::{Getters, MutGetters};

use super::{ChunkEditor, MemGridShift};

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
                offsets: TlcVector(Self::calc_offsets(start_tlc, size)),
                extra: extra_metadata,
            },
            state,
        }
    }

    pub fn chunks_and_state_mut(&mut self) -> (&mut Vec<LayerChunk<C>>, &mut S) {
        (&mut self.chunks, &mut self.state)
    }

    pub fn calc_offsets(start_tlc: TlcPos<i64>, size: usize) -> Vector3<usize> {
        amod(start_tlc.0, size).to_vec()
    }

    pub fn grid_pos_for_virtual_grid_pos(
        &self,
        tlc_pos: TlcVector<usize>,
        vgrid_size: usize,
    ) -> TlcVector<usize> {
        let local_vgrid_pos = tlc_pos.0
            - Vector3::from_value(if vgrid_size > self.metadata.size {
                (vgrid_size - (self.metadata.size - 1)) / 2
            } else {
                0
            });
        TlcVector((local_vgrid_pos + self.metadata.offsets.0) % self.metadata.size)
    }

    pub fn virtual_grid_pos_for_grid_pos(
        &self,
        pos: TlcPos<u32>,
        vgrid_size: usize,
    ) -> TlcPos<u32> {
        let local_vgrid_pos = (pos.0 + Vector3::from_value(self.metadata.size as u32)
            - self.metadata.offsets.0.map(|o| o as u32))
            % self.metadata.size as u32;
        TlcPos(
            local_vgrid_pos
                + Vector3::from_value(if self.metadata.size < vgrid_size {
                    ((vgrid_size - self.metadata.size) / 2) as u32
                } else {
                    0u32
                }),
        )
    }
}

impl<C, MD, S> MemoryGrid for MemoryGridLayer<C, MD, S> {
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
        self.metadata.offsets = TlcVector(
            amod(
                Point3::from_vec(
                    self.metadata().offsets.0.cast::<i64>().unwrap()
                        + shift.offset_delta().cast::<i64>().unwrap(),
                ),
                self.size(),
            )
            .to_vec(),
        );

        // Queue all the chunks that need to be loaded based on the shift
        shift.collect_chunks_to_load(self.metadata().size, self.metadata().start_tlc, |pos| {
            ChunkLoadQueueItem { pos, data: () }
        })
    }

    fn size(&self) -> usize {
        self.metadata().size
    }
    fn start_tlc(&self) -> TlcPos<i64> {
        self.metadata().start_tlc
    }
}

impl<
        'a,
        C: 'static,
        MD: 'static,
        S: 'static,
        CE: ChunkEditor<&'a mut LayerChunk<C>, &'a MemoryGridLayerMetadata<MD>, &'a mut S>,
    > MemoryGridChunkEditor<'a, MemoryGridLayer<C, MD, S>> for Option<CE>
{
    fn edit_chunk_for_size(
        mem_grid: &'a mut MemoryGridLayer<C, MD, S>,
        grid_size: usize,
        pos: TlcVector<usize>,
    ) -> Self {
        let physical_pos = mem_grid.grid_pos_for_virtual_grid_pos(pos, grid_size);
        let physical_grid_size = *mem_grid.metadata().size();
        let chunk_idx = index_for_pos(
            Point3::from_vec(physical_pos.0.map(|a| a as u32)),
            physical_grid_size,
        );
        mem_grid
            .chunks
            .get_mut(chunk_idx)
            .map(|c| CE::edit(c, &mem_grid.metadata, &mut mem_grid.state, chunk_idx))
    }
}
