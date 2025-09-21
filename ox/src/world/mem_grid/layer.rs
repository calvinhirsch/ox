use std::time::Instant;

use crate::world::loader::{ChunkLoadQueueItem, LayerChunk};
use crate::world::mem_grid::utils::{amod, cubed, index_for_pos, pos_for_index};
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor, MemoryGridEditorChunk};
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
pub struct MemoryGridLayer<C, MD = ()> {
    #[getset(get = "pub", get_mut = "pub")]
    chunks: Vec<LayerChunk<C>>,
    #[getset(get = "pub")]
    metadata: MemoryGridLayerMetadata<MD>,
}

impl<C, MD> MemoryGridLayer<C, MD> {
    pub fn new(
        chunks: Vec<LayerChunk<C>>,
        start_tlc: TlcPos<i64>,
        size: usize,
        extra_metadata: MD,
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
        }
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

impl<C, MD> MemoryGrid for MemoryGridLayer<C, MD> {
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

// impl<'a, C, CE, MD> EditMemoryGrid<'a, CE, &'a MD> for MemoryGridLayer<C, MD>
// where
//     C: 'static,
//     CE: From<&'a mut C>,
// {
//     fn edit_with_size(&'a mut self, grid_size: usize) -> MemoryGridEditor<CE, &'a MD> {
//         let mut vgrid: Vec<Option<CE>> = (0..grid_size.pow(3)).map(|_| None).collect();

//         // If this layer is smaller than full grid, add padding to virtual position so it
//         // is centered
//         let vgrid_positions: Vec<_> = (0..cubed(self.size))
//             .map(|i| {
//                 self.virtual_grid_pos_for_grid_pos(
//                     TLCVector(pos_for_index(i, self.size)),
//                     grid_size,
//                 )
//             })
//             .collect();

//         for (chunk_data, vgrid_pos) in self.chunks.iter_mut().zip(vgrid_positions) {
//             vgrid[index_for_pos(vgrid_pos.0, grid_size)] = Some(CE::from(chunk_data));
//         }

//         MemoryGridEditor {
//             chunks: vgrid,
//             size: grid_size,
//             start_tlc: self.start_tlc,
//             metadata: &self.metadata,
//         }
//     }
// }

fn edit_grid_layer_with_size<
    'a,
    C,
    MD,
    CE,
    F: Fn(&'a mut LayerChunk<C>, &'a MemoryGridLayerMetadata<MD>) -> CE,
>(
    mem_grid: &'a mut MemoryGridLayer<C, MD>,
    grid_size: usize,
    edit_chunk_f: F,
) -> MemoryGridEditor<Option<CE>, &'a MemoryGridLayerMetadata<MD>> {
    let start = Instant::now();

    let mut vgrid: Vec<Option<CE>> = (0..cubed(grid_size)).map(|_| None).collect();

    dbg!(mem_grid.metadata().size);
    dbg!(Instant::now() - start);

    // If this layer is smaller than full grid, add padding to virtual position so it
    // is centered
    let vgrid_positions: Vec<_> = (0..cubed(mem_grid.metadata().size))
        .map(|i| {
            mem_grid.virtual_grid_pos_for_grid_pos(
                TlcPos(pos_for_index(i, mem_grid.metadata().size).map(|a| a as u32)),
                grid_size,
            )
        })
        .collect();

    dbg!(Instant::now() - start);

    let start_tlc = mem_grid.metadata().start_tlc;
    let metadata = &mem_grid.metadata;

    dbg!(Instant::now() - start);

    for (chunk_data, vgrid_pos) in mem_grid.chunks.iter_mut().zip(vgrid_positions) {
        vgrid[index_for_pos(vgrid_pos.0, grid_size)] = Some(edit_chunk_f(chunk_data, metadata));
    }

    dbg!(Instant::now() - start);

    MemoryGridEditor {
        chunks: vgrid,
        size: grid_size,
        start_tlc,
        metadata,
    }
}

impl<'a, MD: 'static, C: 'static>
    MemoryGridEditorChunk<'a, MemoryGridLayer<C, MD>, &'a MemoryGridLayerMetadata<MD>>
    for Option<&'a mut LayerChunk<C>>
{
    fn edit_grid_with_size(
        mem_grid: &'a mut MemoryGridLayer<C, MD>,
        grid_size: usize,
    ) -> MemoryGridEditor<Self, &'a MemoryGridLayerMetadata<MD>> {
        edit_grid_layer_with_size(mem_grid, grid_size, |lc, _| lc)
    }
}

impl<
        'a,
        MD: 'static,
        C: 'static,
        CE: ChunkEditor<&'a mut LayerChunk<C>, &'a MemoryGridLayerMetadata<MD>>,
    > MemoryGridEditorChunk<'a, MemoryGridLayer<C, MD>, &'a MemoryGridLayerMetadata<MD>>
    for Option<CE>
{
    fn edit_grid_with_size(
        mem_grid: &'a mut MemoryGridLayer<C, MD>,
        grid_size: usize,
    ) -> MemoryGridEditor<Option<CE>, &'a MemoryGridLayerMetadata<MD>> {
        edit_grid_layer_with_size(mem_grid, grid_size, CE::edit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_grid_layer_for_size() {
        let size = 4;
        let start_tlc = TlcPos(Point3 { x: 0, y: 0, z: 0 });
        let initial_chunks = (0..cubed(size)).map(|_| LayerChunk::new_valid(0)).collect();
        let mut layer = MemoryGridLayer::new(initial_chunks, start_tlc, size, ());
        {
            let mut editor = edit_grid_layer_with_size(&mut layer, 16, |c, _| c);
            for (i, chunk) in editor.chunks.iter_mut().enumerate() {
                chunk.as_mut().map(|c| {
                    c.get_mut().map(|c| {
                        *c = i;
                    })
                });
            }
        }
        for y in 0..4 {
            for z in 0..4 {
                for x in 0..4 {
                    assert_eq!(
                        *layer.chunks[x + y * 16 + z * 4].get().unwrap(),
                        x + 6 + (y + 6) * 256 + (z + 6) * 16
                    );
                }
            }
        }

        {
            let mut editor = edit_grid_layer_with_size(&mut layer, 16, |c, _| c);
            *editor.chunks[1640].as_mut().unwrap().get_mut().unwrap() = 99999;
        }
        assert!(*layer.chunks[2].get().unwrap() == 99999)
    }

    #[test]
    fn test_edit_grid_layer() {
        let size = 16;
        let start_tlc = TlcPos(Point3 { x: 0, y: 0, z: 0 });
        let initial_chunks = (0..cubed(size)).map(|_| LayerChunk::new_valid(0)).collect();
        let mut layer = MemoryGridLayer::new(initial_chunks, start_tlc, size, ());
        {
            let mut editor = edit_grid_layer_with_size(&mut layer, size, |c, _| c);
            for (i, chunk) in editor.chunks.iter_mut().enumerate() {
                chunk.as_mut().map(|c| {
                    c.get_mut().map(|c| {
                        *c = i;
                    })
                });
            }
        }
        for i in 0..cubed(size) {
            assert_eq!(*layer.chunks[i].get().unwrap(), i);
        }
    }
}
