use crate::world::loader::{ChunkLoadQueueItem, LayerChunk};
use crate::world::mem_grid::utils::{amod, cubed, index_for_pos, pos_for_index};
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor, MemoryGridEditorChunk};
use crate::world::{TLCPos, TLCVector};
use cgmath::{Array, EuclideanSpace, Point3, Vector3};
use getset::{Getters, MutGetters};
use hashbrown::HashSet;

use super::ChunkEditor;

#[derive(Clone, Debug, Getters)]
pub struct MemoryGridLayerMetadata<MD> {
    start_tlc: TLCPos<i64>,
    size: usize,
    offsets: TLCVector<usize>,
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
        start_tlc: TLCPos<i64>,
        size: usize,
        extra_metadata: MD,
    ) -> Self {
        MemoryGridLayer {
            chunks,
            metadata: MemoryGridLayerMetadata {
                start_tlc,
                size,
                offsets: TLCVector(Self::calc_offsets(start_tlc, size)),
                extra: extra_metadata,
            },
        }
    }

    pub fn calc_offsets(start_tlc: TLCPos<i64>, size: usize) -> Vector3<usize> {
        amod(start_tlc.0.to_vec(), size)
    }
    pub fn grid_pos_for_virtual_grid_pos(
        &self,
        tlc_pos: TLCVector<usize>,
        vgrid_size: usize,
    ) -> TLCVector<usize> {
        let local_vgrid_pos = tlc_pos.0
            - Vector3::from_value(if vgrid_size > self.metadata.size {
                (vgrid_size - (self.metadata.size - 1)) / 2
            } else {
                0
            });
        TLCVector((local_vgrid_pos + self.metadata.offsets.0) % self.metadata.size)
    }

    pub fn virtual_grid_pos_for_grid_pos(
        &self,
        pos: TLCVector<usize>,
        vgrid_size: usize,
    ) -> TLCVector<usize> {
        let local_vgrid_pos = amod(
            pos.0.cast::<i64>().unwrap() - self.metadata.offsets.0.cast::<i64>().unwrap(),
            self.metadata.size,
        );
        TLCVector(
            local_vgrid_pos
                + Vector3::<usize>::from_value(if self.metadata.size < vgrid_size {
                    (vgrid_size - self.metadata.size) / 2
                } else {
                    0
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
                        pos: TLCPos(start_tlc + Vector3 { x, y, z }),
                        data: (),
                    })
                })
            })
            .collect()
    }

    fn shift(
        &mut self,
        shift: TLCVector<i32>,
        load_in_from_edge: TLCVector<i32>,
        load_buffer: [bool; 3],
    ) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        self.metadata.offsets = TLCVector(amod(
            self.metadata().offsets.0.cast::<i64>().unwrap() + shift.0.cast::<i64>().unwrap(),
            self.size(),
        ));

        // ENHANCEMENT: Using a hash set here to remove duplicates, could probably do this smarter
        //              and avoid duplicates logically.

        fn abc_pos<T: Into<i64>>(av: T, bv: T, cv: T, a: usize, b: usize, c: usize) -> TLCPos<i64> {
            let mut chunk = Point3::<i64> { x: 0, y: 0, z: 0 };
            chunk[a] = av.into();
            chunk[b] = bv.into();
            chunk[c] = cv.into();
            TLCPos(chunk)
        }

        let mut chunk_set = HashSet::new();
        let active_size = self.metadata().size - 1; // size of grid excluding buffer chunks

        let mut queue_chunk = |vgrid_chunk: TLCPos<i64>| {
            chunk_set.insert(ChunkLoadQueueItem {
                pos: TLCPos(vgrid_chunk.0 + self.metadata().start_tlc.0.to_vec()),
                data: (),
            });

            // let grid_chunk = self.grid_pos_for_virtual_grid_pos(
            //     TLCVector(vgrid_chunk.0.to_vec().cast::<usize>().unwrap()),
            //     vgrid_size
            // );
            // self.chunks[index_for_pos(grid_chunk.0, self.size)].invalidate();
        };

        for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            if shift.0[a] != 0 {
                for av in if shift.0[a] < 0 {
                    0..load_in_from_edge.0[a]
                } else {
                    active_size as i32 - 1 - load_in_from_edge.0[a]..active_size as i32
                } {
                    for bv in 0..active_size {
                        for cv in 0..active_size {
                            queue_chunk(abc_pos(av, bv as i32, cv as i32, a, b, c));
                        }
                    }
                }
            }
        }

        for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            if load_buffer[a] {
                let av = if shift.0[a] > 0 {
                    active_size as i32
                } else {
                    -1
                };
                for bv in 0..active_size {
                    for cv in 0..active_size {
                        queue_chunk(abc_pos(av, bv as i32, cv as i32, a, b, c));
                    }
                }
            }
            if load_buffer[a] && load_buffer[b] {
                let av = if shift.0[a] > 0 {
                    active_size as i64
                } else {
                    -1
                };
                let bv = if shift.0[b] > 0 {
                    active_size as i64
                } else {
                    -1
                };
                for cv in 0..active_size as i64 {
                    queue_chunk(abc_pos(av, bv, cv, a, b, c));
                }
            }
        }

        if load_buffer.iter().all(|x| *x) {
            let chunk = Point3 {
                x: if shift.0.x > 0 {
                    active_size as i64
                } else {
                    -1
                },
                y: if shift.0.y > 0 {
                    active_size as i64
                } else {
                    -1
                },
                z: if shift.0.z > 0 {
                    active_size as i64
                } else {
                    -1
                },
            };
            queue_chunk(TLCPos(chunk));
        }

        chunk_set.into_iter().collect()
    }

    fn size(&self) -> usize {
        self.metadata().size
    }
    fn start_tlc(&self) -> TLCPos<i64> {
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
        let mut vgrid: Vec<Option<CE>> = (0..grid_size.pow(3)).map(|_| None).collect();

        // If this layer is smaller than full grid, add padding to virtual position so it
        // is centered
        let vgrid_positions: Vec<_> = (0..cubed(mem_grid.metadata().size))
            .map(|i| {
                mem_grid.virtual_grid_pos_for_grid_pos(
                    TLCVector(pos_for_index(i, mem_grid.metadata().size)),
                    grid_size,
                )
            })
            .collect();

        let start_tlc = mem_grid.metadata().start_tlc;
        let metadata = &mem_grid.metadata;

        for (chunk_data, vgrid_pos) in mem_grid.chunks.iter_mut().zip(vgrid_positions) {
            vgrid[index_for_pos(vgrid_pos.0, grid_size)] = Some(CE::edit(chunk_data, metadata));
        }

        MemoryGridEditor {
            chunks: vgrid,
            size: grid_size,
            start_tlc,
            metadata,
        }
    }
}
