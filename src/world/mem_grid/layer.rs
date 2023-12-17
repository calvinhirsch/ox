use crate::world::mem_grid::utils::{amod, pos_for_index, index_for_pos};
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor, NewMemoryGridEditor};
use crate::world::{TLCPos, TLCVector};
use cgmath::{Array, EuclideanSpace, Vector3, Point3};
use getset::{Getters, MutGetters};
use hashbrown::HashSet;
use crate::world::loader::{ChunkLoadQueueItem, ChunkLoadQueueItemData};


#[derive(Clone, Debug, Getters, MutGetters)]
pub struct MemoryGridLayer<C> {
    #[getset(get = "pub", get_mut = "pub")]
    chunks: Vec<C>,
    start_tlc: TLCPos<i64>,
    size: usize,
    tlc_size: usize,
    offsets: TLCVector<usize>,
}


impl<C> MemoryGridLayer<C> {
    pub fn new(
        chunks: Vec<C>,
        start_tlc: TLCPos<i64>,
        size: usize,
        tlc_size: usize,
    ) -> Self {
        MemoryGridLayer {
            chunks,
            start_tlc,
            size,
            tlc_size,
            offsets: TLCVector(Self::calc_offsets(start_tlc, size)),
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
            - Vector3::from_value(if vgrid_size > self.size {
            (vgrid_size - (self.size - 1)) / 2
        } else {
            0
        });
        TLCVector((local_vgrid_pos + self.offsets.0) % self.size)
    }

    pub fn virtual_grid_pos_for_grid_pos(
        &self,
        pos: TLCVector<usize>,
        vgrid_size: usize,
    ) -> TLCVector<usize> {
        let local_vgrid_pos = amod(
            pos.0.cast::<i64>().unwrap() - self.offsets.0.cast::<i64>().unwrap(),
            self.size,
        );
        TLCVector(
            local_vgrid_pos
                + Vector3::<usize>::from_value(if self.size < vgrid_size {
                (vgrid_size - self.size) / 2
            } else {
                0
            }),
        )
    }
}

impl ChunkLoadQueueItemData for () {}

impl<C> MemoryGrid for MemoryGridLayer<C> {
    type ChunkLoadQueueItemData = ();

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        // for chunk in self.chunks.iter_mut() {
        //     chunk.invalidate();
        // }

        let start_tlc = self.start_tlc.0;
        let size = self.size;

        (0..size as i64).flat_map(|x|
            (0..size as i64).flat_map(move |y|
                (0..size as i64).map(move |z|
                    ChunkLoadQueueItem {
                        pos: TLCPos(start_tlc + Vector3 { x, y, z }),
                        data: (),
                    }
                )
            )
        ).collect()
    }

    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        self.offsets = TLCVector(
            amod(self.offsets.0.cast::<i64>().unwrap() + shift.0.cast::<i64>().unwrap(), self.size())
        );

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
        let vgrid_size = self.size - 1;
        let tlc_size = self.tlc_size;

        let mut queue_chunk = |vgrid_chunk: TLCPos<i64>| {
            chunk_set.insert(ChunkLoadQueueItem {
                pos: TLCPos(vgrid_chunk.0 + self.start_tlc.0.to_vec()),
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
                for av in
                if shift.0[a] < 0 { 0..load_in_from_edge.0[a] }
                else { vgrid_size as i32 - 1 - load_in_from_edge.0[a] .. vgrid_size as i32 }
                {
                    for bv in 0..vgrid_size {
                        for cv in 0..vgrid_size {
                            queue_chunk(abc_pos(av, bv as i32, cv as i32, a, b, c));
                        }
                    }
                }
            }
        }

        for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            if load_buffer[a] {
                let av = if shift.0[a] > 0 { tlc_size as i32 } else { -1 };
                for bv in 0..vgrid_size {
                    for cv in 0..vgrid_size {
                        queue_chunk(abc_pos(av, bv as i32, cv as i32, a, b, c));
                    }
                }
            }
            if load_buffer[a] && load_buffer[b] {
                let av = if shift.0[a] > 0 { tlc_size as i64 } else { -1 };
                let bv = if shift.0[b] > 0 { tlc_size as i64 } else { -1 };
                for cv in 0..vgrid_size as i64 {
                    queue_chunk(abc_pos(av, bv, cv, a, b, c));
                }
            }
        }

        if load_buffer.iter().all(|x| *x) {
            let chunk = Point3 {
                x: if shift.0.x > 0 { tlc_size as i64 } else { -1 },
                y: if shift.0.y > 0 { tlc_size as i64 } else { -1 },
                z: if shift.0.z > 0 { tlc_size as i64 } else { -1 },
            };
            queue_chunk(TLCPos(chunk));
        }

        chunk_set.into_iter().collect()
    }

    fn size(&self) -> usize { self.size }
    fn start_tlc(&self) -> TLCPos<i64> { self.start_tlc }
}


// impl<'a, C: 'static, CE: From<&'a mut C>> EditMemoryGrid<CE, ()> for MemoryGridLayer<C> {
//     fn edit_for_size(&mut self, grid_size: usize) -> MemoryGridEditor<CE, ()> {
//
//         let vgrid_size = grid_size - 1;
//         let mut vgrid: Vec<Option<CE>> = (0..vgrid_size.pow(3)).map(|_| None).collect();
//
//         // If this layer is smaller than full grid, add padding to virtual position so it
//         // is centered
//         let virtual_positions: Vec<_> = (0..grid_size).map(|i|
//            self.virtual_grid_pos_for_grid_pos(
//                TLCVector(pos_for_index(i, self.size)),
//                vgrid_size,
//            )
//         ).collect();
//
//         for (chunk_data, virtual_pos) in self.chunks.iter_mut()
//             .zip(virtual_positions) {
//             vgrid[index_for_pos(virtual_pos.0, vgrid_size)] = Some(CE::from(chunk_data));
//         }
//
//         MemoryGridEditor {
//             // lifetime: PhantomData,
//             chunks: vgrid,
//             size: self.size,
//             start_tlc: self.start_tlc,
//             metadata: (),
//         }
//     }
// }


// #[derive(Deref, DerefMut)]
// pub struct MemoryGridLayerEditor<CE>(MemoryGridEditorData<CE, ()>);

impl<'a, C: 'static, CE: From<&'a mut C>> NewMemoryGridEditor<'a, MemoryGridLayer<C>> for MemoryGridEditor<CE, ()> {
    fn for_grid_with_size(mem_grid: &'a mut MemoryGridLayer<C>, grid_size: usize) -> Self {
        let vgrid_size = grid_size - 1;
        let mut vgrid: Vec<Option<CE>> = (0..vgrid_size.pow(3)).map(|_| None).collect();

        // If this layer is smaller than full grid, add padding to virtual position so it
        // is centered
        let virtual_positions: Vec<_> = (0..grid_size).map(|i|
            mem_grid.virtual_grid_pos_for_grid_pos(
                TLCVector(pos_for_index(i, mem_grid.size)),
                vgrid_size,
            )
        ).collect();

        for (chunk_data, virtual_pos) in mem_grid.chunks.iter_mut()
            .zip(virtual_positions) {
            vgrid[index_for_pos(virtual_pos.0, vgrid_size)] = Some(CE::from(chunk_data));
        }

        MemoryGridEditor {
            // lifetime: PhantomData,
            chunks: vgrid,
            size: mem_grid.size,
            start_tlc: mem_grid.start_tlc,
            metadata: (),
        }
    }
}