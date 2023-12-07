use crate::world::mem_grid::utils::{amod, pos_for_index, index_for_pos};
use crate::world::mem_grid::{FromVirtual, MemoryGridMetadata, PhysicalMemoryGrid, PhysicalMemoryGridStruct, Placeholder, ToVirtual, VirtualMemoryGridStruct};
use crate::world::{TLCPos, TLCVector};
use cgmath::{Array, EuclideanSpace, Vector3, Point3};
use derive_more::Deref;
use derive_new::new;
use hashbrown::HashSet;
use crate::world::loader::ChunkLoadQueueItem;


#[derive(Clone)]
pub struct MemoryGridLayerMetadata<E> {
    size: usize, // Size of grid in number of TLCs, 1 more than the rendered area size in each dimension for pre-loading
    tlc_size: usize,
    start_tlc: TLCPos<i64>,
    offsets: TLCVector<usize>,
    pub extra: E,
}
impl<E> MemoryGridMetadata for MemoryGridLayerMetadata<E> {
    fn size(&self) -> usize { self.size }
}

#[derive(Deref, new)]
pub struct PhysicalMemoryGridLayer<C: Clone, MD>(
    PhysicalMemoryGridStruct<MemoryGridLayerData<C>, MemoryGridLayerMetadata<MD>>
);
pub type VirtualMemoryGridLayer<C, MD> =
    VirtualMemoryGridStruct<MemoryGridLayerChunkData<C>, MemoryGridLayerMetadata<MD>>;

#[derive(Clone, new)]
pub struct MemoryGridLayerData<C: Clone> {
    chunks: Vec<C>,
    chunks_loaded: Vec<bool>,
}

#[derive(new, Clone)]  // Clone is not really necessary, just easier for initializing vecs with None
pub struct MemoryGridLayerChunkData<C: Clone + Default> {
    pub data: C,
    loaded: bool,
}


impl<C: Clone, MD> PhysicalMemoryGridLayer<C, MD> {
    pub fn borrow_mem_mut(&mut self) -> &mut Vec<C> { &mut self.0.data.chunks }
    pub fn borrow_mem(&self) -> &Vec<C> { &self.data.chunks }
}

impl<C: Clone, MD> PhysicalMemoryGrid for PhysicalMemoryGridLayer<C, MD> {
    type Data = MemoryGridLayerData<C>;
    type Metadata = MemoryGridLayerMetadata<MD>;
    type ChunkLoadQueueItemData = ();


    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<()>> {
        let start_tlc = self.metadata.start_tlc.0;
        let size = self.metadata.size;
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

    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<()>> {
        self.0.metadata.offsets = TLCVector(
            amod(self.metadata.offsets.0.cast::<i64>().unwrap() + shift.0.cast::<i64>().unwrap(), self.size())
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
        let vgrid_size = self.metadata.size - 1;

        for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            if shift.0[a] != 0 {
                for av in
                    if shift.0[a] < 0 { 0..load_in_from_edge.0[a] }
                    else { vgrid_size as i32 - 1 - load_in_from_edge.0[a] .. vgrid_size as i32 }
                {
                    for bv in 0..vgrid_size {
                        for cv in 0..vgrid_size {
                            let chunk = abc_pos(av, bv as i32, cv as i32, a, b, c);
                            chunk_set.insert(ChunkLoadQueueItem {
                                pos: TLCPos(chunk.0 + self.metadata.start_tlc.0.to_vec()),
                                data: (),
                            });
                        }
                    }
                }
            }
        }

        let mut chunks: Vec<ChunkLoadQueueItem<()>> = chunk_set.into_iter().collect();

        for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
            if load_buffer[a] {
                let av = if shift.0[a] > 0 { self.metadata.tlc_size as i32 } else { -1 };
                for bv in 0..vgrid_size {
                    for cv in 0..vgrid_size {
                        let chunk = abc_pos(av, bv as i32, cv as i32, a, b, c);
                        chunks.push(ChunkLoadQueueItem {
                            pos: TLCPos(chunk.0 + self.metadata.start_tlc.0.to_vec()),
                            data: (),
                        });
                    }
                }
            }
            if load_buffer[a] && load_buffer[b] {
                let av = if shift.0[a] > 0 { self.metadata.tlc_size as i64 } else { -1 };
                let bv = if shift.0[b] > 0 { self.metadata.tlc_size as i64 } else { -1 };
                for cv in 0..vgrid_size as i64 {
                    let chunk = abc_pos(av, bv, cv, a, b, c);
                    chunks.push(ChunkLoadQueueItem {
                        pos: TLCPos(chunk.0 + self.metadata.start_tlc.0.to_vec()),
                        data: (),
                    });
                }
            }
        }

        if load_buffer.iter().all(|x| *x) {
            let chunk = Point3 {
                x: if shift.0.x > 0 { self.metadata.tlc_size as i64 } else { -1 },
                y: if shift.0.y > 0 { self.metadata.tlc_size as i64 } else { -1 },
                z: if shift.0.z > 0 { self.metadata.tlc_size as i64 } else { -1 },
            };
            chunks.push(ChunkLoadQueueItem {
                pos: TLCPos(chunk + self.metadata.start_tlc.0.to_vec()),
                data: (),
            });
        }

        chunks
    }
}

impl<E: Sized> MemoryGridLayerMetadata<E> {
    pub fn new(start_tlc: TLCPos<i64>, size: usize, tlc_size: usize, extra: E) -> Self {
        MemoryGridLayerMetadata {
            size,
            start_tlc,
            tlc_size,
            offsets: TLCVector(Self::calc_offsets(start_tlc, size)),
            extra,
        }
    }
}

impl<PC: Clone, VC: Default + Clone + From<PC>, MD> ToVirtual<MemoryGridLayerChunkData<VC>, MemoryGridLayerMetadata<MD>> for PhysicalMemoryGridLayer<PC, MD> {
    fn to_virtual_for_size(self, grid_size: usize) -> VirtualMemoryGridLayer<VC, MD> {
        let vgrid_size = grid_size - 1;
        let mut vgrid = vec![None; (vgrid_size).pow(3)];
        let data = self.0.data;
        let metadata = self.0.metadata;

        for (chunk_i, (chunk_data, loaded)) in data.chunks.into_iter().zip(data.chunks_loaded.iter()).enumerate() {
            // If this layer is smaller than full grid, add padding to virtual position so it
            // is centered
            let virtual_pos = metadata
                .virtual_grid_pos_for_grid_pos(
                    TLCVector(pos_for_index(chunk_i, metadata.size)),
                    vgrid_size,
                ).0;

            vgrid[index_for_pos(virtual_pos, vgrid_size)] = Some(
                MemoryGridLayerChunkData::<VC>::new(
                    chunk_data.into(),
                    *loaded,
                )
            );
        }

        VirtualMemoryGridLayer {
            chunks: vgrid,
            metadata: MemoryGridLayerMetadata { size: vgrid_size, ..metadata }
        }
    }
}

impl <PC: Clone, VC: Default + Into<PC> + Clone, MD> FromVirtual<MemoryGridLayerChunkData<VC>, MemoryGridLayerMetadata<MD>> for PhysicalMemoryGridLayer<PC, MD> {
    fn from_virtual_for_size(
        virtual_grid: VirtualMemoryGridStruct<MemoryGridLayerChunkData<VC>, MemoryGridLayerMetadata<MD>>,
        vgrid_size: usize
    ) -> Self {
        let phys_grid_size = vgrid_size + 1;
        let mut grid = vec![None; vgrid_size.pow(3)];
        let mut chunks_loaded = vec![false; vgrid_size.pow(3)];
        let data = virtual_grid.chunks;
        let metadata = virtual_grid.metadata;

        for (chunk_i, chunk_data) in data.into_iter().enumerate() {
            match chunk_data {
                None => {},
                Some(data) => {
                    let phys_pos = metadata.grid_pos_for_virtual_grid_pos(
                        TLCVector(pos_for_index(chunk_i, vgrid_size)),
                        vgrid_size,
                    ).0;

                    let idx = index_for_pos(phys_pos, phys_grid_size);
                    chunks_loaded[idx] = data.loaded;
                    grid[idx] = Some(data.data.into());
                }
            }
        }

        Self(
            PhysicalMemoryGridStruct::new(
                MemoryGridLayerData {
                    chunks: grid.into_iter().map(|x| x.unwrap()).collect(),
                    chunks_loaded,
                },
                MemoryGridLayerMetadata { size: phys_grid_size, ..metadata },
            )
        )
    }
}


impl<E: Sized> MemoryGridLayerMetadata<E> {
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


impl<C: Default + Clone> Placeholder for MemoryGridLayerChunkData<C> {
    fn placeholder(&self) -> Self {
        Self {
            data: C::default(),
            loaded: false,
        }
    }
}