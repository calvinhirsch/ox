use crate::world::mem_grid::utils::{amod, cubed, pos_for_index, pos_index};
use crate::world::mem_grid::{FromVirtual, MemoryGridMetadata, PhysicalMemoryGrid, PhysicalMemoryGridStruct, ToVirtual, VirtualMemoryGridStruct};
use crate::world::{TLCPos, TLCVector};
use cgmath::{Array, EuclideanSpace, Vector3};
use derive_more::Deref;
use derive_new::new;


#[derive(Clone)]
pub struct MemoryGridLayerMetadata<E> {
    size: usize, // Size of grid in number of TLCs, 1 more than the rendered area size in each dimension for pre-loading
    tlc_size: usize,
    offsets: Vector3<usize>,
    chunks_loaded: Vec<bool>,
    extra: E,
}

#[derive(Deref)]
pub struct PhysicalMemoryGridLayer<C, MD>(
    PhysicalMemoryGridStruct<MemoryGridLayerData<C>, MemoryGridLayerMetadata<MD>>
);
#[derive(Deref)]
pub struct VirtualMemoryGridLayer<C, MD>(
    VirtualMemoryGridStruct<MemoryGridLayerChunkData<C>, MemoryGridLayerMetadata<MD>>
);

pub struct MemoryGridLayerData<C> {
    memory: Vec<C>
}

#[derive(new)]
pub struct MemoryGridLayerChunkData<C> {
    data: C,
    loaded: bool,  // TODO: remove? redundant with metadata
}

pub struct LayerChunkLoadingQueue {
    pub chunks: Vec<TLCPos<i64>>
}

impl<C, MD> PhysicalMemoryGridLayer<C, MD> {
    pub fn borrow_mem_mut(&mut self) -> &mut Vec<C> { &mut self.data }
    pub fn borrow_mem(&self) -> &Vec<C> { &self.data }
}

impl<C, MD> PhysicalMemoryGrid<MemoryGridLayerData<C>, MemoryGridLayerMetadata<MD>> for PhysicalMemoryGridLayer<C, MD> {
    type ChunkLoadQueue = LayerChunkLoadingQueue;

    fn shift(&mut self, shift: TLCVector<i32>, load: TLCVector<i32>) -> Self::ChunkLoadQueue {
        self.metadata.offsets = amod(self.metadata.offsets + shift, self.size());
        // ... TODO

        _
    }
}

impl<E: Sized> MemoryGridLayerMetadata<E> {
    pub fn new(start_tlc: TLCPos<i64>, size: usize, tlc_size: usize, extra: E) -> Self {
        MemoryGridLayerMetadata {
            size,
            tlc_size,
            offsets: Self::calc_offsets(start_tlc, size),
            chunks_loaded: vec![false; cubed(size)],
            extra,
        }
    }
}

impl<PC, VC, MD> ToVirtual<MemoryGridLayerChunkData<VC>, MemoryGridLayerMetadata<MD>> for PhysicalMemoryGridLayer<PC, MD> {
    fn to_virtual_for_size(self, grid_size: usize) -> VirtualMemoryGridLayer<VC, MD> {
        let vgrid_size = grid_size - 1;
        let mut vgrid = vec![None; (vgrid_size).pow(3)];
        let mut data = self.0.data;
        let metadata = self.0.metadata;

        for ((chunk_i, loaded), chunk_data) in metadata.chunks_loaded
            .iter().enumerate().zip(data.into_iter()) {
            // If this layer is smaller than full grid, add padding to virtual position so it
            // is centered
            let virtual_pos = metadata
                .virtual_grid_pos_for_grid_pos(
                    TLCVector(pos_for_index(chunk_i, metadata.size)),
                    vgrid_size,
                ).0;

            vgrid[pos_index(virtual_pos, vgrid_size)] = Some(
                MemoryGridLayerChunkData::<VC>::new(
                    chunk_data.into(),
                    *loaded
                )
            );
        }

        VirtualMemoryGridLayer(VirtualMemoryGridStruct { data: vgrid, metadata })
    }
}

impl <PC, VC, MD> FromVirtual<MemoryGridLayerChunkData<VC>, MemoryGridLayerMetadata<MD>> for PhysicalMemoryGridLayer<PC, MD> {
    fn from_virtual_for_size(
        virtual_grid: VirtualMemoryGridStruct<MemoryGridLayerChunkData<VC>, MemoryGridLayerMetadata<MD>>,
        vgrid_size: usize
    ) -> Self {
        let phys_grid_size = vgrid_size + 1;
        let mut grid = vec![None; (vgrid_size).pow(3)];
        let mut data = virtual_grid.data;
        let metadata = virtual_grid.metadata;

        for (chunk_i, chunk_data) in data.into_iter().enumerate() {
            match chunk_data {
                None => {},
                Some(data) => {
                    let phys_pos = metadata.grid_pos_for_virtual_grid_pos(
                        TLCVector(pos_for_index(chunk_i, vgrid_size)),
                        vgrid_size,
                    ).0;

                    grid[pos_index(phys_pos, phys_grid_size)] = data.data;
                }
            }
        }

        Self(
            PhysicalMemoryGridStruct::new(
                MemoryGridLayerData {
                    memory: grid,
                },
                metadata,
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
        let local_vgrid_pos = tlc_pos
            - Vector3::from_value(if vgrid_size > self.size {
                (vgrid_size - (self.size - 1)) / 2
            } else {
                0
            });
        TLCVector((local_vgrid_pos.0 + self.offsets) % self.size)
    }

    pub fn virtual_grid_pos_for_grid_pos(
        &self,
        pos: TLCVector<usize>,
        vgrid_size: usize,
    ) -> TLCVector<usize> {
        let local_vgrid_pos = amod(
            pos.0.cast::<i64>().unwrap() - self.offsets.cast::<i64>().unwrap(),
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
