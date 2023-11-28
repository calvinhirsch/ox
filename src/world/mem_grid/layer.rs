use crate::world::mem_grid::utils::{amod, cubed, pos_for_index, pos_index};
use crate::world::mem_grid::{AsVirtualMemoryGrid, PhysicalMemoryGrid, VirtualMemoryGrid};
use crate::world::{TLCPos, TLCVector};
use cgmath::{Array, EuclideanSpace, Vector3};
use derive_new::new;
use num_traits::Zero;

#[derive(Clone)]
pub struct MemoryGridLayerMetadata<E: Sized> {
    size: usize, // Size of grid in number of TLCs, 1 more than the rendered area size in each dimension for pre-loading
    offsets: Vector3<usize>,
    loaded_upper_chunks: bool,
    chunks_loaded: Vec<bool>,
    extra: E,
}

#[derive(new)]
pub struct MemoryGridLayer<D: Sized, MD: Sized> {
    metadata: MemoryGridLayerMetadata<MD>,
    memory: Vec<D>,
}

pub trait LayerChunkData<D: Sized> {
    fn new(slice: &mut [D], loaded: bool) -> Self;
}

impl<E: Sized> MemoryGridLayerMetadata<E> {
    pub fn new(start_tlc: TLCPos<i64>, size: usize, extra: E) -> Self {
        MemoryGridLayerMetadata {
            size,
            offsets: Self::calc_offsets(start_tlc, size),
            loaded_upper_chunks: false,
            chunks_loaded: vec![false; cubed(size)],
            extra,
        }
    }
}

impl<D: Sized, MD: Sized> MemoryGridLayer<D, MD> {
    pub fn borrow_mem_mut(&mut self) -> &mut Vec<D> {
        &mut self.memory
    }
}
impl<D: Sized, MD: Sized> PhysicalMemoryGrid for MemoryGridLayer<D, MD> {
    fn shift_offsets(&mut self, shift: Vector3<i64>) {
        if !shift.is_zero() {
            todo!();
        }
    }

    fn size(&self) -> usize {
        self.metadata.size
    }
}
impl<D, MD, C> AsVirtualMemoryGrid<VirtualMemoryGrid<C>> for MemoryGridLayer<D, MD>
where
    D: Sized,
    MD: Sized,
    C: LayerChunkData<D>,
{
    fn as_virtual(&mut self) -> Self::Virtual {
        self.as_virtual_for_size(self.metadata.size - 1)
    }

    fn as_virtual_for_size(&mut self, grid_size: usize) -> VirtualMemoryGrid<C> {
        let mut grid = vec![None; (grid_size).pow(3)];

        let n_per_tlc = self.memory.len() / cubed(self.metadata.size);
        let mut slice = self.memory.as_slice();
        for (chunk_i, loaded) in self.metadata.chunks_loaded.iter().enumerate() {
            // If this layer is smaller than full grid, add padding to virtual position so it
            // is centered
            let virtual_pos = self
                .metadata
                .virtual_grid_pos_for_index(chunk_i, grid_size)
                .0;

            let (chunk, rest) = self.memory.split_at_mut(n_per_tlc);
            slice = rest;
            grid[pos_index(virtual_pos, grid_size - 1)] = C::new(chunk, *loaded);
        }

        VirtualMemoryGrid { chunks: grid }
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

    pub fn index_for_virtual_grid_pos(&self, pos: TLCVector<usize>, vgrid_size: usize) -> usize {
        pos_index(
            self.grid_pos_for_virtual_grid_pos(pos, vgrid_size).0,
            self.size,
        )
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

    pub fn virtual_grid_pos_for_index(&self, index: usize, vgrid_size: usize) -> TLCVector<usize> {
        self.virtual_grid_pos_for_grid_pos(TLCVector(pos_for_index(index, self.size)), vgrid_size)
    }
}
