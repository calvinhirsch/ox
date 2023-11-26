use cgmath::{Array, EuclideanSpace, Vector3};
use num_traits::Zero;
use vulkano::image::ImageAspect::MemoryPlane0;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::camera::Camera;
use crate::world::mem_grid::{PhysicalMemoryGrid, VirtualMemoryGrid};
use crate::world::mem_grid::utils::{amod, cubed, pos_for_index, pos_index};
use crate::world::{TLCPos, TLCVector, VoxelPos};
use crate::world::mem_grid::rendering::ChunkVoxelIDs;


#[derive(Clone)]
pub struct MemoryGridLayerMetadata<E: Sized> {
    size: usize, // Size of grid in number of TLCs, 1 more than the rendered area size in each dimension for pre-loading
    offsets: Vector3<usize>,
    loaded_upper_chunks: bool,
    extra: E,
}

pub struct MemoryGridLayer<D: Sized, MD: Sized> {
    metadata: MemoryGridLayerMetadata<MD>,
    memory: Vec<D>,
}

// TODO: might be able to remove VirtualMemoryGrid trait and just use this for everything
pub struct VirtualMemoryGridForLayer<'a, D: Sized, C: From<&mut [D]>, MD: Sized> {
    metadata: &'a MemoryGridLayerMetadata<MD>,
    chunks: Vec<Option<C>>,
}


impl<E: Sized> MemoryGridLayerMetadata<E> {
    pub fn new(start_tlc: TLCPos<i64>, size: usize, extra: E) -> Self {
        MemoryGridLayerMetadata {
            size,
            offsets: Self::calc_offsets(start_tlc, size),
            loaded_upper_chunks: false,
            extra,
        }
    }
}


impl<D: Sized, MD: Sized> MemoryGridLayer<D, MD> {
    pub fn new_raw(metadata: MemoryGridLayerMetadata<MD>, memory: Vec<D>) -> Self {
        MemoryGridLayer { metadata, memory }
    }

    pub fn borrow_mem_mut(&mut self) -> &mut Vec<D> { &mut self.memory }
}
impl<'a, D: Sized, MD: Sized,C: From<&'a mut [D]>, VE: VoxelTypeEnum> PhysicalMemoryGrid<VE, VirtualMemoryGridForLayer<'a, D, C, MD>> for MemoryGridLayer<D, MD> {
    fn shift_offsets(&mut self, shift: Vector3<i64>) {
        if !shift.is_zero() {
            todo!();
        }
    }

    fn size(&self) -> usize {
        self.metadata.size
    }

    fn as_virtual(&mut self) -> Self::Virtual {
        self.as_virtual_for_size(self.metadata.size-1)
    }

    fn as_virtual_for_size(&mut self, grid_size: usize) -> VirtualMemoryGridForLayer<'a, D, C, MD> {
        let mut grid = vec![None; (grid_size).pow(3)];

        let n_per_tlc = self.memory.len() / cubed(self.metadata.size);
        let mut slice = self.memory.as_slice();
        for chunk_i in 0..cubed(self.metadata.size) {
            // If this layer is smaller than full grid, add padding to virtual position so it
            // is centered
            let virtual_pos = self.metadata.virtual_grid_pos_for_index(chunk_i, grid_size).0;

            let (chunk, rest) = self.memory.split_at_mut(n_per_tlc);
            slice = rest;
            grid[pos_index(virtual_pos, grid_size-1)] = C::from(chunk);
        }

        VirtualMemoryGridForLayer {
            metadata: &self.metadata,
            chunks: grid,
        }
    }
}

impl<'a, D: Sized, C: From<&mut [D]>, MD: Sized> VirtualMemoryGridForLayer<'a, D, C, MD> {
    pub fn deconstruct(self) -> (&'a MemoryGridLayerMetadata<MD>, Vec<Option<C>>) {
        (self.metadata, self.chunks)
    }
}
impl<D: Sized, C: From<&mut [D]>, MD: Sized, VE: VoxelTypeEnum> VirtualMemoryGrid<VE> for VirtualMemoryGridForLayer<D, C, MD> {
    fn load_or_generate_tlc(&self, voxel_output: &mut ChunkVoxelIDs, tlc: TLCPos<i64>) {
        todo!()
    }

    fn reload_all(&mut self) {
        todo!()
    }
}


impl<E: Sized> MemoryGridLayerMetadata<E> {
    pub fn calc_offsets(start_tlc: TLCPos<i64>, size: usize) -> Vector3<usize> {
        amod(start_tlc.0.to_vec(), size)
    }

    pub fn grid_pos_for_virtual_grid_pos(&self, tlc_pos: TLCVector<usize>, vgrid_size: usize) -> TLCVector<usize> {
        let local_vgrid_pos = tlc_pos - Vector3::from_value(
            if vgrid_size > self.size { (vgrid_size - (self.size - 1)) / 2 }
            else { 0 }
        );
        TLCVector((local_vgrid_pos.0 + self.offsets) % self.size)
    }

    pub fn index_for_virtual_grid_pos(&self, pos: TLCVector<usize>, vgrid_size: usize) -> usize {
        pos_index(self.grid_pos_for_virtual_grid_pos(pos, vgrid_size).0, self.size)
    }

    pub fn virtual_grid_pos_for_grid_pos(&self, pos: TLCVector<usize>, vgrid_size: usize) -> TLCVector<usize> {
        let local_vgrid_pos = amod(pos.0.cast::<i64>().unwrap() - self.offsets.cast::<i64>().unwrap(), self.size);
        TLCVector(
            local_vgrid_pos + Vector3::<usize>::from_value(
                if self.size < vgrid_size {
                    (vgrid_size - self.size) / 2
                }
                else { 0 }
            )
        )
    }

    pub fn virtual_grid_pos_for_index(&self, index: usize, vgrid_size: usize) -> TLCVector<usize> {
        self.virtual_grid_pos_for_grid_pos(TLCVector(pos_for_index(index, self.size)), vgrid_size)
    }
}