use cgmath::Vector3;

pub mod layer;
mod layer_set;
mod utils;
pub mod voxel;

pub trait PhysicalMemoryGrid {
    fn shift_offsets(&mut self, shift: Vector3<i64>);

    fn size(&self) -> usize;
}

pub struct VirtualMemoryGrid<C> {
    pub chunks: Vec<C>,
}

pub trait AsVirtualMemoryGrid<C>: PhysicalMemoryGrid {
    fn as_virtual(&mut self) -> VirtualMemoryGrid<C> {
        self.as_virtual_for_size(self.size())
    }

    fn as_virtual_for_size(&mut self, grid_size: usize) -> VirtualMemoryGrid<C>;
}
