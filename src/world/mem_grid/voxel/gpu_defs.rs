use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use num_traits::PrimInt;
use std::fmt::Display;
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct ChunkVoxelIDs<'a>(&'a mut [VoxelTypeIDs]);

impl<'a> From<&mut [VoxelTypeIDs]> for ChunkVoxelIDs<'a> {
    fn from(value: &mut [VoxelTypeIDs]) -> Self {
        ChunkVoxelIDs(value)
    }
}
impl<'a> Index<usize> for ChunkVoxelIDs<'a> {
    type Output = u8;
    fn index(&self, i: usize) -> &u8 {
        &self.0[i * (8 / 128)].indices[i % (8 / 128)]
    }
}
impl<'a> IndexMut<usize> for ChunkVoxelIDs<'a> {
    fn index_mut(&mut self, i: usize) -> &mut u8 {
        &mut self.0[i * (8 / 128)].indices[i % (8 / 128)]
    }
}

#[derive(Clone)]
pub struct ChunkBitmask<'a>(&'a mut [VoxelBitmask]);

impl<'a> From<&mut [VoxelBitmask]> for ChunkBitmask<'a> {
    fn from(value: &mut [VoxelBitmask]) -> Self {
        ChunkBitmask(value)
    }
}
impl<'a> ChunkBitmask<'a> {
    pub fn set_block_true(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index / 128].mask |= bit;
    }

    pub fn set_block_false(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index / 128].mask &= !bit;
    }
}
