use std::fmt::{Display};
use std::ops::{Index, IndexMut};
use num_traits::PrimInt;
use crate::renderer::component::voxel::data::{VoxelBitmask, VoxelTypeIDs};


#[derive(Clone)]
pub struct ChunkVoxelIDs<'a>(&'a mut [VoxelTypeIDs]);

impl From<&mut [VoxelTypeIDs]> for ChunkVoxelIDs {
    fn from(value: &mut [VoxelTypeIDs]) -> Self {
        ChunkVoxelIDs(value)
    }
}
impl Index<usize> for ChunkVoxelIDs {
    type Output = u8;
    fn index(&self, i: usize) -> &u8 {
        &self.0[i*(8/128)].indices[i%(8/128)]
    }
}
impl IndexMut<usize> for ChunkVoxelIDs {
    fn index_mut(&mut self, i: usize) -> &mut u8 {
        &mut self.0[i*(8/128)].indices[i%(8/128)]
    }
}


#[derive(Clone)]
pub struct ChunkBitmask<'a>(&'a mut [VoxelBitmask]);

impl From<&mut [VoxelBitmask]> for ChunkBitmask {
    fn from(value: &mut [VoxelBitmask]) -> Self {
        ChunkBitmask(value)
    }
}
impl ChunkBitmask {
    pub fn set_block_true(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index/128].mask |= bit;
    }

    pub fn set_block_false(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index/128].mask &= !bit;
    }
}