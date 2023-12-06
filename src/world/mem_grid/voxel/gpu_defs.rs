use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct ChunkVoxelIDs(Vec<VoxelTypeIDs>);

impl From<Vec<VoxelTypeIDs>> for ChunkVoxelIDs {
    fn from(value: Vec<VoxelTypeIDs>) -> Self {
        ChunkVoxelIDs(value)
    }
}
impl From<ChunkVoxelIDs> for Vec<VoxelTypeIDs> {
    fn from(value: ChunkVoxelIDs) -> Self {
        value.0
    }
}
impl Index<usize> for ChunkVoxelIDs {
    type Output = u8;
    fn index(&self, i: usize) -> &u8 {
        &self.0[i * 8 / 128].indices[i % 8 / 128]
    }
}
impl IndexMut<usize> for ChunkVoxelIDs {
    fn index_mut(&mut self, i: usize) -> &mut u8 {
        &mut self.0[i * 8 / 128].indices[i % 8 / 128]
    }
}

#[derive(Clone)]
pub struct ChunkBitmask(Vec<VoxelBitmask>);

impl From<Vec<VoxelBitmask>> for ChunkBitmask {
    fn from(value: Vec<VoxelBitmask>) -> Self {
        ChunkBitmask(value)
    }
}
impl From<ChunkBitmask> for Vec<VoxelBitmask> {
    fn from(value: ChunkBitmask) -> Self {
        value.0
    }
}
impl ChunkBitmask {
    pub fn set_block_true(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index / 128].mask |= bit;
    }

    pub fn set_block_false(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index / 128].mask &= !bit;
    }
}
