use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
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
impl Default for ChunkVoxelIDs {
    fn default() -> Self {
        ChunkVoxelIDs(vec![])
    }
}
impl ChunkVoxelIDs {
    pub fn n_voxels(&self) -> usize { self.0.len() * 128 / VoxelTypeIDs::BITS_PER_VOXEL }
}

#[derive(Clone, Debug)]
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
impl Default for ChunkBitmask {
    fn default() -> Self {
        ChunkBitmask(vec![])
    }
}
impl ChunkBitmask {
    pub fn n_voxels(&self) -> usize { self.0.len() * 128 / VoxelBitmask::BITS_PER_VOXEL }

    pub fn get(&self, index: usize) -> bool {
        let bit = 1u128.to_le() << (index % 128);
        (self.0[index / 128].mask & bit) > 0
    }

    pub fn set_block(&mut self, index: usize, val: bool) {
        if val {
            self.set_block_true(index);
        }
        else {
            self.set_block_false(index);
        }
    }

    pub fn set_block_true(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index / 128].mask |= bit;
    }

    pub fn set_block_false(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index / 128].mask &= !bit;
    }
}
