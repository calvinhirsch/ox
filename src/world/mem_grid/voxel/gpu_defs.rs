use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use std::ops::{Index, IndexMut};
use vulkano::command_buffer::BufferCopy;

#[derive(Clone, Debug)]
pub struct ChunkVoxels {
    pub ids: Vec<VoxelTypeIDs>,
}

impl Index<usize> for ChunkVoxels {
    type Output = u8;
    fn index(&self, i: usize) -> &u8 {
        &self.ids[i * VoxelTypeIDs::BITS_PER_VOXEL / 128].indices
            [i % VoxelTypeIDs::BITS_PER_VOXEL / 128]
    }
}
impl IndexMut<usize> for ChunkVoxels {
    fn index_mut(&mut self, i: usize) -> &mut u8 {
        debug_assert!(
            i < self.n_voxels(),
            "Tried to index ChunkVoxels with {} (total: {})",
            i,
            self.n_voxels()
        );
        &mut self.ids[i * VoxelTypeIDs::BITS_PER_VOXEL / 128].indices
            [i % (128 / VoxelTypeIDs::BITS_PER_VOXEL)]
    }
}
impl ChunkVoxels {
    pub fn new_blank(n_voxels: usize) -> Self {
        ChunkVoxels {
            ids: VoxelTypeIDs::new_vec(n_voxels),
        }
    }
    pub fn n_voxels(&self) -> usize {
        self.ids.len() * 128 / VoxelTypeIDs::BITS_PER_VOXEL
    }
}

#[derive(Clone, Debug)]
pub struct ChunkBitmask {
    pub bitmask: Vec<VoxelBitmask>,
    pub loaded: bool,
}

impl ChunkBitmask {
    pub fn new_blank(n_voxels: usize) -> Self {
        ChunkBitmask {
            bitmask: VoxelBitmask::new_vec(n_voxels),
            loaded: false,
        }
    }

    pub fn n_voxels(&self) -> usize {
        self.bitmask.len() * 128
    }

    pub fn get(&self, index: usize) -> bool {
        let bit = 1u128 << (index % 128);
        (self.bitmask[index / 128].mask & bit) > 0
    }

    pub fn set_block(&mut self, index: usize, val: bool) {
        if val {
            self.set_block_true(index);
        } else {
            self.set_block_false(index);
        }
    }

    pub fn set_block_true(&mut self, index: usize) {
        let bit = 1u128 << (index % 128);
        self.bitmask[index / 128].mask |= bit;
    }

    pub fn set_block_false(&mut self, index: usize) {
        let bit = 1u128 << (index % 128);
        self.bitmask[index / 128].mask &= !bit;
    }
}

#[derive(Clone, Debug)]
pub struct ChunkUpdateRegions {
    pub regions: Vec<BufferCopy>,
}
impl ChunkUpdateRegions {
    pub fn new() -> Self {
        ChunkUpdateRegions { regions: vec![] }
    }
}
