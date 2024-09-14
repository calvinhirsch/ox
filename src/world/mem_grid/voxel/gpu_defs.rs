use std::mem;
use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use std::ops::{Index, IndexMut};
use vulkano::command_buffer::BufferCopy;
use crate::world::mem_grid::Placeholder;


#[derive(Clone, Debug)]
pub struct ChunkVoxels {
    pub ids: Vec<VoxelTypeIDs>,
    pub loaded: bool,
}

impl Index<usize> for ChunkVoxels {
    type Output = u8;
    fn index(&self, i: usize) -> &u8 {
        debug_assert!(self.loaded, "Tried to index ChunkVoxels for an unloaded chunk");
        &self.ids[i * 8 / 128].indices[i % 8 / 128]
    }
}
impl IndexMut<usize> for ChunkVoxels {
    fn index_mut(&mut self, i: usize) -> &mut u8 {
        debug_assert!(self.loaded, "Tried to index ChunkVoxels for an unloaded chunk");
        &mut self.ids[i * 8 / 128].indices[i % 8 / 128]
    }
}
impl ChunkVoxels {
    pub fn new_blank(n_voxels: usize) -> Self {
        ChunkVoxels {
            ids: VoxelTypeIDs::new_vec(n_voxels),
            loaded: false,
        }
    }
    pub fn n_voxels(&self) -> usize { self.ids.len() * 128 / VoxelTypeIDs::BITS_PER_VOXEL }
}
impl Placeholder for ChunkVoxels {
    fn replace_with_placeholder(&mut self) -> Self {
        Self {
            ids: mem::take(&mut self.ids),
            loaded: mem::replace(&mut self.loaded, false),
        }
    }

    fn is_placeholder(&self) -> bool {
        !self.loaded && self.ids.is_empty()
    }
}


#[derive(Clone, Debug)]
pub struct ChunkBitmask{
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

    pub fn n_voxels(&self) -> usize { self.bitmask.len() * 128 / VoxelBitmask::BITS_PER_VOXEL }

    pub fn get(&self, index: usize) -> bool {
        let bit = 1u128.to_le() << (index % 128);
        (self.bitmask[index / 128].mask & bit) > 0
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
        self.bitmask[index / 128].mask |= bit;
    }

    pub fn set_block_false(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.bitmask[index / 128].mask &= !bit;
    }
}
impl Placeholder for ChunkBitmask {
    fn replace_with_placeholder(&mut self) -> Self {
        let loaded = self.loaded;
        self.loaded = false;

        Self {
            bitmask: mem::replace(&mut self.bitmask, vec![]),
            loaded,
        }
    }

    fn is_placeholder(&self) -> bool {
        return !self.loaded && self.bitmask.len() == 0
    }
}


#[derive(Clone, Debug)]
pub struct ChunkUpdateRegions {
    pub regions: Vec<BufferCopy>,
    pub loaded: bool,
}
impl ChunkUpdateRegions {
    pub fn new() -> Self {
        ChunkUpdateRegions {
            regions: vec![],
            loaded: false,
        }
    }
}
impl Placeholder for ChunkUpdateRegions {
    fn replace_with_placeholder(&mut self) -> Self {
        let loaded = self.loaded;
        self.loaded = false;

        Self {
            regions: vec![],
            loaded,
        }
    }

    fn is_placeholder(&self) -> bool {
        return !self.loaded && self.regions.len() == 0
    }
}