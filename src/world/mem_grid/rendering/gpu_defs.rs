use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};
use num_traits::PrimInt;
use vulkano::buffer::BufferContents;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::mem_grid::{MemoryGridLayerChunkData, NewChunkData};
use crate::world::mem_grid::utils::cubed;


// pub trait BufferContentsVec {
//     type Contents;
//
//     fn vec(&self) -> &Vec<Self::Contents>;
// }

#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
pub struct VoxelTypeIDs {
    indices: [u8; 128/8]  // TODO: Make this generic somehow so you can use u16 or u32
}

#[derive(Clone)]
pub struct ChunkVoxelIDs(Vec<VoxelTypeIDs>);

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
impl<VE: VoxelTypeEnum> MemoryGridLayerChunkData<VE> for ChunkVoxelIDs {
    fn set_voxel(&mut self, index: usize, voxel_type: VE) -> Option<()> {
        self[index] = voxel_type.into();
        Some(())
    }
}
impl ChunkVoxelIDs {
    const BITS_PER_VOXEL: usize = 8;

    pub fn new(chunk_size: usize) -> ChunkVoxelIDs {
        ChunkVoxelIDs(vec![VoxelTypeIDs { indices: [0; 128/8] }; (cubed(chunk_size)*8 + 127) / 128])
    }
}
// impl BufferContentsVec for ChunkVoxelIDs {
//     type Contents = VoxelTypeIDs;
//
//     fn vec(&self) -> &Vec<VoxelTypeIDs> { &self.0 }
// }


#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub struct BlockBitmask {
    mask: u128,
}

impl Display for BlockBitmask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#0128b}", self.mask)
    }
}

#[derive(Clone)]
pub struct ChunkBitmask(Vec<BlockBitmask>);

impl ChunkBitmask {
    const BITS_PER_VOXEL: usize = 1;

    pub fn new(chunk_size: usize) -> ChunkBitmask {
        ChunkBitmask(vec![BlockBitmask { mask: 0 }; (cubed(chunk_size) + 127) / 128])
    }

    pub fn set_block_true(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index/128].mask |= bit;
    }

    pub fn set_block_false(&mut self, index: usize) {
        let bit = 1u128.to_le() << (index % 128);
        self.0[index/128].mask &= !bit;
    }
}
// impl BufferContentsVec for ChunkBitmask {
//     type Contents = BlockBitmask;
//
//     fn vec(&self) -> &Vec<BlockBitmask> { &self.0 }
// }
impl<VE: VoxelTypeEnum> MemoryGridLayerChunkData<VE> for ChunkBitmask {
    fn set_voxel(&mut self, index: usize, voxel_type: VE) -> Option<()> {
        if voxel_type.def().is_visible {
            self.set_block_true(index);
        }
        else {
            self.set_block_false(index);
        }
        Some(())
    }
}