use std::fmt::{Display, Formatter};
use vulkano::buffer::BufferContents;

#[derive(BufferContents, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct VoxelTypeIDs {
    pub indices: [u8; 128 / 8], // ENHANCEMENT: Make this generic somehow so you can use u16 or u32
}
impl VoxelTypeIDs {
    pub const BITS_PER_VOXEL: usize = 8;

    pub fn new_vec(n_voxels: usize) -> Vec<Self> {
        vec![
            VoxelTypeIDs {
                indices: [0; 128 / 8]
            };
            (n_voxels * Self::BITS_PER_VOXEL + 127) / 128
        ]
    }
}

#[derive(BufferContents, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct VoxelBitmask {
    pub mask: u128,
}

impl VoxelBitmask {
    pub fn new_vec(n_voxels: usize) -> Vec<Self> {
        vec![VoxelBitmask { mask: 0 }; (n_voxels + 127) / 128]
    }
}

impl Display for VoxelBitmask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#0128b}", self.mask)
    }
}
