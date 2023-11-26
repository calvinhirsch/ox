use std::fmt::{Display, Formatter};
use vulkano::buffer::BufferContents;

#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
pub struct VoxelTypeIDs {
    pub indices: [u8; 128/8]  // TODO: Make this generic somehow so you can use u16 or u32
}
impl VoxelTypeIDs {
    pub const BITS_PER_VOXEL: usize = 1;
}


#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub struct BlockBitmask {
    pub mask: u128,
}

impl BlockBitmask {
    pub const BITS_PER_VOXEL: usize = 8;
}

impl Display for BlockBitmask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#0128b}", self.mask)
    }
}