use std::error::Error;
use std::fmt::{Display, Formatter};
use std::mem::discriminant;
use enum_iterator::{all, Sequence};
use vulkano::buffer::BufferContents;
use num_traits::{Bounded, PrimInt};


#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
pub struct Material {
    pub color: [f32; 3],
    _pad1: f32,
    pub specular_color: [f32; 3],
    _pad2: f32,
    pub emission_color: [f32; 3],
    pub emission_strength: f32,
    pub specular_prob_perpendicular: f32,
    pub specular_prob_parallel: f32,
    _pad3: [f32; 2],
}

impl Default for Material {
    fn default() -> Self {
        Material {
            color: [0., 0., 0.],
            specular_color: [0., 0., 0.],
            emission_color: [0., 0., 0.],
            emission_strength: 0.,
            specular_prob_perpendicular: 0.,
            specular_prob_parallel: 0.,
            _pad1: 0.,
            _pad2: 0.,
            _pad3: [0., 0.],
        }
    }
}



pub struct VoxelTypeDefinition<A> {
    pub material: Material,
    pub is_visible: bool,
    pub attributes: A,
}

#[derive(Debug)]
pub struct UnknownVoxelIDError {
    val: u8,
}
impl Error for UnknownVoxelIDError {}
impl Display for UnknownVoxelIDError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Voxel ID {} failed to resolve to a known voxel type", self.val)
    }
}

/// Trait for enum of all block types that must be defined. The first value (repr = 0) is assumed to
/// be an empty block (e.g. 'air').
pub trait VoxelTypeEnum: Sequence + Copy + TryFrom<u8, Error = UnknownVoxelIDError> + Into<u8> {
    type VoxelAttributes;

    fn def(&self) -> VoxelTypeDefinition<Self::VoxelAttributes>;

    fn materials(&self) -> Vec<Material> {
        assert!(all::<Self>().collect::<Vec<Self>>().len() <= 2usize.pow(8));
        all::<Self>().map(|voxel_def| voxel_def.def().material).collect()
    }
}