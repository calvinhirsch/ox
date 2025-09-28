pub use crate::renderer::component::materials::Material;
use enum_iterator::{all, Sequence};
use num_traits::{FromPrimitive, ToPrimitive};
use std::{fmt::Debug, hash::Hash};

pub struct VoxelTypeDefinition<A> {
    pub material: Material,
    pub is_visible: bool,
    pub attributes: A,
}

/// Trait for enum of all block types that must be defined. The first value (repr = 0) is assumed to
/// be an empty block (e.g. 'air').
pub trait VoxelTypeEnum:
    Sequence + Copy + FromPrimitive + ToPrimitive + Debug + Eq + Hash + Send
{
    type VoxelAttributes;

    fn def(&self) -> VoxelTypeDefinition<Self::VoxelAttributes>;

    fn empty() -> Self;

    fn materials() -> Vec<Material> {
        // Check that ID fits in a u8.
        // ENHANCEMENT: Make this generic to u16, etc if possible
        assert!(Self::CARDINALITY <= 2usize.pow(8));
        all::<Self>()
            .map(|voxel_def| voxel_def.def().material)
            .collect()
    }

    fn id(&self) -> u8 {
        self.to_u8().unwrap()
    }
}
