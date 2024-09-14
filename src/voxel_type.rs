use std::fmt::Debug;
use enum_iterator::{all, Sequence};
use num_traits::{FromPrimitive, ToPrimitive};
pub use crate::renderer::component::materials::Material;


pub struct VoxelTypeDefinition<A> {
    pub material: Material,
    pub is_visible: bool,
    pub attributes: A,
}


/// Trait for enum of all block types that must be defined. The first value (repr = 0) is assumed to
/// be an empty block (e.g. 'air').
pub trait VoxelTypeEnum:
    Sequence + Copy + FromPrimitive + ToPrimitive + Debug
{
    type VoxelAttributes;

    fn def(&self) -> VoxelTypeDefinition<Self::VoxelAttributes>;

    fn materials() -> Vec<Material> {
        // Check that ID fits in a u8.
        // ENHANCEMENT: Make this generic to u16, etc if possible
        assert!(Self::CARDINALITY <= 2usize.pow(8));
        all::<Self>()
            .map(|voxel_def| voxel_def.def().material)
            .collect()
    }
}
