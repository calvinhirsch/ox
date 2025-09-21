use enum_iterator::Sequence;
use num_derive::{FromPrimitive, ToPrimitive};

use ox::voxel_type::{Material, VoxelTypeDefinition, VoxelTypeEnum};

pub struct BlockTypeAttrs {
    #[allow(dead_code)]
    dollars: u32,
}

#[derive(Debug, Sequence, Clone, Copy, FromPrimitive, ToPrimitive, PartialEq, Eq, Hash)]
pub enum Block {
    Air,
    Debug,
    Grass,
    Dirt,
    Mirror,
    RedLight,
    GreenLight,
    BlueLight,
    Metal,
    GrayCarpet,
}

impl VoxelTypeEnum for Block {
    type VoxelAttributes = BlockTypeAttrs;

    fn def(&self) -> VoxelTypeDefinition<Self::VoxelAttributes> {
        use Block::*;
        match *self {
            Air => VoxelTypeDefinition {
                material: Material::default(),
                is_visible: false,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
            Debug => VoxelTypeDefinition {
                material: Material {
                    color: [1., 0., 0.],
                    emission_color: [1., 0., 0.],
                    emission_strength: 1.2,
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
            Grass => VoxelTypeDefinition {
                material: Material {
                    color: [0.0745, 0.42747, 0.08235],
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 3 },
            },
            Dirt => VoxelTypeDefinition {
                material: Material {
                    color: [0.44, 0.32, 0.25],
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 3 },
            },
            Mirror => VoxelTypeDefinition {
                material: Material {
                    color: [0.5, 0.5, 0.5],
                    specular_color: [1., 1., 1.],
                    specular_prob_parallel: 1.0,
                    specular_prob_perpendicular: 1.0,
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
            RedLight => VoxelTypeDefinition {
                material: Material {
                    color: [1., 0., 0.],
                    emission_color: [1., 0., 0.],
                    emission_strength: 2.0,
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
            GreenLight => VoxelTypeDefinition {
                material: Material {
                    color: [0., 1., 0.],
                    emission_color: [0., 1., 0.],
                    emission_strength: 2.0,
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
            BlueLight => VoxelTypeDefinition {
                material: Material {
                    color: [0., 0., 1.],
                    emission_color: [0., 0., 1.],
                    emission_strength: 2.0,
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
            Metal => VoxelTypeDefinition {
                material: Material {
                    color: [0.3, 0.3, 0.3],
                    specular_color: [0.3, 0.3, 0.3],
                    specular_prob_parallel: 0.6,
                    specular_prob_perpendicular: 0.12,
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
            GrayCarpet => VoxelTypeDefinition {
                material: Material {
                    color: [0.3, 0.3, 0.3],
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
        }
    }

    fn empty() -> Block {
        Block::Air
    }
}
