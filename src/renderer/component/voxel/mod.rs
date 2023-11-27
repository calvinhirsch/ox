use itertools::Itertools;
use vulkano::buffer::BufferContents;
use crate::renderer::buffers::{BufferScheme, ConstantBuffer, DualBuffer};
use crate::renderer::component::{DataComponent, DataComponentSet};
use crate::renderer::component::voxel::lod::{VoxelLODUpdateRegions, RendererVoxelLOD};

pub mod data;
pub mod lod;


pub struct VoxelData {
    lods: Vec<Vec<Option<RendererVoxelLOD>>>,
}

impl VoxelData {
    pub fn new(lods: Vec<Vec<Option<RendererVoxelLOD>>>) -> Self {
        VoxelData { lods }
    }

    pub fn set_updated_regions(&mut self, regions: Vec<Vec<Option<VoxelLODUpdateRegions>>>) {
        for (lvl_regions, lvl) in regions.into_iter().zip(self.lods.iter_mut()) {
            for (lod_regions_o, lod_o) in lvl_regions.into_iter().zip(lvl.iter_mut()) {
                match (lod_regions_o, lod_o) {
                    (Some(lod_regions), Some(lod)) => {
                        lod.update_regions = lod_regions;
                    },
                    (None, None) => {},
                    _ => panic!(),
                }
            }
        }
    }
}

impl DataComponentSet for VoxelData {
    fn list_dynamic_components(&self) -> Vec<&DataComponent<DualBuffer<dyn BufferContents>>> {
        self.lods.iter().flatten().filter_map_ok(|o| o?).collect()
    }

    fn list_constant_components(&self) -> Vec<&DataComponent<ConstantBuffer<dyn BufferContents>>> {
        vec![]
    }
}