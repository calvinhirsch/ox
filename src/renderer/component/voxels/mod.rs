use crate::renderer::buffers::{DynamicBufferScheme};
use crate::renderer::component::voxels::lod::{RendererVoxelLOD, VoxelLODUpdate};
use crate::renderer::component::{DataComponent, DataComponentSet};
use itertools::Itertools;
use vulkano::buffer::BufferContents;
use crate::renderer::buffers::dual::ConstantDeviceLocalBuffer;

pub mod data;
pub mod lod;

pub struct VoxelData {
    lods: Vec<Vec<Option<RendererVoxelLOD>>>,
}

impl VoxelData {
    pub fn new(lods: Vec<Vec<Option<RendererVoxelLOD>>>) -> Self {
        VoxelData { lods }
    }

    pub fn update_staging_buffers(&mut self, updates: Vec<Vec<Option<VoxelLODUpdate>>>) {
        for (lvl_updates, lvl) in updates.into_iter().zip(self.lods.iter_mut()) {
            for (lod_updates_o, lod_o) in lvl_updates.into_iter().zip(lvl.iter_mut()) {
                match (lod_updates_o, lod_o) {
                    (Some(lod_updates), Some(lod)) => {
                        lod.update_staging_buffers(lod_updates);
                    }
                    (None, None) => {}
                    _ => panic!(),
                }
            }
        }
    }
}

impl DataComponentSet for VoxelData {
    fn dynamic_components_mut(&mut self) -> Vec<&mut DataComponent<dyn DynamicBufferScheme>> {
        self.lods.iter().flatten().filter_map_ok(|o| o?).collect()
    }

    fn constant_components_mut(&mut self) -> Vec<&mut DataComponent<ConstantDeviceLocalBuffer<dyn BufferContents>>> {
        vec![]
    }
}
