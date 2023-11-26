use cgmath::{Vector3};
use itertools::Itertools;
use syn::__private::quote::__private::ext::RepToTokensExt;
use crate::renderer::{context::Context};
use crate::renderer::component::voxel::VoxelData;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::mem_grid::{PhysicalMemoryGrid, VirtualMemoryGrid};
use crate::world::mem_grid::rendering::gpu_defs::ChunkVoxelIDs;
use crate::world::TLCPos;
use super::lod::{LODCreateParams, RenderingGridLOD, RenderingGridLODChunkData, RenderingGridLODMetadata};

pub struct RenderingMemoryGrid {
    metadata: RenderingMemoryGridMetadata,
    lods: Vec<Vec<Option<RenderingGridLOD>>>
}

struct RenderingMemoryGridMetadata {
    size: usize,
    n_lvls: usize,
    n_lods: usize,
}

struct RenderingMemoryGridChunkData<'a> {
    lods: Vec<Vec<Option<RenderingGridLODChunkData<'a>>>>
}

pub struct VirtualRenderingMemoryGrid<'a> {
    metadata: &'a RenderingMemoryGridMetadata,
    lod_metadata: Vec<Vec<&'a RenderingGridLODMetadata>>,
    chunks: Vec<RenderingMemoryGridChunkData<'a>>,
}

impl RenderingMemoryGrid {
    fn voxels_per_tlc(chunk_size: usize, n_lvls: usize, lvl: usize, lod: usize) -> usize {
        chunk_size.pow((n_lvls - lvl) as u32).to_le() >> (lod*3)
    }

    pub fn new(lod_params: Vec<Vec<Option<LODCreateParams>>>, chunk_size: usize, start_tlc: TLCPos<i64>) -> (Self, VoxelData) {
        let size = lod_params.iter().flatten().filter_map_ok(|p| Some(p?.size)).iter().max().unwrap();
        let n_lvls = lod_params.len();
        let n_lods = lod_params.iter().next().unwrap().len();

        let renderer_context = Context::new();

        let (grid_lods, lods) = lod_params
            .into_iter().enumerate().map(|(lvl, lvl_sizes)|
            lvl_sizes.into_iter().enumerate().map(|(lod, params)| {
                let voxels_per_tlc = Self::voxels_per_tlc(chunk_size, n_lvls, lvl, lod);
                RenderingGridLOD::new(
                    params?,
                    voxels_per_tlc,
                    start_tlc,
                    renderer_context.memory_allocator,
                )
            }).unzip()
        ).unzip();

        (
            RenderingMemoryGrid {
                lods: grid_lods,
                metadata: RenderingMemoryGridMetadata { size, n_lvls, n_lods }
            },
            VoxelData::new(lods)
        )
    }

    pub fn set_updated_regions(&mut self, renderer_voxel_data: &mut VoxelData) {
        let updated_regions = self.lods.iter_mut().map(|lod_o|
            match lod_o {
                None => None,
                Some(lod) => lod.aggregate_updated_regions(),
            }
        ).collect();

        renderer_voxel_data.set_updated_regions(updated_regions);
    }
}
impl<VE: VoxelTypeEnum> PhysicalMemoryGrid<VE, VirtualRenderingMemoryGrid> for RenderingMemoryGrid {
    fn shift_offsets(&mut self, shift: Vector3<i64>) {
        for lod in self.lods.iter_mut().flatten() {
            lod.shift_offsets(shift);
        }
    }

    fn size(&self) -> usize {
        self.metadata.size
    }

    fn as_virtual(self) -> VirtualRenderingMemoryGrid {
        self.as_virtual_for_size(self.metadata.size)
    }

    fn as_virtual_for_size(self, grid_size: usize) -> VirtualRenderingMemoryGrid {
        let (lod_metadata, v_lods) = self.lods.map(|lvl_lods|
            lvl_lods.map(|lod_o| {
                match lod_o {
                    None => (None, None),
                    Some(lod) => {
                        let (lod_meta, chunks) = lod.as_virtual_for_size(grid_size).deconstruct();
                        (
                            lod_meta,
                            chunks
                        )
                    }
                }
            }).unzip()
        ).unzip();

        VirtualRenderingMemoryGrid {
            metadata: &self.metadata,
            lod_metadata,
            chunks: LODSplitter(v_lods).into_iter().collect(),
        }
    }
}

impl<VE: VoxelTypeEnum> VirtualMemoryGrid<VE> for VirtualRenderingMemoryGrid {
    fn load_or_generate_tlc(&self, voxel_output: &mut ChunkVoxelIDs, tlc: TLCPos<i64>) {
        todo!()
    }

    fn reload_all(&mut self) {
        todo!()
    }
}


struct LODSplitter<I: Iterator>(Vec<Vec<Option<I>>>);

impl<I: Iterator> Iterator for LODSplitter<I> {
    type Item = Vec<Vec<I::Item>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.iter_mut().map(|iters|
            iters.iter().filter_map_ok(|iter| iter?.next()).collect()
        ).collect()
    }
}