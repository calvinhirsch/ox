use std::collections::HashMap;
use std::marker::PhantomData;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::BufferCopy;

pub mod gpu_defs;

use gpu_defs::{ChunkBitmask, ChunkVoxelIDs, BlockBitmask, VoxelTypeIDs};
use crate::{
    world::{
        mem_grid::{
            MemoryGridMetadata, MemoryGridLayerChunkData,
            utils::{cubed},
        }
    },
    voxel_type::{VoxelTypeEnum},
    renderer::Renderer,
};
use crate::world::mem_grid::{MemoryGridLayerCreateParams, MemoryGridLayerMetadata, NewChunkData};
use crate::world::mem_grid::physical_grid::{MemoryGridLayer, MemoryGridLayerExtraMetadata, MemoryGridLayerSet};



/// Rendering data for a single chunk at a single level of detail (i.e. in a specific layer)
#[derive(Clone)]
pub struct RenderingLayerChunkData<VE: VoxelTypeEnum> {
    bitmask: ChunkBitmask,
    voxel_type_ids: Option<ChunkVoxelIDs>,
}
impl<VE: VoxelTypeEnum> NewChunkData<RenderingLayerCreateParams> for RenderingLayerChunkData<VE> {
    fn new(tlc_size: usize, _: &MemoryGridLayerCreateParams<RenderingLayerCreateParams>) -> Self {
        RenderingLayerChunkData {
            bitmask: ChunkBitmask::new(tlc_size),
            voxel_type_ids: VoxelTypeIDs::new(tlc_size),
        }
    }
}

/// Create params for a rendering layer (i.e. a single chunk level and level of detail)
pub struct RenderingLayerCreateParams {
    lvl: usize,
    lod: usize,
    bitmask_staging_buffer: Subbuffer<[BlockBitmask]>,
    voxel_type_ids_staging_buffer: Subbuffer<[VoxelTypeIDs]>,
}

/// Extra metadata for a single rendering layer
pub struct RenderingLayerExtraMetadata {
    lvl: usize,
    lod: usize,
    bitmask_staging_buffer: Subbuffer<[BlockBitmask]>,
    voxel_type_ids_staging_buffer: Subbuffer<[VoxelTypeIDs]>,
}
impl MemoryGridLayerExtraMetadata for RenderingLayerExtraMetadata {
    type ExtraCreateParams = RenderingLayerCreateParams;

    fn new(params: MemoryGridLayerCreateParams<RenderingLayerCreateParams>) -> Self {
        RenderingLayerExtraMetadata {
            lvl: params.extra.lvl,
            lod: params.extra.lod,
            bitmask_staging_buffer: params.extra.bitmask_staging_buffer,
            voxel_type_ids_staging_buffer: params.extra.voxel_type_ids_staging_buffer,
        }
    }
}

pub struct TrackedData<D> {
    data: D,
    updated_regions: Vec<BufferCopy>,
}

/// Full rendering data for a single top level chunk, used in virtual memory grid
#[derive(Clone)]
pub struct ChunkRenderingData<VE: VoxelTypeEnum> {
    layers: Vec<Vec<Option<TrackedData<RenderingLayerChunkData<VE>>>>>,
}

/// Set of all rendering layers, used in physical memory grid
pub struct RenderingLayerSet<VE: VoxelTypeEnum> {
    renderer: Renderer,
    layers: Vec<Vec<Option<MemoryGridLayer<VE, RenderingLayerChunkData<VE>>>>>,
}


/// Metadata for each rendering layer in RenderingLayerSet
pub struct RenderingLayerSetMetadata {
    renderer: Renderer,
    layers: Vec<Vec<Option<MemoryGridLayerMetadata<RenderingLayerExtraMetadata>>>>,
}

/// Create params for each rendering layer in RenderingLayerSet
pub struct RenderingLayerSetCreateParams {
    renderer: Renderer,
    layers: Vec<Vec<Option<MemoryGridLayerCreateParams<RenderingLayerCreateParams>>>>,
}


impl<VE: VoxelTypeEnum> MemoryGridLayerSet for RenderingLayerSet<VE> {
    type ChunkData = ChunkRenderingData<VE>;
    type LayerCreateParams = RenderingLayerCreateParams;
    type LayerSetCreateParams = RenderingLayerSetCreateParams;
    type LayerExtraMetadata = RenderingLayerExtraMetadata;
    type LayerSetMetadata = RenderingLayerSetMetadata;

    fn new(grid_meta: &MemoryGridMetadata, params: RenderingLayerSetCreateParams) -> Self {
        RenderingLayerSet {
            renderer: params.renderer,
            layers: params.layers.into_vec().map(|lvls| {
                lvls.into_iter().map(|params_o| {
                    match params_o {
                        None => None,
                        Some(p) => {
                            Some(MemoryGridLayer::new(p, grid_meta))
                        }
                    }
                })
            }).flatten().collect()
        }
    }

    fn to_virtual_grid_format(self, grid_meta: &MemoryGridMetadata) -> (Vec<Self::ChunkData>, Self::LayerSetMetadata) {
        todo!();

        let mut grid = data_chunks.map(|dc| {
            TopLevelChunk::new(
                dc,
                vec![vec![None; self.meta.n_lods]; self.meta.n_chunk_lvls+1],
            )
        }).collect();
        let mut meta = vec![vec![None; self.meta.n_lods]; self.meta.n_chunk_lvls+1];
        for (lvl, lod_layers) in self.rendering_layers.into_iter().enumerate() {
            for (lod, layer_o) in lod_layers.into_iter().enumerate() {
                match layer_o {
                    None => (),
                    Some(layer) => {
                        meta[lvl][lod] = Some(layer.meta);
                        for (chunk_i, chunk) in layer.chunks.into_iter().enumerate() {
                            let virtual_pos = meta[lvl][lod].unwrap().virtual_grid_pos_for_index(chunk_i, self.meta.size).0;
                            grid[pos_index(virtual_pos, self.meta.size-1)].rendering_layer_data[lvl][lod] = chunk;
                        }
                    }
                }
            }
        }

        ()
    }

    fn from_virtual(chunks: Vec<Self::ChunkData>, data_layer_meta: Self::LayerSetMetadata, grid_meta: &MemoryGridMetadata) -> Option<Self> {
        todo!();

        let mut rendering_layer_chunks_out: Vec<Vec<Option<Vec<Option<RenderingLayerChunkData<VE>>>>>> =
            self.rendering_layer_metadata.iter().map(|lods| {
                lods.iter().map(|data| {
                    match data {
                        None => None,
                        Some(meta) => { Some(vec![None; meta.size]) }
                    }
                }).collect()
            }).collect();

        let mut updated_regions: Vec<Vec<Vec<BufferCopy>>> = vec![vec![vec![]; self.meta.n_lods]; self.meta.n_chunk_lvls+1];

        for (i, chunk) in self.top_level_chunks.into_iter().enumerate() {
            for (lvl, lvl_data) in chunk.rendering_data.into_iter().enumerate() {
                for (lod, data_o) in lvl_data.into_iter().enumerate() {
                    match data_o {
                        None => {},
                        Some(data) => {
                            let meta = self.rendering_layer_metadata[lvl][lod]?;
                            let layer_idx = meta.index_for_virtual_grid_pos(TLCVector(pos_for_index(i, self.meta.size)), self.meta.size);

                            for copy_info in data.bitmask_updated_regions:

                            let bitmask_region_start_bytes = layer_idx * cubed(self.meta.size) * ChunkBitmask::BITS_PER_VOXEL / 8;
                            let voxel_ids_region_start_bytes = layer_idx * cubed(self.meta.size) * ChunkVoxelIDs::BITS_PER_VOXEL / 8;
                            rendering_layer_chunks_out[lvl][lod]?[layer_idx] = ;
                        }
                    }

                }
            }
        }

        // TODO: Aggregate all updated regions, copy to staging buffers, and then return them in updated_regions

        rendering_layer_chunks_out.into_iter().zip(self.layer_metadata).map(|(chunks_lvl, meta_lvl)| {
            chunks_lvl.into_iter().zip(meta_lvl).map(|(chunks, meta)| {
                match chunks {
                    None => None,
                    Some(data) => Some(
                        MemoryGridLayer::new_raw(
                            data.try_into().unwrap(),
                            meta?,
                        )
                    ),
                }
            }).collect()
        }).collect(),
    }
}

impl<VE: VoxelTypeEnum> RenderingLayerChunkData<VE> {
    pub fn set_from_voxels(
        &mut self,
        voxels: ChunkVoxelIDs,
        grid_meta: &MemoryGridMetadata,
    ) {
        self.voxel_type_ids = Some(voxels);
        self.update_full_bitmask(grid_meta);
    }

    fn calc_from_lower_lod_voxels(
        &mut self,
        voxels: &ChunkVoxelIDs,
        curr_lvl: usize,
        curr_lod: usize,
        lower_lod: usize,
        lower_lvl: usize,
        grid_meta: &MemoryGridMetadata,
    ) {
        // let (lower_lod, lower_lvl) = if curr_lod == 0 { (grid_meta.n_lods - 1, curr_lvl - 1) } else { (curr_lod - 1, curr_lvl) };
        let lower_lod_z_incr = grid_meta.chunk_size >> lower_lod;
        let lower_lod_y_incr = lower_lod_z_incr*lower_lod_z_incr;

        for vox_index in 0..cubed((grid_meta.chunk_size*(grid_meta.n_chunk_lvls - self.meta.lvl)).to_le() >> self.meta.lod) {
            // Index of the lower corner of the 2x2x2 area in the lower LOD data we want to look at
            let mut visible_count = 0;
            let mut type_counts = HashMap::<u8, u32>::new();

            // Gather all indices for blocks in lower LOD to check
            // TODO: This is probably inefficient as it stores all the indices in a big ass vec.
            //       There is probably a way to do this without storing.
            let lower_lod_voxel_idxs = |idx: usize| {
                let lower_lod_index = idx.to_le() << 3;
                [
                    lower_lod_index,
                    lower_lod_index + 1,
                    lower_lod_index + lower_lod_z_incr,
                    lower_lod_index + 1 + lower_lod_z_incr,
                    lower_lod_index + lower_lod_y_incr,
                    lower_lod_index + 1 + lower_lod_y_incr,
                    lower_lod_index + lower_lod_z_incr + lower_lod_y_incr,
                    lower_lod_index + 1 + lower_lod_z_incr + lower_lod_y_incr,
                ].into_iter()
            };

            let mut voxel_idxs = vec![vox_index];
            for _ in 0..self.meta.lod {
                voxel_idxs = voxel_idxs.into_iter().flat_map(lower_lod_voxel_idxs).collect();
            }
            for seed_idx in voxel_idxs {
                let voxels_per_seed = cubed((grid_meta.chunk_size*(self.meta.lvl - lower_lvl)) >> lower_lod);
                for i in 0..voxels_per_seed {
                    let id = voxels[seed_idx*voxels_per_seed + i];
                    let vox_type = VE::try_from(id).unwrap();
                    if vox_type.def().is_visible {
                        visible_count += 1;
                        match type_counts.get_mut(&id) {
                            None => { type_counts.insert(id, 1); },
                            Some(c) => { *c += 1; },
                        }
                    }
                }
            }

            if visible_count >= grid_meta.lod_block_fill_thresh {
                self.voxel_type_ids[vox_index] = type_counts.into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1)).map(|(k, v)| k).unwrap();
            }
            else {
                self.voxel_type_ids[vox_index] = 0;
            }
        }

        self.update_full_bitmask(grid_meta);
    }

    fn update_full_bitmask(&mut self, grid_meta: &MemoryGridMetadata) {
        for vox_index in 0..cubed((grid_meta.chunk_size*(grid_meta.n_chunk_lvls - self.meta.lvl)).to_le() >> self.meta.lod) {
            if VE::try_from(self.voxel_type_ids[vox_index]).unwrap().def().is_visible {
                self.bitmask.set_block_true(vox_index);
            }
            else {
                self.bitmask.set_block_false(vox_index);
            }
        }
    }
}