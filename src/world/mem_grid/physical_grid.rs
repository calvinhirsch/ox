use cgmath::{Array, Point3, Vector3};
use num_traits::Zero;
use super::{
    MemoryGridMetadata, MemoryGridLayerMetadata,
    utils::{cubed},
    rendering::RenderingLayerSet,
    virtual_grid::{VirtualMemoryGrid, TopLevelChunk},
};
use crate::{
    world::{
        VoxelPos, TLCPos, WorldCreateParams,
        camera::Camera,
    },
    voxel_type::{VoxelTypeEnum},
};
use crate::world::mem_grid::MemoryGridLayerCreateParams;
use crate::world::mem_grid::virtual_grid::MemoryGridChunkData;


pub trait MemoryGridLayerExtraMetadata {
    type ExtraCreateParams: Sized;

    fn new(params: MemoryGridLayerCreateParams<Self::ExtraCreateParams>) -> Self;
}

pub trait MemoryGridLayerSet: Sized {
    type ChunkData : MemoryGridChunkData;
    type LayerCreateParams: Sized;
    type LayerSetCreateParams : Sized;
    type LayerExtraMetadata: MemoryGridLayerExtraMetadata;
    type LayerSetMetadata: Sized;

    fn new(grid_meta: &MemoryGridMetadata, params: MemoryGridLayerCreateParams<Self::LayerSetCreateParams>) -> Self;

    fn to_virtual_grid_format(self, grid_meta: &MemoryGridMetadata) -> (Vec<Self::ChunkData>,  Self::LayerSetMetadata);

    fn from_virtual(chunks: Vec<Self::ChunkData>, data_layer_meta: Self::LayerSetMetadata, grid_meta: &MemoryGridMetadata) -> Option<Self>;
}


#[derive(Clone)]
pub struct MemoryGridLayer<DL: MemoryGridLayerSet> {
    chunks: Vec<Option<DL::ChunkData>>,
    meta: MemoryGridLayerMetadata<DL::LayerCreateParams, DL::LayerExtraMetadata>,
}
impl<DL: MemoryGridLayerSet> MemoryGridLayer<DL> {
    pub fn new(params: MemoryGridLayerCreateParams<DL::LayerSetCreateParams>, grid_meta: &MemoryGridMetadata) -> Self {
        MemoryGridLayer {
            chunks: vec![DL::ChunkData::new(grid_meta.tlc_size, &params); cubed(params.size)],
            meta: MemoryGridLayerMetadata::new(params, grid_meta),
        }
    }
}

#[derive(Clone)]
pub struct MemoryGrid<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> {
    data_layers: DL,
    rendering_layers: RenderingLayerSet<VE>,
    gen_func: fn(VoxelPos<i64>) -> VE,
    meta: MemoryGridMetadata,
}

impl<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> MemoryGrid<VE, DL> {
    pub fn new(
        params: WorldCreateParams<VE, DL>,
    ) -> Option<MemoryGrid<VE, DL>> {
        let max_size: usize = *(params.render_area_sizes.iter().map(|sizes| sizes.iter().max()).max()??.as_ref())?;
        let start_tlc = TLCPos(params.curr_tlc.0 - Vector3::<i64>::from_value((max_size/2) as i64));

        let meta = MemoryGridMetadata {
            size: max_size,
            n_chunk_lvls: params.n_chunk_lvls,
            tlc_size: params.chunk_size.pow(params.n_chunk_lvls as u32),
            n_lods: params.n_lods,
            start_tlc,
            chunk_size: params.chunk_size,
            load_thresh_dist: params.load_thresh_dist,
            lod_block_fill_thresh: params.lod_block_fill_thresh,
        };

        Some(MemoryGrid {
            data_layers: DL::new(&meta, params.layer_params),
            rendering_layers: RenderingLayerSet::<VE>::new(&meta, params.rendering_layer_params),
            gen_func: params.gen_func,
            meta,
        })
    }

    pub fn new_raw(
        data_layers: DL,
        rendering_layers: RenderingLayerSet<VE>,
        gen_func: fn(VoxelPos<i64>) -> VE,
        meta: MemoryGridMetadata,
    ) -> Self {
        MemoryGrid { data_layers, rendering_layers, gen_func, meta }
    }

    pub fn move_grid(&mut self, camera: &mut Camera) {
        let move_vector = (
            camera.position
            / (self.meta.chunk_size.pow(self.meta.n_chunk_lvls as u32) as f32)
        )
            .map(|a| a.floor() as i64)
            - Point3::<i64>::from_value(((self.meta.size-1)/2) as i64);

        if !move_vector.is_zero() {
            camera.position -= (move_vector * self.meta.chunk_size.pow(self.meta.n_chunk_lvls as u32) as i64)
                .cast::<f32>().unwrap();

            // Shift offsets and possibly start loading new chunks
            todo!();
        }
    }

    pub fn to_virtual(self) -> VirtualMemoryGrid<VE, DL> {
        let (data_chunks, chunk_metadata) =
            self.data_layers.to_virtual_grid_format(&self.meta);
        let (rendering_data_chunks, rendering_chunk_metadata) =
            self.rendering_layers.to_virtual_grid_format(&self.meta);

        let mut vg = VirtualMemoryGrid::new(
            data_chunks.into_iter().zip(rendering_data_chunks.into_iter())
                .map(|(data_chunk, rendering_data_chunk)| {
                    TopLevelChunk::new(data_chunk, rendering_data_chunk)
                }).collect(),
            rendering_chunk_metadata,
            chunk_metadata,
            self.gen_func,
            MemoryGridMetadata {
                size: self.meta.size - 1,
                ..self.meta
            },
        );

        vg
    }
}
