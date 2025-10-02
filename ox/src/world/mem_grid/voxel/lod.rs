use crate::loader::LayerChunk;
use crate::renderer::component::voxels::data::VoxelTypeIDs;
use crate::renderer::component::voxels::lod::RendererVoxelLOD;
use crate::renderer::component::voxels::lod::{VoxelIDUpdate, VoxelLODUpdate};
use crate::voxel_type::VoxelTypeEnum;
use crate::world::mem_grid::layer::MemoryGridLayer;
use crate::world::mem_grid::utils::{cubed, ChunkSize, VoxelPosInLod};
use crate::world::mem_grid::voxel::gpu_defs::{ChunkBitmask, ChunkVoxels};
use crate::world::mem_grid::EditMemoryGridChunk;
use crate::world::TlcPos;
use cgmath::Point3;
use getset::{CopyGetters, Getters, MutGetters};
use hashbrown::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;

use super::grid::lod_tlc_size;

#[derive(Clone, Debug)]
pub struct VoxelLODCreateParams {
    pub voxel_resolution: usize,
    pub lvl: u8,
    pub sublvl: u8,
    pub render_area_size: usize, // size in chunks of one dimension, so total chunks loaded = render_area_size^3
    pub bitmask_binding: u32,
    pub voxel_ids_binding: Option<u32>,
}
impl VoxelLODCreateParams {
    pub fn validate(&self, chunk_size: ChunkSize) {
        debug_assert!(self.voxel_resolution == chunk_size.size().pow(self.lvl as u32) as usize * 2usize.pow(self.sublvl as u32), "VoxelLODCreateParams invalid: voxel resolution for lvl {} sublvl {} expected to be chunk_size^lvl * 2^sublvl = {}", self.lvl, self.sublvl, chunk_size.size().pow(self.lvl as u32) * 2usize.pow(self.sublvl as u32));
    }
}

#[derive(Clone, Debug)]
pub struct LodMetadata {
    pub lvl: u8,
    pub sublvl: u8,
    pub voxels_per_tlc: usize,
}

#[derive(Clone, Debug)]
pub struct LodState {
    pub updated_regions: Vec<UpdateRegion>,
}

#[derive(Clone, Debug, Getters)]
pub struct LodChunkData {
    #[get = "pub"]
    bitmask: ChunkBitmask,
    #[get = "pub"]
    voxel_ids: Option<ChunkVoxels>, // voxel ids are optional because some LODs only have a bitmask
}

#[derive(Clone, Debug, Getters)]
pub struct LodChunkDataWithVoxels<'a> {
    #[get = "pub"]
    bitmask: &'a ChunkBitmask,
    #[get = "pub"]
    voxel_ids: &'a ChunkVoxels,
}

#[derive(Debug, Getters)]
pub struct LodChunkDataWithVoxelsMut<'a> {
    #[get = "pub"]
    bitmask: &'a mut ChunkBitmask,
    #[get = "pub"]
    voxel_ids: &'a mut ChunkVoxels,
}

pub type VoxelMemoryGridLod = MemoryGridLayer<LodChunkData, LodMetadata, LodState>;

#[derive(Debug, Clone)]
pub struct UpdateRegion {
    pub chunk_idx: usize,
    pub voxel_idx: usize,
    pub n_voxels: usize,
}

impl VoxelMemoryGridLod {
    pub fn new_voxel_lod(
        params: VoxelLODCreateParams,
        start_tlc: TlcPos<i64>,
        lod_tlc_size: usize,
        buffer_allocator: Arc<dyn MemoryAllocator>,
    ) -> (Self, RendererVoxelLOD) {
        assert!(
            params.render_area_size % 2 == 1,
            "Render area sizes should be odd so they have a center chunk"
        );
        let bitmask =
            vec![ChunkBitmask::new_blank(cubed(lod_tlc_size)); cubed(params.render_area_size + 1)];
        let voxels = params.voxel_ids_binding.map(|_| {
            vec![
                Some(ChunkVoxels::new_blank(cubed(lod_tlc_size)));
                cubed(params.render_area_size + 1)
            ]
        });
        let renderer_lod = RendererVoxelLOD::new(
            bitmask
                .iter()
                .flat_map(|c| &c.bitmask)
                .copied()
                .collect::<Vec<_>>()
                .into_iter(), // ENHANCEMENT: Do this better (and below)
            voxels.as_ref().map(|voxs| {
                voxs.iter()
                    .flat_map(|c| &c.as_ref().unwrap().ids)
                    .copied()
                    .collect::<Vec<_>>()
                    .into_iter()
            }),
            params.bitmask_binding,
            params.voxel_ids_binding,
            buffer_allocator,
        );

        (
            MemoryGridLayer::new(
                bitmask
                    .into_iter()
                    .zip(
                        voxels.unwrap_or(
                            (0..cubed(params.render_area_size + 1))
                                .map(|_| None)
                                .collect(),
                        ),
                    )
                    .map(|(bm, vx)| {
                        LayerChunk::new(LodChunkData {
                            bitmask: bm,
                            voxel_ids: vx,
                        })
                    })
                    .collect(),
                start_tlc,
                params.render_area_size + 1,
                LodMetadata {
                    voxels_per_tlc: cubed(lod_tlc_size),
                    lvl: params.lvl,
                    sublvl: params.sublvl,
                },
                LodState {
                    updated_regions: vec![],
                },
            ),
            renderer_lod,
        )
    }

    /// Aggregate the values from updated_regions to pass to the renderer and reset these
    /// tracked regions if clear_regions=true.
    pub fn aggregate_updates(&mut self, clear_regions: bool) -> Vec<VoxelLODUpdate> {
        let voxels_per_tlc = self.metadata().extra().voxels_per_tlc;
        let (chunks, state) = self.chunks_and_state_mut();
        let updates: Vec<VoxelLODUpdate> = state
            .updated_regions
            .iter()
            .filter_map(|region| {
                // skip updates to chunks that are not loaded
                chunks[region.chunk_idx].get().map(|chunk| VoxelLODUpdate {
                    bitmask: &chunk.bitmask.bitmask,
                    bitmask_updated_region: region.bitmask_copy_region(voxels_per_tlc),
                    id_update: chunk.voxel_ids.as_ref().map(|ids| VoxelIDUpdate {
                        ids: &ids.ids,
                        updated_region: region.voxel_id_copy_region(voxels_per_tlc),
                    }),
                })
            })
            .collect();

        if clear_regions {
            state.updated_regions.clear();
        }

        updates
    }
}

/// Does not save an update region for this update
pub fn update_bitmask_bit_from_lower_lod_untracked(
    bitmask: &mut ChunkBitmask,
    voxel_pos: VoxelPosInLod,
    voxel_index: usize,
    lower_lod_bitmask: &ChunkBitmask,
    lower_lvl: u8,
    lower_sublvl: u8,
    chunk_size: ChunkSize,
    largest_chunk_lvl: u8,
    block_fill_thresh: f32,
) {
    // Index of the lower corner of the 2x2x2 area in the lower LOD data we want to look at
    let mut visible_count = 0;
    let mut count = 0;

    apply_to_voxel_indices_in_lower_lod(
        voxel_pos,
        voxel_index,
        lower_lvl,
        lower_sublvl,
        chunk_size,
        largest_chunk_lvl,
        |idx| {
            count += 1;

            if lower_lod_bitmask.get(idx) {
                visible_count += 1;
            }
        },
    );

    bitmask.set_block(
        voxel_index,
        visible_count as f32 > block_fill_thresh * count as f32,
    );
}

/// For LODs where there is only a bitmask and no voxel ID data, update the bitmask given a
/// bitmask from a lower level. This should not be called when this LOD contains voxel ID data.
/// Does not save an update region for this change.
pub fn update_bitmask_from_lower_lod_untracked(
    bitmask: &mut ChunkBitmask,
    lower_lod_bitmask: &ChunkBitmask,
    curr_lvl: u8,
    curr_sublvl: u8,
    lower_lvl: u8,
    lower_sublvl: u8,
    chunk_size: ChunkSize,
    largest_chunk_lvl: u8,
    block_fill_thresh: f32,
) {
    apply_to_voxels_in_lod(
        curr_lvl,
        curr_sublvl,
        chunk_size,
        largest_chunk_lvl,
        |voxel_pos| {
            let voxel_index = voxel_pos.index(chunk_size, largest_chunk_lvl);
            update_bitmask_bit_from_lower_lod_untracked(
                bitmask,
                voxel_pos,
                voxel_index,
                lower_lod_bitmask,
                lower_lvl,
                lower_sublvl,
                chunk_size,
                largest_chunk_lvl,
                block_fill_thresh,
            );
        },
    );
}

pub enum LodChunkDataVariant<'a> {
    WithVoxels(LodChunkDataWithVoxels<'a>),
    WithoutVoxels(&'a ChunkBitmask),
}
pub enum LodChunkDataVariantMut<'a> {
    WithVoxels(LodChunkDataWithVoxelsMut<'a>),
    WithoutVoxels(&'a mut ChunkBitmask),
}

impl LodChunkData {
    pub fn check_voxel_ids(&self) -> LodChunkDataVariant {
        match self.voxel_ids.as_ref() {
            Some(voxel_ids) => LodChunkDataVariant::WithVoxels(LodChunkDataWithVoxels {
                bitmask: &self.bitmask,
                voxel_ids,
            }),
            None => LodChunkDataVariant::WithoutVoxels(&self.bitmask),
        }
    }

    pub fn check_voxel_ids_mut(&mut self) -> LodChunkDataVariantMut {
        match self.voxel_ids.as_mut() {
            Some(voxel_ids) => LodChunkDataVariantMut::WithVoxels(LodChunkDataWithVoxelsMut {
                bitmask: &mut self.bitmask,
                voxel_ids,
            }),
            None => LodChunkDataVariantMut::WithoutVoxels(&mut self.bitmask),
        }
    }
}

#[derive(Debug, Getters)]
pub struct UpdatedRegionsMut<'a> {
    regions: &'a mut Vec<UpdateRegion>,
    #[getset(get = "pub")]
    chunk_idx: usize,
}

impl<'a> UpdatedRegionsMut<'a> {
    fn borrow_mut<'b>(&'b mut self) -> UpdatedRegionsMut<'b>
    where
        'a: 'b,
    {
        UpdatedRegionsMut {
            regions: self.regions,
            chunk_idx: self.chunk_idx,
        }
    }

    fn add_region(&mut self, voxel_idx: usize, n_voxels: usize) {
        self.regions.push(UpdateRegion {
            chunk_idx: self.chunk_idx,
            voxel_idx,
            n_voxels,
        });
    }
}

impl<'a> LodChunkDataWithVoxelsMut<'a> {
    pub fn borrow_mut<'b>(&'b mut self) -> LodChunkDataWithVoxelsMut<'b>
    where
        'a: 'b,
    {
        LodChunkDataWithVoxelsMut {
            bitmask: self.bitmask,
            voxel_ids: self.voxel_ids,
        }
    }

    /// Editing this will not track changes to update GPU.
    pub fn raw_voxel_ids_mut(&mut self) -> &mut ChunkVoxels {
        self.voxel_ids
    }

    /// Recalculate LOD voxels from a lower LOD (i.e. a higher resolution LOD). Syncs entire buffer to GPU.
    pub fn update_from_lower_lod_voxels_untracked<VE: VoxelTypeEnum>(
        &mut self,
        lower_lod: LodChunkDataWithVoxels,
        curr_lvl: u8,
        curr_sublvl: u8,
        lower_lvl: u8,
        lower_sublvl: u8,
        chunk_size: ChunkSize,
        largest_chunk_lvl: u8,
        lod_block_fill_thresh: f32,
    ) {
        apply_to_voxels_in_lod(
            curr_lvl,
            curr_sublvl,
            chunk_size,
            largest_chunk_lvl,
            |pos| {
                let index = pos.index(chunk_size, largest_chunk_lvl);
                let voxel_id = self.calc_voxel_from_lower_lod::<VE>(
                    &lower_lod,
                    pos,
                    index,
                    lower_lvl,
                    lower_sublvl,
                    chunk_size,
                    largest_chunk_lvl,
                    lod_block_fill_thresh,
                );
                self.voxel_ids[index] = voxel_id.unwrap_or(VE::empty()).id();
                self.bitmask
                    .set_block(index, voxel_id.map(|v| v.def().is_visible).unwrap_or(false));
            },
        );
    }

    pub fn calc_voxel_from_lower_lod<VE: VoxelTypeEnum>(
        &mut self,
        lower_lod: &LodChunkDataWithVoxels,
        pos: VoxelPosInLod,
        index: usize,
        lower_lvl: u8,
        lower_sublvl: u8,
        chunk_size: ChunkSize,
        largest_chunk_lvl: u8,
        lod_block_fill_thresh: f32,
    ) -> Option<VE> {
        let mut visible_count = 0;
        let mut count = 0;
        let mut type_counts = HashMap::<VE, u32>::new();

        apply_to_voxel_indices_in_lower_lod(
            pos,
            index,
            lower_lvl,
            lower_sublvl,
            chunk_size,
            largest_chunk_lvl,
            |idx| {
                count += 1;
                debug_assert!(
                    idx < lower_lod.voxel_ids.n_voxels(),
                    "bad voxel index: lower_lod.voxel_ids[{}] for {}-{}",
                    idx,
                    lower_lvl,
                    lower_sublvl,
                );
                let id = lower_lod.voxel_ids[idx];
                let vox_type = VE::from_u8(id).unwrap();
                if vox_type.def().is_visible {
                    visible_count += 1;
                    match type_counts.get_mut(&vox_type) {
                        None => {
                            type_counts.insert(vox_type, 1);
                        }
                        Some(c) => {
                            *c += 1;
                        }
                    }
                }
            },
        );

        if visible_count as f32 >= lod_block_fill_thresh * count as f32 {
            Some(
                type_counts
                    .into_iter()
                    .max_by_key(|a| a.1)
                    .map(|(k, _)| k)
                    .unwrap(),
            )
        } else {
            None
        }
    }

    /// Overwrite voxel data. This will allow editing of the voxel IDs directly and automatically
    /// recalculate the full bitmask when VoxelLODChunkEditor is dropped. Please don't resize the
    /// voxel ID vec.
    pub fn overwrite<'o, VE: VoxelTypeEnum>(&'o mut self) -> LodChunkOverwriter<'o, VE>
    where
        'a: 'o,
    {
        LodChunkOverwriter {
            _t: PhantomData::<VE>,
            chunk: self.borrow_mut(),
        }
    }
}

#[derive(Debug, Getters, CopyGetters, MutGetters)]
pub struct LodChunkEditorMaybeUnloaded<'a, VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    #[getset(get = "pub", get_mut = "pub")]
    data: &'a mut LayerChunk<LodChunkData>,
    #[getset(get = "pub")]
    updated_regions: UpdatedRegionsMut<'a>,
    #[get_copy = "pub"]
    lvl: u8,
    #[get_copy = "pub"]
    sublvl: u8,
}

pub struct LodChunkEditor<'a> {
    data: &'a mut LodChunkData,
    updated_regions: UpdatedRegionsMut<'a>,
}

impl<'a, VE: VoxelTypeEnum> LodChunkEditorMaybeUnloaded<'a, VE> {
    pub fn as_loaded<'b>(&'b mut self) -> Option<LodChunkEditor<'b>> {
        Some(LodChunkEditor {
            data: self.data.get_mut()?,
            updated_regions: self.updated_regions.borrow_mut(),
        })
    }
}

impl<VE: VoxelTypeEnum> EditMemoryGridChunk<VE>
    for MemoryGridLayer<LodChunkData, LodMetadata, LodState>
{
    type ChunkEditor<'a> = LodChunkEditorMaybeUnloaded<'a, VE>
        where
            Self: 'a;

    fn edit_chunk(
        &mut self,
        pos: TlcPos<i64>,
        buffer_chunk_states: [crate::world::BufferChunkState; 3],
    ) -> Option<Self::ChunkEditor<'_>> {
        let vgrid_pos = self.chunk_vgrid_pos(pos, buffer_chunk_states)?;
        let chunk_idx = self.index_for_vgrid_pos(vgrid_pos);
        let (lvl, sublvl) = (self.metadata().extra().lvl, self.metadata().extra().sublvl);
        let (chunks, state) = self.chunks_and_state_mut();
        Some(LodChunkEditorMaybeUnloaded {
            voxel_type_enum: PhantomData,
            data: &mut chunks[chunk_idx],
            sublvl,
            lvl,
            updated_regions: UpdatedRegionsMut {
                regions: &mut state.updated_regions,
                chunk_idx,
            },
        })
    }
}

// impl<'a, VE: VoxelTypeEnum> LayerChunkEditor<'a, LodChunkData, LodMetadata, LodState>
//     for LodChunkEditorMaybeUnloaded<'a, VE>
// {
//     fn edit<'e: 'a>(
//         chunk: &'e mut LayerChunk<LodChunkData>,
//         metadata: &'e MemoryGridLayerMetadata<LodMetadata>,
//         state: &'e mut LodState,
//         chunk_idx: usize,
//     ) -> Self {
//         LodChunkEditorMaybeUnloaded {
//             voxel_type_enum: PhantomData,
//             data: chunk,
//             sublvl: metadata.extra().sublvl as u8,
//             lvl: metadata.extra().lvl as u8,
//             updated_regions: UpdatedRegionsMut {
//                 regions: &mut state.updated_regions,
//                 chunk_idx,
//             },
//         }
//     }
// }

pub enum LodChunkEditorVariant<'a> {
    WithVoxels(LodChunkEditorWithVoxels<'a>),
    WithoutVoxels(LodChunkEditorWithoutVoxels<'a>),
}

pub enum LodChunkEditorVariantMut<'a> {
    WithVoxels(LodChunkEditorWithVoxelsMut<'a>),
    WithoutVoxels(LodChunkEditorWithoutVoxelsMut<'a>),
}

impl<'a> LodChunkEditor<'a> {
    pub fn with_voxel_ids(&self) -> LodChunkEditorVariant {
        match self.data.check_voxel_ids() {
            LodChunkDataVariant::WithVoxels(data) => {
                LodChunkEditorVariant::WithVoxels(LodChunkEditorWithVoxels { data })
            }
            LodChunkDataVariant::WithoutVoxels(data) => {
                LodChunkEditorVariant::WithoutVoxels(LodChunkEditorWithoutVoxels { bitmask: data })
            }
        }
    }

    pub fn with_voxel_ids_mut(&mut self) -> LodChunkEditorVariantMut {
        match self.data.check_voxel_ids_mut() {
            LodChunkDataVariantMut::WithVoxels(data) => {
                LodChunkEditorVariantMut::WithVoxels(LodChunkEditorWithVoxelsMut {
                    data,
                    updated_regions: self.updated_regions.borrow_mut(),
                })
            }
            LodChunkDataVariantMut::WithoutVoxels(data) => {
                LodChunkEditorVariantMut::WithoutVoxels(LodChunkEditorWithoutVoxelsMut {
                    bitmask: data,
                    updated_regions: self.updated_regions.borrow_mut(),
                })
            }
        }
    }

    /// Updates a voxel from a provided lower LOD. If this LOD has no voxel IDs and only a bitmask,
    /// only the bitmask will be updated.
    pub fn update_voxel_from_lower_lod<VE: VoxelTypeEnum>(
        &mut self,
        voxel_pos: VoxelPosInLod,
        voxel_index: usize,
        lower_lod: &LodChunkDataWithVoxels,
        lower_lvl: u8,
        lower_sublvl: u8,
        chunk_size: ChunkSize,
        largest_chunk_lvl: u8,
        block_fill_thresh: f32,
    ) {
        match self.with_voxel_ids_mut() {
            LodChunkEditorVariantMut::WithVoxels(mut lod) => lod.update_voxel_from_lower_lod::<VE>(
                lower_lod,
                voxel_pos,
                voxel_index,
                lower_lvl,
                lower_sublvl,
                chunk_size,
                largest_chunk_lvl,
                block_fill_thresh,
            ),
            LodChunkEditorVariantMut::WithoutVoxels(mut lod) => lod
                .update_bitmask_bit_from_lower_lod(
                    voxel_pos,
                    voxel_index,
                    lower_lod.bitmask,
                    lower_lvl,
                    lower_sublvl,
                    chunk_size,
                    largest_chunk_lvl,
                    block_fill_thresh,
                ),
        }
    }
}

pub struct LodChunkEditorWithVoxelsMut<'a> {
    data: LodChunkDataWithVoxelsMut<'a>,
    updated_regions: UpdatedRegionsMut<'a>,
}
pub struct LodChunkEditorWithVoxels<'a> {
    #[allow(dead_code)]
    data: LodChunkDataWithVoxels<'a>,
}
pub struct LodChunkEditorWithoutVoxelsMut<'a> {
    bitmask: &'a mut ChunkBitmask,
    updated_regions: UpdatedRegionsMut<'a>,
}
pub struct LodChunkEditorWithoutVoxels<'a> {
    #[allow(dead_code)]
    bitmask: &'a ChunkBitmask,
}

pub fn apply_to_voxels_in_lod<F: FnMut(VoxelPosInLod)>(
    lvl: u8,
    sublvl: u8,
    chunk_size: ChunkSize,
    largest_chunk_lvl: u8,
    mut f: F,
) {
    let curr_lod_tlc_size = lod_tlc_size(chunk_size, largest_chunk_lvl, lvl, sublvl) as u32;
    for y in 0..curr_lod_tlc_size {
        for z in 0..curr_lod_tlc_size {
            for x in 0..curr_lod_tlc_size {
                f(VoxelPosInLod {
                    pos: Point3 { x, y, z },
                    lvl,
                    sublvl,
                });
            }
        }
    }
}

impl<'a> LodChunkEditorWithVoxelsMut<'a> {
    pub fn borrow_mut<'b>(&'b mut self) -> LodChunkEditorWithVoxelsMut<'b>
    where
        'a: 'b,
    {
        LodChunkEditorWithVoxelsMut {
            data: self.data.borrow_mut(),
            updated_regions: self.updated_regions.borrow_mut(),
        }
    }

    pub fn data<'b>(&'b mut self) -> LodChunkDataWithVoxels<'b>
    where
        'a: 'b,
    {
        LodChunkDataWithVoxels {
            bitmask: self.data.bitmask,
            voxel_ids: self.data.voxel_ids,
        }
    }

    /// Sets self.updated_regions to a single region covering the whole buffer so that it
    /// will be fully copied to the GPU.
    pub fn update_full_buffer_gpu(&mut self) {
        let chunk_idx = self.updated_regions.chunk_idx;
        self.updated_regions.regions.push(UpdateRegion {
            voxel_idx: 0,
            chunk_idx,
            n_voxels: self.data.bitmask.n_voxels(),
        });
    }

    /// Set a single voxel and add an update region for later GPU transfer
    pub fn set_voxel<VE: VoxelTypeEnum>(&mut self, index: usize, voxel_typ: VE) {
        self.data.voxel_ids[index] = voxel_typ.to_u8().unwrap();
        self.data
            .bitmask
            .set_block(index, voxel_typ.def().is_visible);
        self.updated_regions.add_region(index, 1);
    }

    /// Recalculate LOD voxels from a lower LOD (i.e. a higher resolution LOD). Syncs entire buffer to GPU.
    pub fn update_from_lower_lod_voxels<VE: VoxelTypeEnum>(
        &mut self,
        lower_lod: LodChunkDataWithVoxels,
        curr_lvl: u8,
        curr_sublvl: u8,
        lower_lvl: u8,
        lower_sublvl: u8,
        chunk_size: ChunkSize,
        largest_chunk_lvl: u8,
        lod_block_fill_thresh: f32,
    ) {
        self.data.update_from_lower_lod_voxels_untracked::<VE>(
            lower_lod,
            curr_lvl,
            curr_sublvl,
            lower_lvl,
            lower_sublvl,
            chunk_size,
            largest_chunk_lvl,
            lod_block_fill_thresh,
        );
        self.update_full_buffer_gpu();
    }

    pub fn update_voxel_from_lower_lod<VE: VoxelTypeEnum>(
        &mut self,
        lower_lod: &LodChunkDataWithVoxels,
        pos: VoxelPosInLod,
        index: usize,
        lower_lvl: u8,
        lower_sublvl: u8,
        chunk_size: ChunkSize,
        largest_chunk_lvl: u8,
        lod_block_fill_thresh: f32,
    ) {
        let voxel_type = self.data.calc_voxel_from_lower_lod::<VE>(
            &lower_lod,
            pos,
            index,
            lower_lvl,
            lower_sublvl,
            chunk_size,
            largest_chunk_lvl,
            lod_block_fill_thresh,
        );
        self.set_voxel(index, voxel_type.unwrap_or(VE::empty()));
    }
}

impl<'a> LodChunkEditorWithoutVoxelsMut<'a> {
    /// Sets self.updated_regions to a single region covering the whole buffer so that it
    /// will be fully copied to the GPU.
    pub fn update_full_buffer_gpu(&mut self) {
        let chunk_idx = self.updated_regions.chunk_idx;
        self.updated_regions.regions.push(UpdateRegion {
            voxel_idx: 0,
            chunk_idx,
            n_voxels: self.bitmask.n_voxels(),
        });
    }

    pub fn update_bitmask_bit_from_lower_lod(
        &mut self,
        voxel_pos: VoxelPosInLod,
        voxel_index: usize,
        lower_lod_bitmask: &ChunkBitmask,
        lower_lvl: u8,
        lower_sublvl: u8,
        chunk_size: ChunkSize,
        largest_chunk_lvl: u8,
        block_fill_thresh: f32,
    ) {
        update_bitmask_bit_from_lower_lod_untracked(
            self.bitmask,
            voxel_pos,
            voxel_index,
            lower_lod_bitmask,
            lower_lvl,
            lower_sublvl,
            chunk_size,
            largest_chunk_lvl,
            block_fill_thresh,
        );
        self.updated_regions.add_region(voxel_index, 1);
    }

    /// For LODs where there is only a bitmask and no voxel ID data, update the bitmask given a
    /// bitmask from a lower level. This should not be called when this LOD contains voxel ID data.
    pub fn update_bitmask_from_lower_lod(
        &mut self,
        lower_lod_bitmask: &ChunkBitmask,
        curr_lvl: u8,
        curr_sublvl: u8,
        lower_lvl: u8,
        lower_sublvl: u8,
        chunk_size: ChunkSize,
        largest_chunk_lvl: u8,
        block_fill_thresh: f32,
    ) {
        update_bitmask_from_lower_lod_untracked(
            self.bitmask,
            lower_lod_bitmask,
            curr_lvl,
            curr_sublvl,
            lower_lvl,
            lower_sublvl,
            chunk_size,
            largest_chunk_lvl,
            block_fill_thresh,
        );
        self.update_full_buffer_gpu();
    }
}

pub fn calc_full_bitmask<VE: VoxelTypeEnum>(voxels: &ChunkVoxels, bitmask: &mut ChunkBitmask) {
    for i in 0..voxels.n_voxels() {
        if VE::from_u8(voxels[i]).unwrap().def().is_visible {
            bitmask.set_block_true(i);
        } else {
            bitmask.set_block_false(i);
        }
    }
}

/// Given a current lvl/sublvl and a lower lvl/sublvl, find all the voxels in the lower LOD that make
/// up the voxel at `index`/`pt` in the current LOD and return an iterator over their indices.
fn apply_to_voxel_indices_in_lower_lod<F: FnMut(usize)>(
    voxel: VoxelPosInLod,
    voxel_index: usize,
    lower_lvl: u8,
    lower_sublvl: u8,
    chunk_size: ChunkSize,
    largest_chunk_lvl: u8,
    mut f: F,
) {
    if voxel.sublvl > 0 {
        // First find all voxels in same lvl but sublvl zero.
        // If voxel.lvl == lower_lvl then just convert to lower_sublvl instead of zero.
        // Thus, `target_sublvl` is either zero or `lower_sublvl`
        let (target_sublvl, target_sublvl_is_final) = if voxel.lvl == lower_lvl {
            debug_assert!(lower_sublvl < voxel.sublvl);
            (lower_sublvl, true)
        } else {
            (0, false)
        };

        let scale_relative_to_target_sublvl = 1u32 << (voxel.sublvl - target_sublvl);
        let pos_in_target = voxel.pos * scale_relative_to_target_sublvl; // botleft pos in target sublvl
        let start_idx_in_target = VoxelPosInLod {
            pos: pos_in_target,
            lvl: voxel.lvl,
            sublvl: target_sublvl,
        }
        .index(chunk_size, largest_chunk_lvl);
        // index increment when shifting z by 1 in target sublvl
        let target_z_incr = 1u32 << (chunk_size.exp() - target_sublvl as u8);
        // index increment when shifting y by 1 in target sublvl
        let target_y_incr = target_z_incr * target_z_incr;
        // (shifting x is always an increment of 1)

        if target_sublvl_is_final {
            for dy in 0..scale_relative_to_target_sublvl {
                for dz in 0..scale_relative_to_target_sublvl {
                    for dx in 0..scale_relative_to_target_sublvl {
                        f(start_idx_in_target
                            + (dx + dy * target_y_incr + dz * target_z_incr) as usize);
                    }
                }
            }
        } else {
            for dy in 0..scale_relative_to_target_sublvl {
                for dz in 0..scale_relative_to_target_sublvl {
                    for dx in 0..scale_relative_to_target_sublvl {
                        apply_to_voxel_indices_in_lower_lod_for_lvl(
                            start_idx_in_target
                                + (dx + dy * target_y_incr + dz * target_z_incr) as usize,
                            voxel.lvl,
                            lower_lvl,
                            lower_sublvl,
                            chunk_size,
                            &mut f,
                        );
                    }
                }
            }
        }
    } else {
        apply_to_voxel_indices_in_lower_lod_for_lvl(
            voxel_index,
            voxel.lvl,
            lower_lvl,
            lower_sublvl,
            chunk_size,
            &mut f,
        );
    }
}

/// `apply_to_voxel_indices_in_lower_lod` special case where sublvl == 0
fn apply_to_voxel_indices_in_lower_lod_for_lvl<F: FnMut(usize)>(
    voxel_index: usize,
    lvl: u8,
    lower_lvl: u8,
    lower_sublvl: u8,
    chunk_size: ChunkSize,
    f: &mut F,
) {
    let scale_relative_to_lower = 1u32 << (chunk_size.exp() * (lvl - lower_lvl) - lower_sublvl);
    let lower_voxels_per = cubed(scale_relative_to_lower) as usize;
    let first_idx_in_lower = voxel_index * lower_voxels_per;
    for idx in first_idx_in_lower..(first_idx_in_lower + lower_voxels_per) {
        f(idx);
    }
}

#[derive(Debug, Getters, MutGetters, CopyGetters)]
pub struct BorrowedLodChunk<VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    #[getset(get_mut = "pub", get = "pub")]
    data: LodChunkData,
    #[get_copy = "pub"]
    chunk_idx: usize,
    #[get_copy = "pub"]
    lvl: u8,
    #[get_copy = "pub"]
    sublvl: u8,
}

impl<VE: VoxelTypeEnum> BorrowedLodChunk<VE> {
    pub fn new(
        LodChunkEditorMaybeUnloaded {
            voxel_type_enum: _,
            data,
            lvl,
            sublvl,
            updated_regions,
        }: &mut LodChunkEditorMaybeUnloaded<VE>,
    ) -> Option<Self> {
        Some(Self {
            voxel_type_enum: PhantomData,
            data: data.take()?,
            chunk_idx: updated_regions.chunk_idx,
            lvl: *lvl,
            sublvl: *sublvl,
        })
    }
}

impl<VE: VoxelTypeEnum> BorrowedLodChunk<VE> {
    pub fn return_data(self, lod: &mut VoxelMemoryGridLod) {
        lod.chunks_mut()[self.chunk_idx] = LayerChunk::new_valid(self.data);
    }
}

/// Provides access to chunk voxels to edit and recalculates the full bitmask when dropped.
pub struct LodChunkOverwriter<'a, VE: VoxelTypeEnum> {
    pub chunk: LodChunkDataWithVoxelsMut<'a>,
    _t: PhantomData<VE>,
}

impl<'a, VE: VoxelTypeEnum> Drop for LodChunkOverwriter<'a, VE> {
    fn drop(&mut self) {
        calc_full_bitmask::<VE>(&self.chunk.voxel_ids, &mut self.chunk.bitmask);
    }
}

const MIN_BITS_PER_TLC_BITMASK: usize = 128;

impl UpdateRegion {
    pub fn bitmask_copy_region(&self, voxels_per_tlc: usize) -> BufferCopy {
        let voxel_offset = self.voxel_idx / 8;
        BufferCopy {
            src_offset: voxel_offset as u64,
            dst_offset: (self.chunk_idx * voxels_per_tlc.max(MIN_BITS_PER_TLC_BITMASK) / 8
                + voxel_offset) as u64,
            size: (self.n_voxels / 8).max(1) as u64,
            ..Default::default()
        }
    }

    pub fn voxel_id_copy_region(&self, voxels_per_tlc: usize) -> BufferCopy {
        let bytes_per_voxel = if voxels_per_tlc >= MIN_BITS_PER_TLC_BITMASK {
            VoxelTypeIDs::BITS_PER_VOXEL / 8
        } else {
            (voxels_per_tlc * VoxelTypeIDs::BITS_PER_VOXEL).max(MIN_BITS_PER_TLC_BITMASK) / 8
        };
        BufferCopy {
            src_offset: (self.voxel_idx * bytes_per_voxel) as u64,
            dst_offset: ((self.chunk_idx * voxels_per_tlc + self.voxel_idx) * bytes_per_voxel)
                as u64,
            size: (self.n_voxels * bytes_per_voxel).max(1) as u64,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use enum_iterator::Sequence;
    use num_derive::{FromPrimitive, ToPrimitive};

    use crate::voxel_type::{Material, VoxelTypeDefinition};

    use super::*;

    #[test]
    fn test_apply_to_all_voxels_in_lod_0_0() {
        let mut indices = [false; 64 * 64 * 64];
        let cs = ChunkSize::new(3);
        apply_to_voxels_in_lod(0, 0, cs, 2, |pos| {
            let idx = pos.index(cs, 2);
            assert!(!indices[idx]);
            indices[idx] = true;
        });
        assert!(indices.into_iter().all(|x| x));
    }

    #[test]
    fn test_apply_to_all_voxels_in_lod_0_1() {
        let mut indices = [false; 32 * 32 * 32];
        let cs = ChunkSize::new(3);
        apply_to_voxels_in_lod(0, 1, cs, 2, |pos| {
            let idx = pos.index(cs, 2);
            assert!(!indices[idx]);
            indices[idx] = true;
        });
        assert!(indices.into_iter().all(|x| x));
    }

    #[test]
    fn test_apply_to_all_voxels_in_lod_0_2() {
        let mut indices = [false; 16 * 16 * 16];
        let cs = ChunkSize::new(3);
        apply_to_voxels_in_lod(0, 2, cs, 2, |pos| {
            let idx = pos.index(cs, 2);
            assert!(!indices[idx]);
            indices[idx] = true;
        });
        assert!(indices.into_iter().all(|x| x));
    }

    #[test]
    fn test_apply_to_all_voxels_in_lod_1_0() {
        let mut indices = [false; 8 * 8 * 8];
        let cs = ChunkSize::new(3);
        apply_to_voxels_in_lod(1, 0, cs, 2, |pos| {
            let idx = pos.index(cs, 2);
            assert!(!indices[idx]);
            indices[idx] = true;
        });
        assert!(indices.into_iter().all(|x| x));
    }

    #[derive(Debug, Sequence, Clone, Copy, FromPrimitive, ToPrimitive, Hash, PartialEq, Eq)]
    pub enum Block {
        AIR,
        SOLID,
    }

    impl VoxelTypeEnum for Block {
        type VoxelAttributes = ();

        fn def(&self) -> VoxelTypeDefinition<Self::VoxelAttributes> {
            use Block::*;
            match *self {
                AIR => VoxelTypeDefinition {
                    material: Material::default(),
                    is_visible: false,
                    attributes: (),
                },
                SOLID => VoxelTypeDefinition {
                    material: Material {
                        color: [1., 0., 0.],
                        emission_color: [1., 0., 0.],
                        emission_strength: 1.2,
                        ..Default::default()
                    },
                    is_visible: true,
                    attributes: (),
                },
            }
        }

        fn empty() -> Block {
            Block::AIR
        }
    }

    #[test]
    fn test_calc_full_bitmask() {
        let mut voxels = ChunkVoxels::new_blank(32 * 32 * 32);
        let mut bm = ChunkBitmask::new_blank(32 * 32 * 32);
        voxels[0] = 1;
        calc_full_bitmask::<Block>(&voxels, &mut bm);

        let true_bm = {
            let mut bm = ChunkBitmask::new_blank(32 * 32 * 32);
            bm.set_block_true(0);
            bm
        };
        assert_eq!(bm, true_bm);
    }
}
