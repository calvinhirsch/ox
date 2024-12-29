use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::component::voxels::lod::RendererVoxelLOD;
use crate::renderer::component::voxels::lod::{VoxelIDUpdate, VoxelLODUpdate};
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::LayerChunk;
use crate::world::mem_grid::layer::{MemoryGridLayer, MemoryGridLayerMetadata};
use crate::world::mem_grid::utils::cubed;
use crate::world::mem_grid::voxel::gpu_defs::{ChunkBitmask, ChunkUpdateRegions, ChunkVoxels};
use crate::world::mem_grid::ChunkEditor;
use crate::world::TLCPos;
use getset::{CopyGetters, Getters, MutGetters};
use hashbrown::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;
use vulkano::DeviceSize;

#[derive(Clone, Debug)]
pub struct VoxelLODCreateParams {
    pub voxel_resolution: usize,
    pub lvl: usize,
    pub sublvl: usize,
    pub render_area_size: usize, // size in chunks of one dimension, so total chunks loaded = render_area_size^3
    pub bitmask_binding: u32,
    pub voxel_types_binding: Option<u32>,
}
impl VoxelLODCreateParams {
    pub fn validate(&self, chunk_size: usize) {
        debug_assert!(self.voxel_resolution == chunk_size.pow(self.lvl as u32) * 2usize.pow(self.sublvl as u32), "VoxelLODCreateParams invalid: voxel resolution for lvl {} sublvl {} expected to be chunk_size^lvl * 2^sublvl = {}", self.lvl, self.sublvl, chunk_size.pow(self.lvl as u32) * 2usize.pow(self.sublvl as u32));
    }
}

#[derive(Clone, Debug)]
pub struct LODMetadata {
    pub lvl: usize,
    pub sublvl: usize,
    pub voxels_per_tlc: usize,
}

#[derive(Clone, Debug)]
pub struct LODLayerData {
    bitmask: ChunkBitmask,
    updated_bitmask_regions: ChunkUpdateRegions,
    voxel_ids: Option<ChunkVoxels>, // voxel ids are optional because some LODs only have a bitmask
}

pub type VoxelMemoryGridLOD = MemoryGridLayer<LODLayerData, LODMetadata>;

impl VoxelMemoryGridLOD {
    pub fn new_voxel_lod(
        params: VoxelLODCreateParams,
        start_tlc: TLCPos<i64>,
        lod_tlc_size: usize,
        buffer_allocator: Arc<dyn MemoryAllocator>,
    ) -> (Self, RendererVoxelLOD) {
        let bitmask =
            vec![ChunkBitmask::new_blank(cubed(lod_tlc_size)); cubed(params.render_area_size)];
        let voxels = params.voxel_types_binding.map(|_| {
            vec![Some(ChunkVoxels::new_blank(cubed(lod_tlc_size))); cubed(params.render_area_size)]
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
            params.voxel_types_binding,
            buffer_allocator,
        );

        (
            MemoryGridLayer::new(
                bitmask
                    .into_iter()
                    .zip(
                        voxels
                            .unwrap_or((0..cubed(params.render_area_size)).map(|_| None).collect()),
                    )
                    .map(|(bm, vx)| {
                        LayerChunk::new(LODLayerData {
                            bitmask: bm,
                            voxel_ids: vx,
                            updated_bitmask_regions: ChunkUpdateRegions::new(),
                        })
                    })
                    .collect(),
                start_tlc,
                params.render_area_size,
                LODMetadata {
                    voxels_per_tlc: cubed(lod_tlc_size),
                    lvl: params.lvl,
                    sublvl: params.sublvl,
                },
            ),
            renderer_lod,
        )
    }

    /// Aggregate the values from updated_bitmask_regions_layer into absolute regions in the
    /// bitmask buffer, generate regions for updating the voxel type ID buffer, and reset these
    /// tracked regions if clear_regions=true.
    pub fn aggregate_updates(&mut self, clear_regions: bool) -> Vec<VoxelLODUpdate> {
        let voxels_per_tlc = self.metadata().extra().voxels_per_tlc;

        self.chunks_mut()
            .iter_mut()
            .enumerate()
            .filter_map(|(chunk_i, chunk)| {
                chunk
                    .get_mut()
                    .map(|chunk| {
                        if chunk.updated_bitmask_regions.regions.len() > 0 {
                            let bm_offset =
                                (chunk_i * voxels_per_tlc / VoxelBitmask::BITS_PER_VOXEL) as u64;
                            let r = Some(VoxelLODUpdate {
                                bitmask: &chunk.bitmask.bitmask,
                                bitmask_updated_regions: chunk
                                    .updated_bitmask_regions
                                    .regions
                                    .iter()
                                    .map(|r| BufferCopy {
                                        src_offset: r.src_offset,
                                        dst_offset: r.dst_offset + bm_offset,
                                        size: r.size,
                                        ..Default::default()
                                    })
                                    .collect(),
                                id_update: chunk.voxel_ids.as_ref().map(|v| VoxelIDUpdate {
                                    ids: &v.ids,
                                    updated_regions: {
                                        let scale = (VoxelTypeIDs::BITS_PER_VOXEL
                                            / VoxelBitmask::BITS_PER_VOXEL)
                                            as u64;
                                        chunk
                                            .updated_bitmask_regions
                                            .regions
                                            .iter()
                                            .map(|r| BufferCopy {
                                                src_offset: r.src_offset * scale,
                                                dst_offset: (r.dst_offset + bm_offset) * scale,
                                                size: r.size * scale as u64,
                                                ..Default::default()
                                            })
                                            .collect()
                                    },
                                }),
                            });

                            if clear_regions {
                                chunk.updated_bitmask_regions.regions.clear();
                            }

                            r
                        } else {
                            None
                        }
                    })
                    .flatten()
            })
            .collect()
    }
}

#[derive(Debug, Getters, CopyGetters, MutGetters)]
pub struct VoxelLODChunkEditor<'a, VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    #[getset(get_mut = "pub", get = "pub")]
    data: &'a mut LayerChunk<LODLayerData>,
    #[get_copy = "pub"]
    lvl: u8,
    #[get_copy = "pub"]
    sublvl: u8,
}

impl<'a, VE: VoxelTypeEnum>
    ChunkEditor<&'a mut LayerChunk<LODLayerData>, &'a MemoryGridLayerMetadata<LODMetadata>>
    for VoxelLODChunkEditor<'a, VE>
{
    fn edit(
        chunk_data: &'a mut LayerChunk<LODLayerData>,
        metadata: &'a MemoryGridLayerMetadata<LODMetadata>,
    ) -> Self {
        VoxelLODChunkEditor {
            voxel_type_enum: PhantomData,
            data: chunk_data,
            sublvl: metadata.extra().sublvl as u8,
            lvl: metadata.extra().lvl as u8,
        }
    }
}

impl LODLayerData {
    /// Call in order to immutably access voxel data. None if no voxel data is present.
    pub fn with_voxel_ids(&self) -> Option<LODLayerDataWithVoxelIDs> {
        if self.voxel_ids.is_some() {
            Some(LODLayerDataWithVoxelIDs(self))
        } else {
            None
        }
    }

    /// Call in order to mutably access voxel data. None if no voxel data is present.
    pub fn with_voxel_ids_mut(&mut self) -> Option<LODLayerDataWithVoxelIDsMut> {
        if self.voxel_ids.is_some() {
            Some(LODLayerDataWithVoxelIDsMut(self))
        } else {
            None
        }
    }

    /// For LODs where there is only a bitmask and no voxel ID data, update the bitmask given a
    /// bitmask from a lower level. This should not be called when this LOD contains voxel ID data.
    pub fn update_bitmask_from_lower_lod(
        &mut self,
        lower_lod: &LODLayerData,
        curr_lvl: usize,
        curr_sublvl: usize,
        lower_lvl: usize,
        lower_sublvl: usize,
        chunk_size: usize,
        n_chunk_lvls: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        for vox_index in 0..cubed((chunk_size * (n_chunk_lvls - curr_lvl)).to_le() >> curr_sublvl) {
            self.update_bitmask_bit_from_lower_lod(
                vox_index,
                lower_lod,
                curr_lvl,
                curr_sublvl,
                lower_lvl,
                lower_sublvl,
                chunk_size,
                n_lods,
                lod_block_fill_thresh,
            )
        }
    }

    pub fn update_bitmask_bit_from_lower_lod(
        &mut self,
        voxel_index: usize,
        lower_lod: &LODLayerData,
        curr_lvl: usize,
        curr_sublvl: usize,
        lower_lvl: usize,
        lower_sublvl: usize,
        chunk_size: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        // Index of the lower corner of the 2x2x2 area in the lower LOD data we want to look at
        let mut visible_count = 0;
        let mut count = 0;

        for idx in voxels_in_lower_lod(
            voxel_index,
            curr_lvl,
            curr_sublvl,
            lower_lvl,
            lower_sublvl,
            n_lods,
            chunk_size,
        ) {
            count += 1;

            if lower_lod.bitmask.get(idx) {
                visible_count += 1;
            }
        }

        self.bitmask.set_block(
            voxel_index,
            visible_count as f32 >= lod_block_fill_thresh * count as f32,
        );
        self.updated_bitmask_regions.regions.push(BufferCopy {
            src_offset: (voxel_index / 8) as DeviceSize,
            dst_offset: (voxel_index / 8) as DeviceSize,
            size: 1,
            ..Default::default()
        })
    }
}

pub struct LODLayerDataWithVoxelIDsMut<'a>(&'a mut LODLayerData);
pub struct LODLayerDataWithVoxelIDs<'a>(&'a LODLayerData);

impl<'a> LODLayerDataWithVoxelIDs<'a> {
    fn voxel_ids(&self) -> &ChunkVoxels {
        self.0.voxel_ids.as_ref().unwrap()
    }
}

impl<'a> LODLayerDataWithVoxelIDsMut<'a> {
    // fn voxel_ids(&self) -> &ChunkVoxels {
    //     self.0.voxel_ids.as_ref().unwrap()
    // }

    fn voxel_ids_mut(&mut self) -> &mut ChunkVoxels {
        self.0.voxel_ids.as_mut().unwrap()
    }

    /// Overwrite voxel data. This will allow editing of the voxel IDs directly and automatically
    /// recalculate the full bitmask when VoxelLODChunkEditor is dropped. Please don't resize the
    /// voxel ID vec.
    pub fn overwrite<VE: VoxelTypeEnum>(&mut self) -> VoxelEditor<VE> {
        VoxelEditor {
            voxel_type_enum: PhantomData::<VE>,
            voxel_ids: self.0.voxel_ids.as_mut().unwrap(),
            bitmask: &mut self.0.bitmask,
            updated_bitmask_regions: &mut self.0.updated_bitmask_regions,
        }
    }

    /// Set a single voxel
    pub fn set_voxel<VE: VoxelTypeEnum>(&mut self, index: usize, voxel_typ: VE) {
        self.voxel_ids_mut()[index] = voxel_typ.to_u8().unwrap();
        self.0.bitmask.set_block(index, voxel_typ.def().is_visible);
        self.0.updated_bitmask_regions.regions.push(BufferCopy {
            src_offset: (index / 8) as DeviceSize,
            dst_offset: (index / 8) as DeviceSize,
            size: 1,
            ..Default::default()
        })
    }

    pub fn calc_from_lower_lod_voxels<VE: VoxelTypeEnum>(
        &mut self,
        lower_lod: LODLayerDataWithVoxelIDs,
        curr_lvl: usize,
        curr_sublvl: usize,
        lower_lvl: usize,
        lower_sublvl: usize,
        chunk_size: usize,
        n_chunk_lvls: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        for vox_index in 0..cubed((chunk_size * (n_chunk_lvls - curr_lvl)).to_le() >> curr_sublvl) {
            // Index of the lower corner of the 2x2x2 area in the lower LOD data we want to look at
            let mut visible_count = 0;
            let mut count = 0;
            let mut type_counts = HashMap::<u8, u32>::new();

            for idx in voxels_in_lower_lod(
                vox_index,
                curr_lvl,
                curr_sublvl,
                lower_lvl,
                lower_sublvl,
                n_lods,
                chunk_size,
            ) {
                count += 1;

                let id = lower_lod.voxel_ids()[idx];
                let vox_type = VE::from_u8(id).unwrap();
                if vox_type.def().is_visible {
                    visible_count += 1;
                    match type_counts.get_mut(&id) {
                        None => {
                            type_counts.insert(id, 1);
                        }
                        Some(c) => {
                            *c += 1;
                        }
                    }
                }
            }

            if visible_count as f32 >= lod_block_fill_thresh * count as f32 {
                self.voxel_ids_mut()[vox_index] = type_counts
                    .into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1))
                    .map(|(k, _)| k)
                    .unwrap();
            } else {
                self.voxel_ids_mut()[vox_index] = 0;
            }
        }

        calc_full_bitmask::<VE>(self.0.voxel_ids.as_mut().unwrap(), &mut self.0.bitmask);
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
/// up the voxel at 'index' in the current LOD and return an iterator over their indices.
fn voxels_in_lower_lod(
    index: usize,
    lvl: usize,
    sublvl: usize,
    lower_lvl: usize,
    lower_sublvl: usize,
    n_lods: usize,
    chunk_size: usize,
) -> Box<dyn Iterator<Item = usize>> {
    let (next_lvl, next_sublvl) = {
        if (lvl > lower_lvl && sublvl > 0) || (sublvl > lower_sublvl) {
            // Decrement sublvl
            (lvl, sublvl - 1)
        } else if lvl > lower_lvl {
            // Decrement lvl
            (lvl - 1, n_lods)
        } else {
            // At lower_lvl, lower_sublvl so done accumulating indices
            return Box::new([index].into_iter()) as Box<dyn Iterator<Item = usize>>;
        }
    };

    let next_lod_z_incr = chunk_size >> next_sublvl;
    let next_lod_y_incr = next_lod_z_incr * next_lod_z_incr;
    let next_lod_index = index.to_le() << 3;

    Box::new(
        [
            next_lod_index,
            next_lod_index + 1,
            next_lod_index + next_lod_z_incr,
            next_lod_index + 1 + next_lod_z_incr,
            next_lod_index + next_lod_y_incr,
            next_lod_index + 1 + next_lod_y_incr,
            next_lod_index + next_lod_z_incr + next_lod_y_incr,
            next_lod_index + 1 + next_lod_z_incr + next_lod_y_incr,
        ]
        .into_iter()
        .flat_map(move |i| {
            voxels_in_lower_lod(
                i,
                next_lvl,
                next_sublvl,
                lower_lvl,
                lower_sublvl,
                n_lods,
                chunk_size,
            )
        }),
    )
}

/// Provides access to chunk voxels to edit and recalculates the full bitmask when dropped.
pub struct VoxelEditor<'a, VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    pub voxel_ids: &'a mut ChunkVoxels,
    bitmask: &'a mut ChunkBitmask,
    updated_bitmask_regions: &'a mut ChunkUpdateRegions,
}
impl<'a, VE: VoxelTypeEnum> Drop for VoxelEditor<'a, VE> {
    fn drop(&mut self) {
        calc_full_bitmask::<VE>(self.voxel_ids, self.bitmask);
        self.updated_bitmask_regions.regions.push(BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: ((self.bitmask.n_voxels() + 7) / 8) as DeviceSize,
            ..Default::default()
        })
    }
}
