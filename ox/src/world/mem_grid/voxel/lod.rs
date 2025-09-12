use crate::renderer::component::voxels::data::VoxelTypeIDs;
use crate::renderer::component::voxels::lod::RendererVoxelLOD;
use crate::renderer::component::voxels::lod::{VoxelIDUpdate, VoxelLODUpdate};
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::LayerChunk;
use crate::world::mem_grid::layer::{MemoryGridLayer, MemoryGridLayerMetadata};
use crate::world::mem_grid::utils::{cubed, ChunkSize, VoxelPosInLod};
use crate::world::mem_grid::voxel::gpu_defs::{ChunkBitmask, ChunkUpdateRegions, ChunkVoxels};
use crate::world::mem_grid::ChunkEditor;
use crate::world::TlcPos;
use cgmath::Point3;
use getset::{CopyGetters, Getters, MutGetters};
use hashbrown::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;
use vulkano::DeviceSize;

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
pub struct LODMetadata {
    pub lvl: u8,
    pub sublvl: u8,
    pub voxels_per_tlc: usize,
}

#[derive(Clone, Debug, Getters)]
pub struct LODLayerData {
    #[get = "pub"]
    bitmask: ChunkBitmask,
    // NOTE: `dst_offset` values in `BufferCopy`s are relative to the current chunk when stored here.
    // Later, the offset of these chunks in the full staging/local buffer will be added to it.
    #[get = "pub"]
    updated_bitmask_regions: ChunkUpdateRegions,
    #[get = "pub"]
    voxel_ids: Option<ChunkVoxels>, // voxel ids are optional because some LODs only have a bitmask
}

pub type VoxelMemoryGridLOD = MemoryGridLayer<LODLayerData, LODMetadata>;

impl VoxelMemoryGridLOD {
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
                        LayerChunk::new(LODLayerData {
                            bitmask: bm,
                            voxel_ids: vx,
                            updated_bitmask_regions: ChunkUpdateRegions::new(),
                        })
                    })
                    .collect(),
                start_tlc,
                params.render_area_size + 1,
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
                            let bm_bytes_per_tlc = voxels_per_tlc.max(128) / 8; // take max with 128 here because if voxels per tlc is < 128 we still use a full u128
                            let bm_offset = (chunk_i * bm_bytes_per_tlc) as u64; // offset for this chunk's bitmask in bytes

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
                                        let scale = if voxels_per_tlc >= 128 {
                                            VoxelTypeIDs::BITS_PER_VOXEL
                                        } else {
                                            ((voxels_per_tlc * VoxelTypeIDs::BITS_PER_VOXEL)
                                                .max(128)
                                                / 8)
                                                / bm_bytes_per_tlc
                                        }
                                            as u64;
                                        chunk
                                            .updated_bitmask_regions
                                            .regions
                                            .iter()
                                            .map(|r| BufferCopy {
                                                src_offset: r.src_offset * scale,
                                                dst_offset: (r.dst_offset + bm_offset) * scale,
                                                size: r.size * scale,
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
                self.update_bitmask_bit_from_lower_lod(
                    voxel_pos,
                    voxel_index,
                    lower_lod,
                    lower_lvl,
                    lower_sublvl,
                    chunk_size,
                    largest_chunk_lvl,
                    block_fill_thresh,
                )
            },
        );
    }

    pub fn update_bitmask_bit_from_lower_lod(
        &mut self,
        voxel_pos: VoxelPosInLod,
        voxel_index: usize,
        lower_lod: &LODLayerData,
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

                if lower_lod.bitmask.get(idx) {
                    visible_count += 1;
                }
            },
        );

        self.bitmask.set_block(
            voxel_index,
            visible_count as f32 > block_fill_thresh * count as f32,
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
    pub fn voxel_ids(&self) -> &ChunkVoxels {
        self.0.voxel_ids.as_ref().unwrap()
    }
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

impl<'a> LODLayerDataWithVoxelIDsMut<'a> {
    // fn voxel_ids(&self) -> &ChunkVoxels {
    //     self.0.voxel_ids.as_ref().unwrap()
    // }

    /// Direct mutable access to voxel IDs. WARNING: modifying this will not automatically track
    /// updated regions so the changes may not be synced to the GPU. In order to do this, use
    /// either `overwrite` or `set_voxel`.
    pub fn voxel_ids_mut(&mut self) -> &mut ChunkVoxels {
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

    /// Sets self.updated_bitmask_regions to a single region covering the whole buffer so that it
    /// will be fully copied to the GPU.
    pub fn update_full_buffer_gpu(&mut self) {
        self.0.updated_bitmask_regions.regions = vec![BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: ((self.0.bitmask.n_voxels() + 7) / 8) as DeviceSize,
            ..Default::default()
        }];
    }

    /// Set a single voxel and add an update region for later GPU transfer
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

    /// Recalculate LOD voxels from a lower LOD (i.e. a higher resolution LOD). Syncs entire buffer to GPU.
    pub fn calc_from_lower_lod_voxels<VE: VoxelTypeEnum>(
        &mut self,
        lower_lod: LODLayerDataWithVoxelIDs,
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
            |voxel_pos| {
                let voxel_index = voxel_pos.index(chunk_size, largest_chunk_lvl);

                let mut visible_count = 0;
                let mut count = 0;
                let mut type_counts = HashMap::<u8, u32>::new();

                apply_to_voxel_indices_in_lower_lod(
                    voxel_pos,
                    voxel_index,
                    lower_lvl,
                    lower_sublvl,
                    chunk_size,
                    largest_chunk_lvl,
                    |idx| {
                        count += 1;
                        debug_assert!(
                                idx < lower_lod.voxel_ids().n_voxels(),
                                "bad voxel index: lower_lod.voxel_ids[{}] for {}-{}  (curr LOD: {}-{} count {})",
                                idx,
                                lower_lvl,
                                lower_sublvl,
                                curr_lvl,
                                curr_sublvl,
                                cubed(lod_tlc_size(
                                    chunk_size,
                                    largest_chunk_lvl,
                                    curr_lvl,
                                    curr_sublvl,
                                )),
                            );
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
                    },
                );

                if visible_count as f32 >= lod_block_fill_thresh * count as f32 {
                    self.voxel_ids_mut()[voxel_index] = type_counts
                        .into_iter()
                        .max_by_key(|a| a.1)
                        .map(|(k, _)| k)
                        .unwrap();
                } else {
                    self.voxel_ids_mut()[voxel_index] = VE::empty();
                }
            },
        );

        calc_full_bitmask::<VE>(self.0.voxel_ids.as_mut().unwrap(), &mut self.0.bitmask);
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
pub struct BorrowedVoxelLODChunkEditor<VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    #[getset(get_mut = "pub", get = "pub")]
    data: *mut LayerChunk<LODLayerData>,
    #[get_copy = "pub"]
    lvl: u8,
    #[get_copy = "pub"]
    sublvl: u8,
}

impl<VE: VoxelTypeEnum> BorrowedVoxelLODChunkEditor<VE> {
    pub fn new(
        VoxelLODChunkEditor {
            voxel_type_enum: _,
            data,
            lvl,
            sublvl,
        }: &mut VoxelLODChunkEditor<VE>,
    ) -> Self {
        Self {
            voxel_type_enum: PhantomData,
            data: *data as *mut _,
            lvl: *lvl,
            sublvl: *sublvl,
        }
    }
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
        self.updated_bitmask_regions.regions = vec![BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: ((self.bitmask.n_voxels() + 7) / 8) as DeviceSize,
            ..Default::default()
        }];
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

    #[derive(Debug, Sequence, Clone, Copy, FromPrimitive, ToPrimitive)]
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

        fn empty() -> u8 {
            0
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
