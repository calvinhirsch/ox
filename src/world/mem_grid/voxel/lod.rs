use std::marker::PhantomData;
use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::component::voxels::lod::RendererVoxelLOD;
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::world::mem_grid::layer::{MemoryGridLayerChunkData, MemoryGridLayerData, MemoryGridLayerMetadata, PhysicalMemoryGridLayer};
use crate::world::mem_grid::utils::cubed;
use crate::world::mem_grid::voxel::gpu_defs::{ChunkBitmask, ChunkVoxelIDs};
use crate::world::mem_grid::{FromVirtual, MemoryGridMetadata, PhysicalMemoryGrid, PhysicalMemoryGridStruct, Placeholder, ToVirtual, VirtualMemoryGridStruct};
use crate::world::{TLCPos, TLCVector};
use std::sync::Arc;
use derive_more::Deref;
use hashbrown::HashMap;
use itertools::Itertools;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::ChunkLoadQueueItem;


pub struct VoxelLODCreateParams {
    pub size: usize,
    pub bitmask_binding: u32,
    pub voxel_type_ids_binding: Option<u32>,
}

#[derive(Deref)]
pub struct VoxelLOD<VE: VoxelTypeEnum>(PhysicalMemoryGridStruct<VoxelLODData<VE>, VoxelLODMetadata>);
pub type VirtualVoxelLOD<VE> = VirtualMemoryGridStruct<VoxelLODChunkData<VE>, VirtualVoxelLODMetadata>;

pub struct VoxelLODData<VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    bitmask_layer: PhysicalMemoryGridLayer<Vec<VoxelBitmask>, ()>,
    voxel_type_id_layer: Option<PhysicalMemoryGridLayer<Vec<VoxelTypeIDs>, ()>>,
    updated_bitmask_regions_layer: PhysicalMemoryGridLayer<Vec<BufferCopy>, ()>,
}

#[derive(Clone)]
pub struct VoxelLODMetadata {
    size: usize,
    voxels_per_tlc: usize,
    start_tlc: TLCPos<i64>,
}
impl MemoryGridMetadata for VoxelLODMetadata {
    // ENHANCEMENT: Requiring these to be stored in metadata is kind of a bad solution, especially
    //              for start_tlc because you always have to remember to update it in shift(). The
    //              main annoyance with fixing this is the set_voxel functions.
    fn size(&self) -> usize { self.size }
    fn start_tlc(&self) -> TLCPos<i64> { self.start_tlc }
}

#[derive(Clone)]
pub struct VirtualVoxelLODMetadata {
    this: VoxelLODMetadata,
    bitmask: MemoryGridLayerMetadata<()>,
    voxel_type_ids: Option<MemoryGridLayerMetadata<()>>,
    updated_regions: MemoryGridLayerMetadata<()>,
}
impl MemoryGridMetadata for VirtualVoxelLODMetadata {
    fn size(&self) -> usize { self.this.size }
    fn start_tlc(&self) -> TLCPos<i64> { self.this.start_tlc() }
}

#[derive(Clone)]
pub struct VoxelLODChunkData<VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    pub bitmask: MemoryGridLayerChunkData<ChunkBitmask>,
    pub voxel_type_ids: Option<MemoryGridLayerChunkData<ChunkVoxelIDs>>,
    pub updated_bitmask_regions: MemoryGridLayerChunkData<Vec<BufferCopy>>,
}


impl<VE: VoxelTypeEnum> VoxelLOD<VE> {
    pub fn new(
        params: VoxelLODCreateParams,
        voxels_per_tlc: usize,
        start_tlc: TLCPos<i64>,
        tlc_size: usize,
        buffer_allocator: Arc<dyn MemoryAllocator>,
    ) -> (Self, RendererVoxelLOD) {
        let bitmask = vec![VoxelBitmask::new_vec(voxels_per_tlc); cubed(params.size)];
        let voxel_ids = params.voxel_type_ids_binding.map(|_| vec![VoxelTypeIDs::new_vec(voxels_per_tlc); cubed(params.size)]);
        let lod = RendererVoxelLOD::new(
            bitmask.iter().flatten().copied().collect::<Vec<_>>().into_iter(),  // ENHANCEMENT: Do this better (and below)
            voxel_ids.as_ref().map(|ids| ids.iter().flatten().copied().collect::<Vec<_>>().into_iter()),
            params.bitmask_binding,
            params.voxel_type_ids_binding,
            buffer_allocator,
        );

        let common_layer_meta = MemoryGridLayerMetadata::new(
            start_tlc,
            params.size,
            tlc_size,
            ()
        );

        (
            VoxelLOD(
                PhysicalMemoryGridStruct::new(
                    VoxelLODData {
                        voxel_type_enum: PhantomData,
                        bitmask_layer: PhysicalMemoryGridLayer::new(
                            PhysicalMemoryGridStruct {
                                metadata: common_layer_meta.clone(),
                                data: MemoryGridLayerData::new(bitmask, vec![false; cubed(params.size)]),
                            }
                        ),
                        voxel_type_id_layer: voxel_ids.map(|vids| PhysicalMemoryGridLayer::new(
                                PhysicalMemoryGridStruct {
                                    data: MemoryGridLayerData::new(vids, vec![false; cubed(params.size)]),
                                    metadata: common_layer_meta.clone(),
                                }
                            )),
                        updated_bitmask_regions_layer: PhysicalMemoryGridLayer::new(
                            PhysicalMemoryGridStruct {
                                data: MemoryGridLayerData::new(vec![vec![]; cubed(params.size)], vec![false; cubed(params.size)]),
                                metadata: common_layer_meta,
                            }
                        )
                    },
                    VoxelLODMetadata {
                        size: params.size,
                        voxels_per_tlc,
                        start_tlc,
                    }
                )

            ),
            lod,
        )
    }

    /// Aggregate the values from updated_bitmask_regions_layer into absolute regions in the
    /// bitmask buffer, generate regions for updating the voxel type ID buffer, and reset these
    /// tracked regions if clear_regions=true.
    pub fn aggregate_updates(&mut self, clear_regions: bool) -> Vec<VoxelLODUpdate> {
        let mut updates = vec![];
        let has_voxel_ids = self.data.voxel_type_id_layer.is_some();
        let voxels_per_tlc = self.metadata.voxels_per_tlc;
        let n_chunks = cubed(self.0.metadata.size);

        for (chunk_i, ((regions, bitmask), voxel_type_ids)) in
            self.0.data
            .updated_bitmask_regions_layer
            .borrow_mem_mut()
            .iter_mut()
            .zip(self.0.data.bitmask_layer.borrow_mem().iter())
            .zip(match self.0.data.voxel_type_id_layer {
                None => (0..n_chunks).map(|_| None).collect::<Vec<_>>(),
                Some(ref layer) =>
                    layer.borrow_mem().iter().map(Some).collect::<Vec<_>>(),
            })
            .enumerate()
        {
            let mut bitmask_updated_regions = vec![];
            let mut voxel_id_updated_regions = match has_voxel_ids {
                false => None,
                true => Some(vec![]),
            };
            let bm_offset = chunk_i * voxels_per_tlc / VoxelBitmask::BITS_PER_VOXEL;

            for region in regions.iter_mut() {
                bitmask_updated_regions.push(BufferCopy {
                    src_offset: region.src_offset,
                    dst_offset: region.dst_offset + bm_offset as u64,
                    size: region.size,
                    ..Default::default()
                })
            }

            match &mut voxel_id_updated_regions {
                None => {}
                Some(vi_regions) => {
                    let scale = VoxelTypeIDs::BITS_PER_VOXEL / VoxelBitmask::BITS_PER_VOXEL;
                    for region in regions.iter_mut() {
                        vi_regions.push(BufferCopy {
                            src_offset: region.src_offset * scale as u64,
                            dst_offset: (region.dst_offset + bm_offset as u64) * scale as u64,
                            size: region.size * scale as u64,
                            ..Default::default()
                        })
                    }
                }
            }

            if clear_regions {
                regions.clear();
            }

            updates.push(
                VoxelLODUpdate {
                    bitmask,
                    voxel_type_ids,
                    bitmask_updated_regions,
                    voxel_id_updated_regions,
                }
            )
        }

        updates
    }
}


impl<VE: VoxelTypeEnum> PhysicalMemoryGrid for VoxelLOD<VE> {
    type Data = VoxelLODData<VE>;
    type Metadata = VoxelLODMetadata;
    type ChunkLoadQueueItemData = ();

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<()>> {
        // Because the queues for all three of the layers will be the same size, only need to get one.
        self.0.data.bitmask_layer.queue_load_all()
    }

    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<()>> {
        // Because all three of these queues will be the same size, only need to track one.
        self.0.data.voxel_type_id_layer.as_mut().map(
            |layer| layer.shift(shift, load_in_from_edge, load_buffer)
        );
        self.0.data.updated_bitmask_regions_layer.shift(shift, load_in_from_edge, load_buffer);
        let r = self.0.data.bitmask_layer.shift(shift, load_in_from_edge, load_buffer);

        self.0.metadata.start_tlc = self.data.bitmask_layer.start_tlc();

        r
    }
}


impl<VE: VoxelTypeEnum> ToVirtual<VoxelLODChunkData<VE>, VirtualVoxelLODMetadata> for VoxelLOD<VE> {
    fn to_virtual_for_size(self, grid_size: usize) -> VirtualVoxelLOD<VE> {
        // ENHANCEMENT: Make this call .to_virtual_for_size(self.size()) on children instead of
        // use grid_size and then add the necessary padding in this function.
        let (data, metadata) = self.0.deconstruct();

        let (bitmask_chunks, bitmask_meta) = data.bitmask_layer
            .to_virtual_for_size(grid_size)
            .deconstruct();

        let (voxel_id_chunks, voxel_id_meta) = match data.voxel_type_id_layer {
            None => (None, None),
            Some(voxel_id_layer) => {
                let (a, b) = voxel_id_layer
                    .to_virtual_for_size(grid_size)
                    .deconstruct();
                (Some(a), Some(b))
            },
        };

        let (chunk_regions, regions_meta) = data.updated_bitmask_regions_layer
            .to_virtual_for_size(grid_size)
            .deconstruct();

        VirtualMemoryGridStruct::new(
            bitmask_chunks
                .into_iter()
                .zip(voxel_id_chunks.unwrap_or(vec![None; chunk_regions.len()]))
                .zip(chunk_regions)
                .map(
                    |(
                         (
                             bitmask,
                             voxel_type_ids
                         ),
                         updated_bitmask_regions
                     )|
                        bitmask.map(|bm| VoxelLODChunkData {
                                voxel_type_enum: PhantomData,
                                bitmask: bm,
                                voxel_type_ids,
                                updated_bitmask_regions: updated_bitmask_regions.unwrap(),
                            })
                )
                .collect(),
            VirtualVoxelLODMetadata {
                this: metadata,
                bitmask: bitmask_meta,
                voxel_type_ids: voxel_id_meta,
                updated_regions: regions_meta,
            }
        )
    }
}


impl<VE: VoxelTypeEnum> FromVirtual<VoxelLODChunkData<VE>, VirtualVoxelLODMetadata> for VoxelLOD<VE> {
    fn from_virtual_for_size(virtual_grid: VirtualMemoryGridStruct<VoxelLODChunkData<VE>, VirtualVoxelLODMetadata>, grid_size: usize) -> Self {
        let (data, metadata) = virtual_grid.deconstruct();
        let (bitmask_grid, voxel_id_grid, update_grid) =
            data.into_iter().map(
                |chunk_o| match chunk_o {
                    None => (None, None, None),
                    Some(chunk) => (Some(chunk.bitmask), chunk.voxel_type_ids, Some(chunk.updated_bitmask_regions)),
                }
            ).multiunzip();

        VoxelLOD(
            PhysicalMemoryGridStruct {
                data: VoxelLODData {
                    voxel_type_enum: PhantomData,
                    bitmask_layer: PhysicalMemoryGridLayer::from_virtual_for_size(
                        VirtualMemoryGridStruct::new(
                            bitmask_grid,
                            metadata.bitmask,
                        ),
                        grid_size,
                    ),
                    voxel_type_id_layer: metadata.voxel_type_ids.map(|m|
                        PhysicalMemoryGridLayer::from_virtual_for_size(
                                VirtualMemoryGridStruct::new(
                                    voxel_id_grid,
                                    m,
                                ),
                                grid_size,
                            )),
                    updated_bitmask_regions_layer: PhysicalMemoryGridLayer::from_virtual_for_size(
                        VirtualMemoryGridStruct::new(
                            update_grid,
                            metadata.updated_regions,
                        ),
                        grid_size,
                    ),
                },
                metadata: metadata.this,
            }
        )
    }
}


pub fn calc_full_bitmask<VE: VoxelTypeEnum>(voxel_ids: &ChunkVoxelIDs, bitmask: &mut ChunkBitmask) {
    for i in 0..voxel_ids.n_voxels() {
        if VE::from_u8(voxel_ids[i]).unwrap().def().is_visible {
            bitmask.set_block_true(i);
        }
        else {
            bitmask.set_block_false(i);
        }
    }
}


/// Given a current lvl/LOD and a lower lvl/LOD, find all the voxels in the lower lvl/LOD that make
/// up the voxel at 'index' in the current lvl/LOD and return an iterator over their indices.
fn voxels_in_lower_lod(
    index: usize,
    lvl: usize,
    lod: usize,
    lower_lvl: usize,
    lower_lod: usize,
    n_lods: usize,
    chunk_size: usize,
) -> Box<dyn Iterator<Item = usize>> {
    let (next_lvl, next_lod) = {
        if (lvl > lower_lvl && lod > 0) || (lod > lower_lod) {
            // Decrement LOD
            (lvl, lod-1)
        }
        else if lvl > lower_lvl {
            // Decrement lvl
            (lvl-1, n_lods)
        }
        else {
            // At lower_lvl, lower_lod so done accumulating indices
            return Box::new([index].into_iter()) as Box<dyn Iterator<Item = usize>>
        }
    };

    let next_lod_z_incr = chunk_size >> next_lod;
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
        ].into_iter().flat_map(move |i|
            voxels_in_lower_lod(
                i,
                next_lvl,
                next_lod,
                lower_lvl,
                lower_lod,
                n_lods,
                chunk_size,
            )
        )
    )
}


/// Provides access to chunk voxels to edit and recalculates the full bitmask when dropped.
pub struct VoxelLODChunkEditor<'a, VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    pub voxel_ids: &'a mut ChunkVoxelIDs,
    bitmask: &'a mut ChunkBitmask,
}
impl<'a, VE: VoxelTypeEnum> Drop for VoxelLODChunkEditor<'a, VE> {
    fn drop(&mut self) {
        calc_full_bitmask::<VE>(self.voxel_ids, self.bitmask);
    }
}

impl<VE: VoxelTypeEnum> Placeholder for VoxelLODChunkData<VE> {
    fn placeholder(&self) -> Self {
        VoxelLODChunkData {
            voxel_type_enum: PhantomData,
            bitmask: self.bitmask.placeholder(),
            voxel_type_ids: self.voxel_type_ids.as_ref().map(|vti| vti.placeholder()),
            updated_bitmask_regions: self.updated_bitmask_regions.placeholder(),
        }
    }
}
impl<VE: VoxelTypeEnum> VoxelLODChunkData<VE> {
    /// Overwrite voxel data. This will allow editing of the voxel IDs directly and automatically
    /// recalculate the full bitmask when VoxelLODChunkEditor is dropped. Please don't resize the
    /// voxel ID vec.
    pub fn overwrite(&mut self) -> VoxelLODChunkEditor<VE> {
        debug_assert!(self.voxel_type_ids.is_some());

        VoxelLODChunkEditor {
            voxel_type_enum: PhantomData::<VE>,
            voxel_ids: &mut self.voxel_type_ids.as_mut().unwrap().data,
            bitmask: &mut self.bitmask.data,
        }
    }

    /// Set a single voxel
    pub fn set_voxel(&mut self, index: usize, voxel_typ: VE) {
        debug_assert!(self.voxel_type_ids.is_some());

        self.voxel_type_ids.as_mut().unwrap().data[index] = voxel_typ.to_u8().unwrap();
        self.bitmask.data.set_block(index, voxel_typ.def().is_visible);
    }

    /// For LODs where there is only a bitmask and no voxel ID data, update the bitmask given a
    /// bitmask from a lower level. This should not be called when this LOD contains voxel ID data.
    pub fn update_bitmask_from_lower_lod(
        &mut self,
        bitmask: &ChunkBitmask,
        curr_lvl: usize,
        curr_lod: usize,
        lower_lod: usize,
        lower_lvl: usize,
        chunk_size: usize,
        n_chunk_lvls: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        // ENHANCEMENT: Figure out how to encode this limitation (and others in this impl) in type
        //              state. This will probably require a big rework because you couldn't store
        //              LODs as a vector anymore.
        debug_assert!(self.voxel_type_ids.is_none());

        for vox_index in 0..cubed((chunk_size*(n_chunk_lvls - curr_lvl)).to_le() >> curr_lod) {
            self.update_bitmask_bit_from_lower_lod(
                vox_index,
                bitmask,
                curr_lvl,
                curr_lod,
                lower_lod,
                lower_lvl,
                chunk_size,
                n_lods,
                lod_block_fill_thresh,
            )
        }
    }

    pub fn update_bitmask_bit_from_lower_lod(
        &mut self,
        voxel_index: usize,
        bitmask: &ChunkBitmask,
        curr_lvl: usize,
        curr_lod: usize,
        lower_lod: usize,
        lower_lvl: usize,
        chunk_size: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        debug_assert!(self.voxel_type_ids.is_none());

        // Index of the lower corner of the 2x2x2 area in the lower LOD data we want to look at
        let mut visible_count = 0;
        let mut count = 0;

        for idx in voxels_in_lower_lod(
            voxel_index,
            curr_lvl,
            curr_lod,
            lower_lvl,
            lower_lod,
            n_lods,
            chunk_size
        ) {
            count += 1;

            if bitmask.get(idx) {
                visible_count += 1;
            }
        }

        self.bitmask.data.set_block(voxel_index, visible_count as f32 >= lod_block_fill_thresh * count as f32);
    }

    pub fn calc_from_lower_lod_voxels(
        &mut self,
        voxels: &ChunkVoxelIDs,
        curr_lvl: usize,
        curr_lod: usize,
        lower_lod: usize,
        lower_lvl: usize,
        chunk_size: usize,
        n_chunk_lvls: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        debug_assert!(self.voxel_type_ids.is_some());

        for vox_index in 0..cubed((chunk_size*(n_chunk_lvls - curr_lvl)).to_le() >> curr_lod) {
            // Index of the lower corner of the 2x2x2 area in the lower LOD data we want to look at
            let mut visible_count = 0;
            let mut count = 0;
            let mut type_counts = HashMap::<u8, u32>::new();

            for idx in voxels_in_lower_lod(
                vox_index,
                curr_lvl,
                curr_lod,
                lower_lvl,
                lower_lod,
                n_lods,
                chunk_size
            ) {
                count += 1;

                let id = voxels[idx];
                let vox_type = VE::from_u8(id).unwrap();
                if vox_type.def().is_visible {
                    visible_count += 1;
                    match type_counts.get_mut(&id) {
                        None => { type_counts.insert(id, 1); },
                        Some(c) => { *c += 1; },
                    }
                }
            }

            if visible_count as f32 >= lod_block_fill_thresh * count as f32 {
                self.voxel_type_ids.as_mut().unwrap().data[vox_index] = type_counts.into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1)).map(|(k, _)| k).unwrap();
            }
            else {
                self.voxel_type_ids.as_mut().unwrap().data[vox_index] = 0;
            }
        }

        calc_full_bitmask::<VE>(&self.voxel_type_ids.as_mut().unwrap().data, &mut self.bitmask.data);
    }
}