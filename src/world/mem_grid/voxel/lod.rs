use std::marker::PhantomData;
use std::mem;
use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::component::voxels::lod::RendererVoxelLOD;
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::world::mem_grid::layer::{MemoryGridLayer};
use crate::world::mem_grid::utils::cubed;
use crate::world::mem_grid::voxel::gpu_defs::{ChunkBitmask, ChunkVoxels};
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor, ChunkEditor, NewMemoryGridEditor};
use crate::world::{TLCPos, TLCVector};
use std::sync::Arc;
use getset::Getters;
use hashbrown::HashMap;
use vulkano::command_buffer::BufferCopy;
use vulkano::DeviceSize;
use vulkano::memory::allocator::MemoryAllocator;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::ChunkLoadQueueItem;


#[derive(Clone, Debug)]
pub struct VoxelLODCreateParams {
    pub size: usize,
    pub bitmask_binding: u32,
    pub voxel_type_ids_binding: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct VoxelMemoryGridLOD {
    bitmask_layer: MemoryGridLayer<ChunkBitmask>,
    voxel_layer: Option<MemoryGridLayer<ChunkVoxels>>,
    updated_bitmask_regions_layer: MemoryGridLayer<Vec<BufferCopy>>,
    voxels_per_tlc: usize,
}

impl VoxelMemoryGridLOD {
    pub fn new(
        params: VoxelLODCreateParams,
        voxels_per_tlc: usize,
        start_tlc: TLCPos<i64>,
        tlc_size: usize,
        buffer_allocator: Arc<dyn MemoryAllocator>,
    ) -> (Self, RendererVoxelLOD) {
        let bitmask = vec![ChunkBitmask::new_blank(voxels_per_tlc); cubed(params.size)] ;
        let voxels = params.voxel_type_ids_binding.map(|_| vec![ChunkVoxels::new_blank(voxels_per_tlc); cubed(params.size)]);
        let lod = RendererVoxelLOD::new(
            bitmask.iter().map(|c| &c.bitmask).flatten().copied().collect::<Vec<_>>().into_iter(),  // ENHANCEMENT: Do this better (and below)
            voxels.as_ref().map(|voxs| voxs.iter().map(|c| &c.ids).flatten().copied().collect::<Vec<_>>().into_iter()),
            params.bitmask_binding,
            params.voxel_type_ids_binding,
            buffer_allocator,
        );

        (
            VoxelMemoryGridLOD {
                // voxel_type_enum: PhantomData,
                bitmask_layer: MemoryGridLayer::new(
                    bitmask,
                    start_tlc,
                    params.size,
                    tlc_size,
                ),
                voxel_layer: voxels.map(|voxs|
                    MemoryGridLayer::new(
                        voxs,
                        start_tlc,
                        params.size,
                        tlc_size,
                    )
                ),
                updated_bitmask_regions_layer: MemoryGridLayer::new(
                    vec![vec![]; cubed(params.size)],
                    start_tlc,
                    params.size,
                    tlc_size,
                ),
                voxels_per_tlc: cubed(tlc_size),
            },
            lod,
        )
    }

    /// Aggregate the values from updated_bitmask_regions_layer into absolute regions in the
    /// bitmask buffer, generate regions for updating the voxel type ID buffer, and reset these
    /// tracked regions if clear_regions=true.
    pub fn aggregate_updates(&mut self, clear_regions: bool) -> Vec<VoxelLODUpdate> {
        let mut updates = vec![];
        let has_voxel_ids = self.voxel_layer.is_some();
        let voxels_per_tlc = self.voxels_per_tlc;
        let n_chunks = cubed(self.size());

        for (chunk_i, ((regions, bitmask), voxel_ids)) in
        self.updated_bitmask_regions_layer
            .chunks_mut()
            .iter_mut()
            .zip(self.bitmask_layer.chunks().iter())
            .zip(match self.voxel_layer {
                None => Box::new((0..n_chunks).map(|_| None)),
                Some(ref layer) =>
                    Box::new(layer.chunks().iter().map(Some)) as Box<dyn Iterator<Item = Option<&ChunkVoxels>>>,
            })
            .enumerate()
        {
            let mut bitmask_updated_regions = vec![];
            let mut voxels_updated_regions = match has_voxel_ids {
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

            match &mut voxels_updated_regions {
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
                    bitmask: &bitmask.bitmask,
                    voxel_type_ids: voxel_ids.map(|x| &x.ids),
                    bitmask_updated_regions,
                    voxel_id_updated_regions: voxels_updated_regions,
                }
            )
        }

        updates
    }
}

impl MemoryGrid for VoxelMemoryGridLOD {
    type ChunkLoadQueueItemData = ();

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<()>> {
        // Because the queues for all three of the layers will be the same size, only need to get one.
        self.bitmask_layer.queue_load_all()
    }

    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<()>> {
        // Because all three of these queues will be the same size, only need to return one.
        self.voxel_layer.as_mut().map(
            |layer| layer.shift(shift, load_in_from_edge, load_buffer)
        );
        self.updated_bitmask_regions_layer.shift(shift, load_in_from_edge, load_buffer);
        self.bitmask_layer.shift(shift, load_in_from_edge, load_buffer)
    }

    fn size(&self) -> usize { self.bitmask_layer.size() }
    fn start_tlc(&self) -> TLCPos<i64> { self.bitmask_layer.start_tlc() }
}

// impl<'a, VE: VoxelTypeEnum> EditMemoryGrid<VoxelLODChunkEditor<'a, VE>, ()> for VoxelMemoryGridLOD {
//     fn edit_for_size(&mut self, grid_size: usize) -> MemoryGridEditor<VoxelLODChunkEditor<'a, VE>, ()> {
//         // ENHANCEMENT: Make this call .to_virtual_for_size(self.size()) on children instead of
//         // use grid_size and then add the necessary padding in this function.
//
//         let bitmask_editor = self.bitmask_layer
//             .edit_for_size(grid_size);
//
//         let voxel_editor: Option<MemoryGridEditor<&mut ChunkVoxels, ()>> =
//             self.voxel_layer.as_mut().map(|l| l.edit_for_size(grid_size));
//
//         let regions_editor = self.updated_bitmask_regions_layer
//             .edit_for_size(grid_size);
//
//         let size = bitmask_editor.size;
//         let start_tlc = bitmask_editor.start_tlc;
//
//         MemoryGridEditor {
//             // lifetime: PhantomData,
//             chunks: bitmask_editor.chunks.into_iter()
//                 .zip(
//                     match voxel_editor {
//                         None => (0..cubed(size)).map(|_| None).collect::<Vec<_>>(),
//                         Some(l) => l.chunks.into_iter().collect(),
//                     }
//                 ).zip(regions_editor.chunks)
//                 .map(|((bm_o, vid), regions)|
//                     bm_o.map(|bm|
//                         VoxelLODChunkEditor {
//                             voxel_type_enum: PhantomData,
//                             voxels: vid,
//                             bitmask: bm,
//                             updated_bitmask_regions: regions.unwrap(),
//                         }
//                     )
//                 ).collect(),
//             size,
//             start_tlc,
//             metadata: (),
//         }
//     }
// }

impl<'a, VE: VoxelTypeEnum> NewMemoryGridEditor<'a, VoxelMemoryGridLOD> for MemoryGridEditor<VoxelLODChunkEditor<'a, VE>, ()> {
    fn for_grid_with_size(mem_grid: &'a mut VoxelMemoryGridLOD, grid_size: usize) -> Self {
        // ENHANCEMENT: Make this call .to_virtual_for_size(self.size()) on children instead of
        // use grid_size and then add the necessary padding in this function.

        let bitmask_editor = MemoryGridEditor::for_grid_with_size(&mut mem_grid.bitmask_layer, grid_size);

        let voxel_editor: Option<MemoryGridEditor<&mut ChunkVoxels, ()>> =
            mem_grid.voxel_layer.as_mut().map(|l| MemoryGridEditor::for_grid_with_size(l, grid_size));

        let regions_editor = MemoryGridEditor::for_grid_with_size(&mut mem_grid.updated_bitmask_regions_layer, grid_size);

        let size = bitmask_editor.size;
        let start_tlc = bitmask_editor.start_tlc;

        MemoryGridEditor {
            // lifetime: PhantomData,
            chunks: bitmask_editor.chunks.into_iter()
                .zip(
                    match voxel_editor {
                        None => (0..cubed(size)).map(|_| None).collect::<Vec<_>>(),
                        Some(l) => l.chunks.into_iter().collect(),
                    }
                ).zip(regions_editor.chunks)
                .map(|((bm_o, vid), regions)|
                    bm_o.map(|bm|
                        VoxelLODChunkEditor {
                            voxel_type_enum: PhantomData,
                            voxels: vid,
                            bitmask: bm,
                            updated_bitmask_regions: regions.unwrap(),
                        }
                    )
                ).collect(),
            size,
            start_tlc,
            metadata: (),
        }
    }
}


#[derive(Debug, Getters)]
pub struct VoxelLODChunkEditor<'a, VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    #[get="pub"]
    voxels: Option<&'a mut ChunkVoxels>,
    #[get="pub"]
    bitmask: &'a mut ChunkBitmask,
    updated_bitmask_regions: &'a mut Vec<BufferCopy>,
}

#[derive(Debug, Clone)]
pub struct VoxelLODChunkCapsule {
    voxels: Option<ChunkVoxels>,
    bitmask: ChunkBitmask,
    updated_bitmask_regions: Vec<BufferCopy>
}


impl<'a, VE: VoxelTypeEnum> ChunkEditor<'a> for VoxelLODChunkEditor<'a, VE> {
    type Capsule = VoxelLODChunkCapsule;

    fn new_from_capsule(capsule: &'a mut Self::Capsule) -> Self {
        VoxelLODChunkEditor {
            voxel_type_enum: PhantomData,
            voxels: capsule.voxels.as_mut(),
            bitmask: &mut capsule.bitmask,
            updated_bitmask_regions: &mut capsule.updated_bitmask_regions,
        }
    }

    fn replace_with_placeholder(&mut self) -> VoxelLODChunkCapsule {
        VoxelLODChunkCapsule {
            voxels: self.voxels.as_mut().map(|v| v.replace_with_placeholder()),
            bitmask: self.bitmask.replace_with_placeholder(),
            updated_bitmask_regions: mem::take(self.updated_bitmask_regions),
        }
    }

    fn replace_with_capsule(&mut self, capsule: Self::Capsule) {
        self.voxels.as_mut().map(|v| **v = capsule.voxels.unwrap());
        *self.bitmask = capsule.bitmask;
        *self.updated_bitmask_regions = capsule.updated_bitmask_regions;
    }
}

impl <'a, VE: VoxelTypeEnum> VoxelLODChunkEditor<'a, VE> {
    pub fn has_voxel_ids(&self) -> bool { self.voxels.is_some() }

    /// Overwrite voxel data. This will allow editing of the voxel IDs directly and automatically
    /// recalculate the full bitmask when VoxelLODChunkEditor is dropped. Please don't resize the
    /// voxel ID vec.
    pub fn overwrite(&mut self) -> VoxelEditor<VE> {
        debug_assert!(self.voxels.is_some());

        VoxelEditor {
            voxel_type_enum: PhantomData::<VE>,
            voxels: self.voxels.as_mut().unwrap(),
            bitmask: self.bitmask,
            updated_bitmask_regions: self.updated_bitmask_regions,
        }
    }

    /// Set a single voxel
    pub fn set_voxel(&mut self, index: usize, voxel_typ: VE) {
        debug_assert!(self.voxels.is_some());

        self.voxels.as_mut().unwrap()[index] = voxel_typ.to_u8().unwrap();
        self.bitmask.set_block(index, voxel_typ.def().is_visible);
        self.updated_bitmask_regions.push(BufferCopy {
            src_offset: (index / 8) as DeviceSize,
            dst_offset: (index / 8) as DeviceSize,
            size: 1,
            ..Default::default()
        })
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
        debug_assert!(self.voxels.is_none());

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
        debug_assert!(self.voxels.is_none());

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

        self.bitmask.set_block(voxel_index, visible_count as f32 >= lod_block_fill_thresh * count as f32);
        self.updated_bitmask_regions.push(BufferCopy {
            src_offset: (voxel_index / 8) as DeviceSize,
            dst_offset: (voxel_index / 8) as DeviceSize,
            size: 1,
            ..Default::default()
        })
    }

    pub fn calc_from_lower_lod_voxels(
        &mut self,
        voxels: &ChunkVoxels,
        curr_lvl: usize,
        curr_lod: usize,
        lower_lod: usize,
        lower_lvl: usize,
        chunk_size: usize,
        n_chunk_lvls: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        debug_assert!(self.voxels.is_some());

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
                self.voxels.as_mut().unwrap()[vox_index] = type_counts.into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1)).map(|(k, _)| k).unwrap();
            }
            else {
                self.voxels.as_mut().unwrap()[vox_index] = 0;
            }
        }

        calc_full_bitmask::<VE>(self.voxels.as_mut().unwrap(), self.bitmask);
    }
}


pub fn calc_full_bitmask<VE: VoxelTypeEnum>(voxels: &ChunkVoxels, bitmask: &mut ChunkBitmask) {
    for i in 0..voxels.n_voxels() {
        if VE::from_u8(voxels[i]).unwrap().def().is_visible {
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
pub struct VoxelEditor<'a, VE: VoxelTypeEnum> {
    voxel_type_enum: PhantomData<VE>,
    pub voxels: &'a mut ChunkVoxels,
    bitmask: &'a mut ChunkBitmask,
    updated_bitmask_regions: &'a mut Vec<BufferCopy>,
}
impl<'a, VE: VoxelTypeEnum> Drop for VoxelEditor<'a, VE> {
    fn drop(&mut self) {
        calc_full_bitmask::<VE>(self.voxels, self.bitmask);
        self.updated_bitmask_regions.push(BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: ((self.bitmask.n_voxels() + 7) / 8) as DeviceSize,
            ..Default::default()
        })
    }
}