use std::sync::Arc;
use cgmath::{Array, EuclideanSpace, Vector3};
use getset::Getters;
use super::lod::{VoxelLODChunkCapsule, VoxelLODChunkEditor, VoxelLODCreateParams, VoxelMemoryGridLOD};
use crate::renderer::component::voxels::VoxelData;
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor, ChunkEditor, NewMemoryGridEditor};
use crate::world::{TLCPos, TLCVector, VoxelPos};
use hashbrown::{HashMap};
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::{ChunkLoadQueueItem, ChunkLoadQueueItemData};
use crate::world::mem_grid::utils::{cubed, index_for_pos_in_tlc, IteratorWithIndexing};
use crate::world::mem_grid::voxel::gpu_defs::ChunkVoxels;


#[derive(Clone, Debug, Getters)]
pub struct VoxelMemoryGrid {
    lods: Vec<Vec<Option<VoxelMemoryGridLOD>>>,
    #[get = "pub"]
    metadata: VoxelMemoryGridMetadata,
}

#[derive(Clone, Debug, Getters)]
pub struct VoxelMemoryGridMetadata {
    #[get = "pub"]
    largest_lvl: usize,
    #[get = "pub"]
    largest_lod: usize,  // which (lvl, LOD) has the largest grid size
    #[get = "pub"]
    chunk_size: usize,
    #[get = "pub"]
    n_lvls: usize,
    #[get = "pub"]
    n_lods: usize,
    #[get = "pub"]
    lod_block_fill_thresh: f32,
}

#[derive(Clone, Debug)]
pub struct VoxelChunkLoadQueueItemData {
    pub lods: Vec<Vec<bool>>,
}
impl ChunkLoadQueueItemData for VoxelChunkLoadQueueItemData {}


fn lod_tlc_size(chunk_size: usize, n_lvls: usize, lvl: usize, lod: usize) -> usize {
    chunk_size.pow((n_lvls - lvl) as u32).to_le() >> lod
}

impl VoxelMemoryGrid {
    pub fn new(
        lod_params: Vec<Vec<Option<VoxelLODCreateParams>>>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        chunk_size: usize,
        start_tlc: TLCPos<i64>,
        lod_block_fill_thresh: f32,
    ) -> (Self, VoxelData) {
        let (largest_lvl, largest_lod, size) = lod_params
            .iter().enumerate()
            .filter_map(|(lvl, lvl_lods)|
                lvl_lods.iter().enumerate().filter_map(|(lod, lod_data)|
                    lod_data.as_ref().map(|d| (lvl, lod, d.size))
                ).max_by_key(|x| x.2)
            ).max_by_key(|x| x.2)
            .unwrap();
        let n_lvls = lod_params.len() - 1;
        let n_lods = lod_params.first().unwrap().len();

        let (grid_lods, lods) = lod_params
            .into_iter()
            .enumerate()
            .map(|(lvl, lvl_sizes)| {
                lvl_sizes
                    .into_iter()
                    .enumerate()
                    .map(|(lod, params_o)| {
                        match params_o {
                            None => (None, None),
                            Some(params) => {
                                let lod_start_tlc = TLCPos(start_tlc.0 + Vector3::from_value(((size - params.size)/2) as i64));
                                let (a, b) = VoxelMemoryGridLOD::new(
                                    params,
                                    lod_start_tlc,
                                    lod_tlc_size(chunk_size, n_lvls, lvl, lod),
                                    Arc::clone(&memory_allocator),
                                );
                                (Some(a), Some(b))
                            }
                        }
                    }).unzip()
            })
            .unzip();

        (
            VoxelMemoryGrid {
                lods: grid_lods,
                metadata: VoxelMemoryGridMetadata {
                    largest_lvl,
                    largest_lod,
                    chunk_size,
                    n_lvls,
                    n_lods,
                    lod_block_fill_thresh,
                }
            },
            VoxelData::new(lods),
        )
    }

    pub fn get_updates(&mut self) -> Vec<Vec<Option<Vec<VoxelLODUpdate>>>> {
        self.lods
            .iter_mut()
            .map(|lvl_lods|
                lvl_lods.iter_mut().map(|lod_o|
                    lod_o.as_mut().map(|lod| lod.aggregate_updates(true))
                ).collect()
            ).collect()
    }

    fn apply_and_queue<F: FnMut(&mut VoxelMemoryGridLOD) -> Vec<ChunkLoadQueueItem<()>>>(
        &mut self,
        mut to_apply: F
    ) -> Vec<ChunkLoadQueueItem<VoxelChunkLoadQueueItemData>> {
        let mut chunks = HashMap::new();
        let (n_lods, n_lvls) = (self.metadata.n_lods, self.metadata.n_lvls);

        for (lvl, lvl_lods) in self.lods.iter_mut().enumerate() {
            for (lod, lod_o) in lvl_lods.iter_mut().enumerate() {
                match lod_o {
                    None => {},
                    Some(lod_data) => {
                        for item in to_apply(lod_data) {
                            let e = chunks.entry(item.pos.0).or_insert(
                                ChunkLoadQueueItem {
                                    pos: item.pos,
                                    data: VoxelChunkLoadQueueItemData {
                                        lods: vec![vec![false; n_lods]; n_lvls+1],
                                    }
                                }
                            );
                            e.data.lods[lvl][lod] = true;
                        }
                    }
                }
            }
        }

        chunks.into_values().collect()
    }

    pub fn largest_lod(&self) -> &VoxelMemoryGridLOD {
        self.lods[self.metadata.largest_lvl][self.metadata.largest_lod].as_ref().unwrap()
    }

    pub fn tlc_size(&self) -> usize {
        self.metadata.chunk_size.pow(self.metadata.largest_lvl as u32) << self.metadata.largest_lod
    }
}

impl MemoryGrid for VoxelMemoryGrid {
    type ChunkLoadQueueItemData = VoxelChunkLoadQueueItemData;

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        self.apply_and_queue(|lod| lod.queue_load_all())
    }

    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
       self.apply_and_queue(|lod| lod.shift(shift, load_in_from_edge, load_buffer))
    }

    fn size(&self) -> usize {self.largest_lod().size() }
    fn start_tlc(&self) -> TLCPos<i64> { self.largest_lod().start_tlc() }
}

// impl<'a, VE: VoxelTypeEnum> EditMemoryGrid<ChunkVoxelEditor<'a, VE>, &'a VoxelMemoryGridMetadata> for VoxelMemoryGrid {
//     fn edit_for_size(&mut self, grid_size: usize) -> MemoryGridEditor<ChunkVoxelEditor<'a, VE>, &'a VoxelMemoryGridMetadata> {
//         let size = self.size();
//         let start_tlc = self.start_tlc();
//         let chunk_iters = self.lods.iter_mut().map(|lvl_lods|
//             lvl_lods.iter_mut().map(|lod_o|
//                 lod_o.as_mut().map(|lod| lod.edit_for_size(grid_size).chunks.into_iter())
//             ).collect()
//         ).collect();
//
//         MemoryGridEditor {
//             // lifetime: PhantomData,
//             chunks: LODSplitter(chunk_iters).map(|chunk_lods|
//                 Some(
//                     ChunkVoxelEditor {
//                         lods: chunk_lods,
//                     }
//                 )
//             ).collect(),
//             size,
//             start_tlc,
//             metadata: &self.metadata,
//         }
//     }
// }

impl<'a, VE: VoxelTypeEnum> NewMemoryGridEditor<'a, VoxelMemoryGrid> for MemoryGridEditor<ChunkVoxelEditor<'a, VE>, VoxelMemoryGridMetadata> {
    fn for_grid_with_size(mem_grid: &'a mut VoxelMemoryGrid, grid_size: usize) -> Self {
        let start_tlc = mem_grid.start_tlc();
        let chunk_iters: Vec<Vec<_>> = mem_grid.lods.iter_mut().map(|lvl_lods|
            lvl_lods.iter_mut().map(|lod_o|
                lod_o.as_mut().map(|lod| MemoryGridEditor::for_grid_with_size(lod, grid_size).chunks.into_iter())
            ).collect()
        ).collect();

        MemoryGridEditor {
            chunks: LODSplitter(chunk_iters).map(|chunk_lods|
                Some(
                    ChunkVoxelEditor {
                        lods: chunk_lods.into_iter().map(|lvl|
                            lvl.into_iter().map(|lod| lod.flatten()).collect()
                        ).collect(),
                    }
                )
            ).collect(),
            size: grid_size,
            start_tlc,
            metadata: mem_grid.metadata.clone(),
        }
    }
}


struct LODSplitter<I: Iterator>(Vec<Vec<Option<I>>>);

impl<I: Iterator> Iterator for LODSplitter<I> {
    type Item = Vec<Vec<Option<I::Item>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_items: Vec<Vec<_>> = self.0
            .iter_mut()
            .map(|iters| iters.iter_mut().map(|iter| iter.as_mut().map(Iterator::next)).collect()).collect();

        // If any of the iterators are Some (meaning the LOD exists) but the item in it is None, then stop iterating
        if next_items.iter().any(|iters| iters.iter().any(|iter|
            matches!(iter, Some(None))
        )) {
            // If one iterator is depleted, make sure all others are too
            debug_assert!(
                next_items.iter().any(|iters| iters.iter().any(|iter|
                    matches!(iter, Some(None)) || iter.is_none()
                )),
                "LOD chunk iterators passed to LODSplitter have different numbers of chunks."
            );
            return None;
        }

        Some(
            next_items.into_iter().map(|items|
                items.into_iter().map(|item| item.map(|i| i.unwrap())).collect()
            ).collect()
        )
    }
}


#[derive(Debug)]
pub struct ChunkVoxelEditor<'a, VE: VoxelTypeEnum> {
    lods: Vec<Vec<Option<VoxelLODChunkEditor<'a, VE>>>>,
}

#[derive(Debug, Clone)]
pub struct ChunkVoxelCapsule {
    lods: Vec<Vec<Option<VoxelLODChunkCapsule>>>
}


impl<'a, VE: VoxelTypeEnum> ChunkEditor<'a> for ChunkVoxelEditor<'a, VE> {
    type Capsule = ChunkVoxelCapsule;

    fn on_queued_for_loading(&mut self) {
        for lvl in self.lods.iter_mut() {
            for lod in lvl.iter_mut().flatten() {
                lod.on_queued_for_loading();
            }
        }
    }

    fn new_from_capsule(capsule: &'a mut Self::Capsule) -> Self {
        ChunkVoxelEditor {
            lods: capsule.lods.iter_mut().map(|lvl|
                lvl.iter_mut().map(|lod|
                    lod.as_mut().map(VoxelLODChunkEditor::new_from_capsule)
                ).collect()
            ).collect()
        }
    }

    fn replace_with_placeholder(&mut self) -> ChunkVoxelCapsule {
        ChunkVoxelCapsule {
            lods: self.lods.iter_mut().map(|lvl|
                lvl.iter_mut().map(|lod_o|
                    lod_o.as_mut().map(|lod| lod.replace_with_placeholder())
                ).collect()
            ).collect()
        }
    }

    fn replace_with_capsule(&mut self, capsule: Self::Capsule) {
        for (dest_lvl, lvl) in self.lods.iter_mut().zip(capsule.lods.into_iter()) {
            for (dest_lod, lod) in dest_lvl.iter_mut().zip(lvl.into_iter()) {
                match (dest_lod.as_mut(), lod) {
                    (None, None) => {},
                    (Some(dest_l), Some(l)) => {
                        dest_l.replace_with_capsule(l);
                    }
                    _ => { panic!("Internal error: chunks not aligned") }
                }
            }
        }
    }

    fn ok_to_replace_with_placeholder(&self) -> bool {
        self.lods.iter().all(|lvl| lvl.iter()
            .all(|lod| lod.is_none() || lod.as_ref().unwrap().ok_to_replace_with_placeholder()))
    }

    fn ok_to_replace_with_capsule(&self) -> bool {
        self.lods.iter().all(|lvl| lvl.iter()
            .all(|lod| lod.is_none() || lod.as_ref().unwrap().ok_to_replace_with_capsule()))
    }
}

impl ChunkVoxelCapsule {
    pub fn set_loaded(&mut self) {
        for lod in self.lods.iter_mut().flatten() {
            if let Some(l) = lod.as_mut() {
                l.set_loaded();
            }
        }
    }
}

pub enum SetVoxelErr {
    LODDoesNotExist,
    LODNotLoaded,
    LODVoxelsNotLoaded,
}

impl<'a, VE: VoxelTypeEnum> ChunkVoxelEditor<'a, VE> {
    pub fn load_new<F: Fn(TLCPos<i64>, usize, usize, &mut ChunkVoxels)>(
        &'a mut self,
        pos: TLCPos<i64>,
        lods_to_load: Vec<Vec<bool>>,
        gen_func: F,
        chunk_size: usize,
        n_chunk_lvls: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        println!("\npos {:?}", pos);
        // Last lvl/LOD that contained voxel ID info
        let (mut last_vox_lvl, mut last_vox_lod) = (None, None);
        // Last lvl/LOD that contained bitmask info
        let (mut last_bitmask_lvl, mut last_bitmask_lod) = (None, None);

        let lods: Vec<_> = self.lods.iter_mut().flatten().collect();

        let len = lods.len();
        IteratorWithIndexing::new(lods, len)
            .apply(|i, lod, lods_to_index| {
                let lvl_i = i / n_lods;
                let lod_i = i % n_lods;
                let load = lods_to_load[lvl_i][lod_i];

                if let Some(lod_data) = lod {
                    if load {
                        debug_assert!(!lod_data.bitmask().loaded, "Trying to load into already loaded chunk {:?} ({}, {})", pos, lvl_i, lod_i);  // Should have been marked not loaded before loading
                        lod_data.set_loaded();

                        // Need to load the info in this chunk
                        if lod_data.has_voxel_ids() {
                            if let (Some(l_lvl), Some(l_lod)) = (last_vox_lvl, last_vox_lod) {
                                // Load voxels based on higher fidelity LOD that is already loaded
                                (*lod_data).calc_from_lower_lod_voxels(
                                    lods_to_index[l_lvl*n_lods + l_lod].as_ref().unwrap().voxels().as_ref().unwrap(),
                                    lvl_i,
                                    lod_i,
                                    l_lvl,
                                    l_lod,
                                    chunk_size,
                                    n_chunk_lvls,
                                    n_lods,
                                    lod_block_fill_thresh,
                                );
                            }
                            else {
                                // Generate voxels
                                gen_func(
                                    pos,
                                    lvl_i,
                                    lod_i,
                                    lod_data.overwrite().voxels
                                );
                            }

                            last_vox_lvl = Some(lvl_i);
                            last_vox_lod = Some(lod_i);
                        }
                        else {
                            // If this chunk only has a bitmask, update from previous LOD bitmask
                            lod_data.update_bitmask_from_lower_lod(
                                lods_to_index[last_bitmask_lvl.unwrap()*n_lods + last_bitmask_lod.unwrap()].as_ref().unwrap().bitmask(),
                                lvl_i,
                                lod_i,
                                last_bitmask_lvl.unwrap(),
                                last_bitmask_lod.unwrap(),
                                chunk_size,
                                n_chunk_lvls,
                                n_lods,
                                lod_block_fill_thresh,
                            )
                        }

                        last_bitmask_lvl = Some(lvl_i);
                        last_bitmask_lod = Some(lod_i);
                    }
                    else if lod_data.bitmask().loaded {
                        last_bitmask_lod = Some(lod_i);
                        last_bitmask_lvl = Some(lvl_i);

                        // Note: voxels should only ever be loaded when the bitmask is
                        if let Some(true) = lod_data.voxels().as_ref().map(|v| v.loaded) {
                            last_vox_lvl = Some(lvl_i);
                            last_vox_lod = Some(lod_i);
                        }
                    }
                }
            });
    }

    pub fn set_voxel(&mut self, index_in_tlc: usize, voxel_typ: VE, meta: &VoxelMemoryGridMetadata) {
        for lvl in 0..meta.n_lvls {
            for lod in 0..meta.n_lods {
                if let Some(data) = self.lods[lvl][lod].as_mut() {
                    if let Some(voxels) = data.voxels().as_ref() {
                        if voxels.loaded {
                            let block_size = meta.chunk_size.pow(lvl as u32) *2usize.pow(lod as u32);
                            data.set_voxel(index_in_tlc / cubed(block_size), voxel_typ);
                        }
                    }
                }
            }
        }
    }

    pub fn set_voxel_pos(&mut self, pos: VoxelPos<u32>, voxel_typ: VE, meta: &VoxelMemoryGridMetadata) {
        self.set_voxel(
            index_for_pos_in_tlc(pos.0, meta.chunk_size, meta.n_lvls, 0, 0),
            voxel_typ,
            meta,
        );
    }
}


#[derive(Clone, Copy)]
pub struct GlobalVoxelPos {
    pub tlc: TLCPos<i64>,
    pub voxel_index: usize,
}
impl GlobalVoxelPos {
    pub fn new(
        global_pos: VoxelPos<i64>,
        chunk_size: usize,
        n_chunk_lvls: usize,
    ) -> Self {
        let tlc_size = chunk_size.pow(n_chunk_lvls as u32);
        let global_tlc = global_pos.0 / tlc_size as i64;
        let pos_in_tlc = (global_pos.0 - (global_tlc * tlc_size as i64).to_vec()).cast::<u32>().unwrap();

        GlobalVoxelPos {
            tlc: TLCPos(global_tlc),
            voxel_index: index_for_pos_in_tlc(pos_in_tlc, chunk_size, n_chunk_lvls, 0, 0),
        }
    }
}