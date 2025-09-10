use std::sync::Arc;
use cgmath::{Array, EuclideanSpace, Vector3};
use super::lod::{VirtualVoxelLOD, VirtualVoxelLODMetadata, VoxelLOD, VoxelLODChunkData, VoxelLODCreateParams};
use crate::renderer::component::voxels::VoxelData;
use crate::world::mem_grid::{FromVirtual, MemoryGridMetadata, PhysicalMemoryGrid, PhysicalMemoryGridStruct, Placeholder, ToVirtual, VirtualMemoryGridStruct};
use crate::world::{TLCPos, TLCVector, VoxelPos};
use derive_more::Deref;
use hashbrown::{HashMap};
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::{ChunkLoadQueueItem};
use crate::world::mem_grid::utils::{cubed, index_for_pos_in_tlc};
use crate::world::mem_grid::voxel::ChunkVoxelIDs;


#[derive(Deref)]
pub struct VoxelMemoryGrid<VE: VoxelTypeEnum>(PhysicalMemoryGridStruct<VoxelMemoryGridData<VE>, VoxelMemoryGridMetadata>);
#[derive(Deref)]
pub struct VirtualVoxelMemoryGrid<VE: VoxelTypeEnum>(VirtualMemoryGridStruct<VoxelMemoryGridChunkData<VE>, VirtualVoxelMemoryGridMetadata>);

pub struct VoxelMemoryGridData<VE: VoxelTypeEnum> {
    lods: Vec<Vec<Option<VoxelLOD<VE>>>>
}

#[derive(Clone)]
pub struct VoxelMemoryGridMetadata {
    size: usize,
    chunk_size: usize,
    start_tlc: TLCPos<i64>,
    n_lvls: usize,
    n_lods: usize,
}
impl MemoryGridMetadata for VoxelMemoryGridMetadata {
    fn size(&self) -> usize { self.size }
    fn start_tlc(&self) -> TLCPos<i64> { self.start_tlc }
}

#[derive(Clone)]
pub struct VirtualVoxelMemoryGridMetadata {
    this: VoxelMemoryGridMetadata,
    lods: Vec<Vec<Option<VirtualVoxelLODMetadata>>>
}
impl MemoryGridMetadata for VirtualVoxelMemoryGridMetadata {
    fn size(&self) -> usize { self.this.size }
    fn start_tlc(&self) -> TLCPos<i64> { self.this.start_tlc }
}

#[derive(Clone)]
pub struct VoxelMemoryGridChunkData<VE: VoxelTypeEnum> {
    pub lods: Vec<Vec<Option<VoxelLODChunkData<VE>>>>,
}

#[derive(Clone)]
pub struct VoxelChunkLoadingQueueItemData {
    pub lods: Vec<Vec<bool>>,
}


impl<VE: VoxelTypeEnum> VoxelMemoryGrid<VE> {
    fn voxels_per_tlc(chunk_size: usize, n_lvls: usize, lvl: usize, lod: usize) -> usize {
        chunk_size.pow((n_lvls - lvl) as u32).to_le() >> (lod * 3)
    }

    pub fn new(
        lod_params: Vec<Vec<Option<VoxelLODCreateParams>>>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        chunk_size: usize,
        start_tlc: TLCPos<i64>,
        tlc_size: usize,
    ) -> (Self, VoxelData) {
        let size = lod_params
            .iter()
            .flatten()
            .filter_map(|p| Some(p.as_ref()?.size))
            .max()
            .unwrap();
        let n_lvls = lod_params.len();
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
                                let voxels_per_tlc = Self::voxels_per_tlc(chunk_size, n_lvls, lvl, lod);
                                let lod_start_tlc = TLCPos(start_tlc.0 + Vector3::from_value(((size - params.size)/2) as i64));
                                let (a, b) = VoxelLOD::new(
                                    params,
                                    voxels_per_tlc,
                                    lod_start_tlc,
                                    tlc_size,
                                    Arc::clone(&memory_allocator),
                                );
                                (Some(a), Some(b))
                            }
                        }
                    }).unzip()
            })
            .unzip();

        (
            VoxelMemoryGrid(
                PhysicalMemoryGridStruct::new(
                    VoxelMemoryGridData { lods: grid_lods },
                    VoxelMemoryGridMetadata {
                        size,
                        chunk_size,
                        start_tlc,
                        n_lvls,
                        n_lods,
                    },
                )
            ),
            VoxelData::new(lods),
        )
    }

    pub fn get_updates(&mut self) -> Vec<Vec<Option<Vec<VoxelLODUpdate>>>> {
        self.0.data.lods
            .iter_mut()
            .map(|lvl_lods|
                lvl_lods.iter_mut().map(|lod_o|
                    lod_o.as_mut().map(|lod| lod.aggregate_updates(true))
                ).collect()
            ).collect()
    }

    fn apply_and_queue<F: FnMut(&mut VoxelLOD<VE>) -> Vec<ChunkLoadQueueItem<()>>>(
        &mut self,
        mut to_apply: F
    ) -> Vec<ChunkLoadQueueItem<VoxelChunkLoadingQueueItemData>> {
        let mut chunks = HashMap::new();
        let (n_lods, n_lvls) = (self.metadata.n_lods, self.metadata.n_lvls);

        for (lvl, lvl_lods) in self.0.data.lods.iter_mut().enumerate() {
            for (lod, lod_o) in lvl_lods.iter_mut().enumerate() {
                match lod_o {
                    None => {},
                    Some(lod_data) => {
                        for item in to_apply(lod_data) {
                            let e = chunks.entry(item.pos.0).or_insert(
                                ChunkLoadQueueItem {
                                    pos: item.pos,
                                    data: VoxelChunkLoadingQueueItemData {
                                        lods: vec![vec![false; n_lods]; n_lvls],
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
}
impl<VE: VoxelTypeEnum> PhysicalMemoryGrid for VoxelMemoryGrid<VE> {
    type Data = VoxelMemoryGridData<VE>;
    type Metadata = VoxelMemoryGridMetadata;
    type ChunkLoadQueueItemData = VoxelChunkLoadingQueueItemData;

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<VoxelChunkLoadingQueueItemData>> {
        self.apply_and_queue(|lod| lod.queue_load_all())
    }

    fn shift(
        &mut self,
        shift: TLCVector<i32>,
        load_in_from_edge: TLCVector<i32>,
        load_buffer: [bool; 3]
    ) -> Vec<ChunkLoadQueueItem<VoxelChunkLoadingQueueItemData>> {
        let r =
            self.apply_and_queue(|lod| lod.shift(shift, load_in_from_edge, load_buffer));

        for lvl in self.data.lods.iter() {
            for lod_o in lvl {
                if let Some(lod) = lod_o {
                    if lod.size() == self.size() {
                        self.0.metadata.start_tlc = lod.start_tlc();
                        return r;
                    }
                }
            }
        }

        panic!();
    }
}
impl<VE: VoxelTypeEnum> ToVirtual<VoxelMemoryGridChunkData<VE>, VirtualVoxelMemoryGridMetadata> for VoxelMemoryGrid<VE> {
    fn to_virtual_for_size(self, grid_size: usize) -> VirtualMemoryGridStruct<VoxelMemoryGridChunkData<VE>, VirtualVoxelMemoryGridMetadata> {
        let (chunk_iters, metadata): (Vec<Vec<_>>, Vec<Vec<_>>) =
            self.0.data.lods.into_iter()
            .map(|lvl_lods| {
                lvl_lods.into_iter()
                    .map(|lod_o| match lod_o {
                        None => (None, None),
                        Some(lod) => {
                            let (a, b) =
                                lod.to_virtual_for_size(grid_size).deconstruct();
                            (Some(a.into_iter()), Some(b))
                        },
                    })
                    .unzip()
            })
            .unzip();

        VirtualMemoryGridStruct::new(
            LODSplitter(chunk_iters).map(
                |chunk_lods| Some(VoxelMemoryGridChunkData {
                    lods: chunk_lods,
                })
            ).collect(),
            VirtualVoxelMemoryGridMetadata {
                this: self.0.metadata,
                lods: metadata,
            }
        )
    }
}
impl<VE: VoxelTypeEnum> FromVirtual<VoxelMemoryGridChunkData<VE>, VirtualVoxelMemoryGridMetadata> for VoxelMemoryGrid<VE> {
    fn from_virtual_for_size(virtual_grid: VirtualMemoryGridStruct<VoxelMemoryGridChunkData<VE>, VirtualVoxelMemoryGridMetadata>, grid_size: usize) -> Self {
        let (data, metadata) = virtual_grid.deconstruct();

        let mut chunk_data: Vec<Vec<Option<Vec<Option<VoxelLODChunkData<VE>>>>>> = metadata.lods.iter().map(|lvl_meta|
            lvl_meta.iter().map(|meta|
                meta.as_ref().map(|_| vec![None; cubed(grid_size)])
            ).collect()
        ).collect();

        for (i, chunk_o) in data.into_iter().enumerate() {
            if let Some(chunk) = chunk_o {
                for (lvl, chunk_lvl) in chunk.lods.into_iter().enumerate() {
                    for (lod, chunk_lod) in chunk_lvl.into_iter().enumerate() {
                        if chunk_lod.is_some() {
                           chunk_data[lvl][lod].as_mut().unwrap()[i] = chunk_lod;
                        }
                    }
                }
            }
        }

        VoxelMemoryGrid(
            PhysicalMemoryGridStruct::new(
                VoxelMemoryGridData {
                    lods: chunk_data.into_iter().zip(metadata.lods)
                        .map(|(lvl_chunk_data, lvl_meta)|
                            lvl_chunk_data.into_iter().zip(lvl_meta)
                                .map(|(lod_chunk_data, meta)|
                                    match (lod_chunk_data, meta) {
                                        (None, None) => None,
                                        (Some(cd), Some(m)) => {
                                            Some(
                                                VoxelLOD::from_virtual_for_size(
                                                    VirtualVoxelLOD::new(
                                                        cd,
                                                        m,
                                                    ),
                                                    grid_size
                                                )
                                            )
                                        }
                                        _ => panic!(),
                                    }
                            ).collect()
                    ).collect()
                },
                metadata.this,
            )
        )
    }
}

struct LODSplitter<I: Iterator>(Vec<Vec<Option<I>>>);

impl<I: Iterator> Iterator for LODSplitter<I> {
    type Item = Vec<Vec<I::Item>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .iter_mut()
            .map(|iters| iters.iter_mut().map(|iter| iter.as_mut()?.next()).collect())
            .collect()
    }
}


impl<VE: VoxelTypeEnum> Placeholder for VoxelMemoryGridChunkData<VE> {
    fn placeholder(&self) -> Self {
        VoxelMemoryGridChunkData {
            lods: self.lods.iter().map(|lvl_lods|
                lvl_lods.iter().map(|lod_o|
                    lod_o.as_ref().map(|lod| lod.placeholder())
                ).collect()
            ).collect()
        }
    }
}
impl<VE: VoxelTypeEnum> VoxelMemoryGridChunkData<VE> {
    pub fn load_new<F: Fn(TLCPos<i64>, usize, usize, &mut ChunkVoxelIDs)>(
        &mut self,
        pos: TLCPos<i64>,
        lods_to_load: Vec<Vec<bool>>,
        gen_func: F,
        chunk_size: usize,
        n_chunk_lvls: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        // Last lvl/LOD that contained voxel ID info
        let (mut last_lvl, mut last_lod) = (None, None);
        let (mut last_bitmask_lvl, mut last_bitmask_lod) = (None, None);

        let mut visited_lods: Vec<Vec<&Option<VoxelLODChunkData<VE>>>> = vec![vec![]; n_chunk_lvls];

        for (lvl, (lvl_lods_to_load, lvl_data)) in lods_to_load.iter()
            .zip(self.lods.iter_mut()).enumerate() {
            for (lod, (load, lod_data_o)) in
                lvl_lods_to_load.iter().zip(lvl_data.iter_mut()).enumerate() {

                if let Some(ref mut lod_data) = lod_data_o {
                    if *load {
                        // Need to load the info in this chunk
                        match lod_data.voxel_type_ids {
                            // If this chunk only has a bitmask, update from previous LOD bitmask
                            None => lod_data.update_bitmask_from_lower_lod(
                                &((&visited_lods[last_bitmask_lvl.unwrap()] as &Vec<&Option<_>>)
                                    [last_bitmask_lod.unwrap()] as &Option<VoxelLODChunkData<VE>>)
                                    .as_ref().unwrap().bitmask.data,
                                lvl,
                                lod,
                                last_bitmask_lvl.unwrap(),
                                last_bitmask_lod.unwrap(),
                                chunk_size,
                                n_chunk_lvls,
                                n_lods,
                                lod_block_fill_thresh,
                            ),
                            Some(_) => {
                                if let (Some(l_lvl), Some(l_lod)) = (last_lvl, last_lod) {
                                    // Load voxels based on higher fidelity LOD that is already loaded
                                    lod_data.calc_from_lower_lod_voxels(
                                        &((&visited_lods[l_lvl] as &Vec<&Option<_>>)
                                            [l_lod] as &Option<VoxelLODChunkData<VE>>)
                                            .as_ref().unwrap().voxel_type_ids.as_ref().unwrap().data,
                                        lvl,
                                        lod,
                                        l_lvl,
                                        l_lod,
                                        chunk_size,
                                        n_chunk_lvls,
                                        n_lods,
                                        lod_block_fill_thresh,
                                    )
                                } else {
                                    // Generate voxels
                                    gen_func(
                                        pos,
                                        lvl,
                                        lod,
                                        lod_data.overwrite().voxel_ids
                                    )
                                }
                            }
                        }
                    }
                    else if lod_data.voxel_type_ids.is_some() {
                        last_lvl = Some(lvl);
                        last_lod = Some(lod);
                    }

                    last_bitmask_lod = Some(lod);
                    last_bitmask_lvl = Some(lvl);
                }

                visited_lods[lvl].push(lod_data_o);
            }
        }
    }

    pub fn set_voxel(&mut self, index_in_tlc: usize, voxel_typ: VE, meta: &VirtualVoxelMemoryGridMetadata) {
        for lvl in 0..meta.this.n_lvls {
            for lod in 0..meta.this.n_lods {
                if let Some(data) = self.lods[lvl][lod].as_mut() {
                    let block_size = meta.this.chunk_size.pow(lvl as u32) *2usize.pow(lod as u32);
                    data.set_voxel(index_in_tlc / cubed(block_size), voxel_typ);
                }
            }
        }
    }

    pub fn set_voxel_pos(&mut self, pos: VoxelPos<u32>, voxel_typ: VE, meta: &VirtualVoxelMemoryGridMetadata) {
        self.set_voxel(
            index_for_pos_in_tlc(pos.0, meta.this.chunk_size, meta.this.n_lvls, 0, 0),
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