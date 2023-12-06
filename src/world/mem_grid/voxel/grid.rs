use std::sync::Arc;
use cgmath::{Array, Vector3};
use super::lod::{VirtualVoxelLOD, VirtualVoxelLODMetadata, VoxelLOD, VoxelLODChunkData, VoxelLODCreateParams};
use crate::renderer::component::voxels::VoxelData;
use crate::world::mem_grid::{FromVirtual, MemoryGridMetadata, PhysicalMemoryGrid, PhysicalMemoryGridStruct, ToVirtual, VirtualMemoryGridStruct};
use crate::world::{TLCPos, TLCVector};
use derive_more::Deref;
use hashbrown::{HashMap};
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::world::loader::ChunkLoadQueueItem;
use crate::world::mem_grid::utils::cubed;


#[derive(Deref)]
pub struct VoxelMemoryGrid(PhysicalMemoryGridStruct<VoxelMemoryGridData, VoxelMemoryGridMetadata>);
#[derive(Deref)]
pub struct VirtualVoxelMemoryGrid(VirtualMemoryGridStruct<VoxelMemoryGridChunkData, VirtualVoxelMemoryGridMetadata>);

pub struct VoxelMemoryGridData {
    lods: Vec<Vec<Option<VoxelLOD>>>
}

pub struct VoxelMemoryGridMetadata {
    size: usize,
    n_lvls: usize,
    n_lods: usize,
}
impl MemoryGridMetadata for VoxelMemoryGridMetadata {
    fn size(&self) -> usize { self.size }
}

pub struct VirtualVoxelMemoryGridMetadata {
    this: VoxelMemoryGridMetadata,
    lods: Vec<Vec<Option<VirtualVoxelLODMetadata>>>
}
impl MemoryGridMetadata for VirtualVoxelMemoryGridMetadata {
    fn size(&self) -> usize { self.this.size }
}

#[derive(Clone)]
pub struct VoxelMemoryGridChunkData {
    lods: Vec<Vec<Option<VoxelLODChunkData>>>,
}

pub struct VoxelChunkLoadingQueueItemData {
    pub lods: Vec<Vec<bool>>,
}


impl VoxelMemoryGrid {
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

    fn apply_and_queue<F: FnMut(&mut VoxelLOD) -> Vec<ChunkLoadQueueItem<()>>>(
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
impl PhysicalMemoryGrid for VoxelMemoryGrid {
    type Data = VoxelMemoryGridData;
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
        self.apply_and_queue(|lod| lod.shift(shift, load_in_from_edge, load_buffer))
    }
}
impl ToVirtual<VoxelMemoryGridChunkData, VirtualVoxelMemoryGridMetadata> for VoxelMemoryGrid {
    fn to_virtual_for_size(self, grid_size: usize) -> VirtualMemoryGridStruct<VoxelMemoryGridChunkData, VirtualVoxelMemoryGridMetadata> {
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
impl FromVirtual<VoxelMemoryGridChunkData, VirtualVoxelMemoryGridMetadata> for VoxelMemoryGrid {
    fn from_virtual_for_size(virtual_grid: VirtualMemoryGridStruct<VoxelMemoryGridChunkData, VirtualVoxelMemoryGridMetadata>, grid_size: usize) -> Self {
        let (data, metadata) = virtual_grid.deconstruct();

        let mut chunk_data: Vec<Vec<Option<Vec<Option<VoxelLODChunkData>>>>> = metadata.lods.iter().map(|lvl_meta|
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
