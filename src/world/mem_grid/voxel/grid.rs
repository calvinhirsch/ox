use std::sync::Arc;
use super::lod::{VoxelLOD, VoxelLODChunkData, VoxelLODCreateParams, VoxelLODMetadata};
use crate::renderer::component::voxels::VoxelData;
use crate::world::mem_grid::{PhysicalMemoryGrid, PhysicalMemoryGridStruct, ToVirtual, VirtualMemoryGridStruct};
use crate::world::{TLCPos, TLCVector, VoxelPos};
use derive_more::Deref;
use itertools::Itertools;
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::world::mem_grid::layer::LayerChunkLoadingQueue;


#[derive(Deref)]
pub struct VoxelMemoryGrid(PhysicalMemoryGridStruct<VoxelMemoryGridData, VoxelMemoryGridMetadata>);
#[derive(Deref)]
pub struct VirtualVoxelMemoryGrid(VirtualMemoryGridStruct<VoxelMemoryGridChunkData, VirtualVoxelMemoryGridMetadata>);

pub struct VoxelMemoryGridData {
    lods: Vec<Vec<Option<VoxelLOD>>>
}

struct VoxelMemoryGridMetadata {
    size: usize,
    n_lvls: usize,
    n_lods: usize,
}

struct VirtualVoxelMemoryGridMetadata {
    this: VoxelMemoryGridMetadata,
    lods: Vec<Vec<Option<VoxelLODMetadata>>>
}

pub struct VoxelMemoryGridChunkData {
    lods: Vec<Vec<Option<VoxelLODChunkData>>>,
}

pub struct VoxelMemoryGridChunkLoadingQueue {
    lods: Vec<Vec<Option<LayerChunkLoadingQueue>>>
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
            .filter_map_ok(|p| Some(p?.size))
            .iter()
            .max()
            .unwrap();
        let n_lvls = lod_params.len();
        let n_lods = lod_params.iter().next().unwrap().len();

        let (grid_lods, lods) = lod_params
            .into_iter()
            .enumerate()
            .map(|(lvl, lvl_sizes)| {
                lvl_sizes
                    .into_iter()
                    .enumerate()
                    .map(|(lod, params)| {
                        let voxels_per_tlc = Self::voxels_per_tlc(chunk_size, n_lvls, lvl, lod);
                        VoxelLOD::new(params?, voxels_per_tlc, start_tlc, tlc_size, memory_allocator)
                    })
                    .unzip()
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

    pub fn get_updates(&mut self) -> Vec<Vec<Option<VoxelLODUpdate>>> {
        self.data.lods
            .iter_mut()
            .flat_map(|lod_o| match lod_o {
                None => None,
                Some(lod) => lod.aggregate_updates(),
            })
            .collect()
    }
}
impl PhysicalMemoryGrid<VoxelMemoryGridData, VoxelMemoryGridMetadata> for VoxelMemoryGrid {
    type ChunkLoadQueue = VoxelMemoryGridChunkLoadingQueue;

    fn shift(&mut self, shift: TLCVector<i32>, load: TLCVector<i32>) -> Self::ChunkLoadQueue {
        for lod in self.data.lods.iter_mut().flatten() {
            lod.shift_offsets(shift, load);
        }
        self.data.lods.iter_mut().map(|lvl_lods|
            lvl_lods.iter_mut().map(|lod_o|
                match lod_o {
                    None => None,
                    Some(lod) => Some(lod.shift(shift, load)),
                }
            ).collect()
        ).collect()
    }
}
impl ToVirtual<VoxelMemoryGridChunkData, VirtualVoxelMemoryGridMetadata> for VoxelMemoryGrid {
    fn to_virtual_for_size(self, grid_size: usize) -> VirtualMemoryGridStruct<Option<VoxelMemoryGridChunkData>, VirtualVoxelMemoryGridMetadata> {
        let v_lods = self.0.data.lods
            .map(|lvl_lods| {
                lvl_lods
                    .map(|lod_o| match lod_o {
                        None => (None, None),
                        Some(lod) => lod.as_virtual_for_size(grid_size).deconstruct(),
                    })
                    .collect()
            })
            .collect();

        VirtualMemoryGrid::<VoxelMemoryGridChunkData<'a>> {
            chunks: LODSplitter(v_lods).into_iter().collect(),
        }    }
}

struct LODSplitter<I: Iterator>(Vec<Vec<Option<I>>>);

impl<I: Iterator> Iterator for LODSplitter<I> {
    type Item = Vec<Vec<I::Item>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .iter_mut()
            .map(|iters| iters.iter().filter_map_ok(|iter| iter?.next()).collect())
            .collect()
    }
}
