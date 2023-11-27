use cgmath::{Vector3};
use itertools::Itertools;
use vulkano::memory::allocator::MemoryAllocator;
use crate::renderer::component::voxel::VoxelData;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::mem_grid::{PhysicalMemoryGrid};
use crate::world::mem_grid::voxel::gpu_defs::ChunkVoxelIDs;
use crate::world::TLCPos;
use super::lod::{VoxelLODCreateParams, VoxelLOD, VoxelLODChunkData};

pub struct VoxelMemoryGrid {
    metadata: VoxelMemoryGridMetadata,
    lods: Vec<Vec<Option<VoxelLOD>>>
}

struct VoxelMemoryGridMetadata {
    size: usize,
    n_lvls: usize,
    n_lods: usize,
}

pub struct VoxelMemoryGridChunkData<'a> {
    lods: Vec<Vec<Option<VoxelLODChunkData<'a>>>>
}

pub struct VirtualVoxelMemoryGrid<'a> {
    chunks: Vec<VoxelMemoryGridChunkData<'a>>,
}

impl VoxelMemoryGrid {
    fn voxels_per_tlc(chunk_size: usize, n_lvls: usize, lvl: usize, lod: usize) -> usize {
        chunk_size.pow((n_lvls - lvl) as u32).to_le() >> (lod*3)
    }

    pub fn new(lod_params: Vec<Vec<Option<VoxelLODCreateParams>>>, memory_allocator: Box<dyn MemoryAllocator>, chunk_size: usize, start_tlc: TLCPos<i64>) -> (Self, VoxelData) {
        let size = lod_params.iter().flatten().filter_map_ok(|p| Some(p?.size)).iter().max().unwrap();
        let n_lvls = lod_params.len();
        let n_lods = lod_params.iter().next().unwrap().len();

        let (grid_lods, lods) = lod_params
            .into_iter().enumerate().map(|(lvl, lvl_sizes)|
            lvl_sizes.into_iter().enumerate().map(|(lod, params)| {
                let voxels_per_tlc = Self::voxels_per_tlc(chunk_size, n_lvls, lvl, lod);
                VoxelLOD::new(
                    params?,
                    voxels_per_tlc,
                    start_tlc,
                    memory_allocator,
                )
            }).unzip()
        ).unzip();

        (
            VoxelMemoryGrid {
                lods: grid_lods,
                metadata: VoxelMemoryGridMetadata { size, n_lvls, n_lods }
            },
            VoxelData::new(lods)
        )
    }

    pub fn set_renderer_updated_regions(&mut self, renderer_voxel_data: &mut VoxelData) {
        let updated_regions = self.lods.iter_mut().map(|lod_o|
            match lod_o {
                None => None,
                Some(lod) => lod.aggregate_updated_regions(),
            }
        ).collect();

        renderer_voxel_data.set_updated_regions(updated_regions);
    }
}
impl<VE: VoxelTypeEnum> PhysicalMemoryGrid<VirtualVoxelMemoryGrid> for VoxelMemoryGrid {
    fn shift_offsets(&mut self, shift: Vector3<i64>) {
        for lod in self.lods.iter_mut().flatten() {
            lod.shift_offsets(shift);
        }
    }

    fn size(&self) -> usize {
        self.metadata.size
    }

    fn as_virtual(self) -> VirtualVoxelMemoryGrid {
        self.as_virtual_for_size(self.metadata.size)
    }

    fn as_virtual_for_size(self, grid_size: usize) -> VirtualVoxelMemoryGrid {
        let v_lods = self.lods.map(|lvl_lods|
            lvl_lods.map(|lod_o| {
                match lod_o {
                    None => (None, None),
                    Some(lod) => { lod.as_virtual_for_size(grid_size).deconstruct() },
                }
            }).collect()
        ).collect();

        VirtualVoxelMemoryGrid {
            chunks: LODSplitter(v_lods).into_iter().collect(),
        }
    }
}


struct LODSplitter<I: Iterator>(Vec<Vec<Option<I>>>);

impl<I: Iterator> Iterator for LODSplitter<I> {
    type Item = Vec<Vec<I::Item>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.iter_mut().map(|iters|
            iters.iter().filter_map_ok(|iter| iter?.next()).collect()
        ).collect()
    }
}