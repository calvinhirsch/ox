use std::sync::Arc;
use cgmath::{Vector3};
use derive_new::new;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::{MemoryAllocator};
use crate::renderer::component::voxel::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::component::voxel::lod::RendererVoxelLOD;
use crate::renderer::component::voxel::VoxelLODUpdateRegions;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::mem_grid::layer::{MemoryGridLayer, MemoryGridLayerMetadata, VirtualGridChunkData, VirtualMemoryGridForLayer};
use crate::world::mem_grid::{PhysicalMemoryGrid};
use crate::world::mem_grid::voxel::gpu_defs::{ChunkBitmask, ChunkVoxelIDs};
use crate::world::mem_grid::utils::cubed;
use crate::world::TLCPos;

pub struct VoxelLODCreateParams {
    pub size: usize,
    pub bitmask_binding: u32,
    pub voxel_type_ids_binding: Option<u32>,
}

#[derive(new)]
pub struct VoxelLOD {
    metadata: VoxelLODMetadata,
    bitmask_layer: MemoryGridLayer<VoxelBitmask, ()>,
    voxel_type_id_layer: Option<MemoryGridLayer<VoxelTypeIDs, ()>>,
    updated_bitmask_regions_layer: MemoryGridLayer<Vec<BufferCopy>, ()>,
}

pub struct VoxelLODMetadata {
    size: usize,
    voxels_per_tlc: usize,
}

struct ChunkBitmaskData<'a> {
    bitmask: ChunkBitmask<'a>,
    loaded: bool,
}
struct ChunkVoxelIDData<'a> {
    voxel_ids: ChunkVoxelIDs<'a>,
    loaded: bool,
}

pub struct VoxelLODChunkData<'a> {
    bitmask: ChunkBitmaskData<'a>,
    voxel_type_ids: Option<ChunkVoxelIDData<'a>>,
    updated_bitmask_regions: &'a mut Vec<BufferCopy>,
}

pub struct VirtualizedVoxelLOD<'a> {
    chunks: Vec<Option<VoxelLODChunkData<'a>>>,
}

impl VoxelLOD {
    pub fn new(
        params: VoxelLODCreateParams,
        voxels_per_tlc: usize,
        start_tlc: TLCPos<i64>,
        buffer_allocator: Arc<dyn MemoryAllocator>,
    ) -> (Self, RendererVoxelLOD) {
        let bitmask = VoxelBitmask::new_vec(voxels_per_tlc * cubed(params.size));
        let voxel_ids = match params.voxel_type_ids_binding {
            Some(_) => Some(ChunkVoxelIDs::new_vec(voxels_per_tlc * cubed(params.size))),
            None => None
        };
        let lod = RendererVoxelLOD::new(
            bitmask.iter().copied(),
            match voxel_ids { None => None, Some(ids) => ids.iter().copied() },
            params.bitmask_binding,
            params.voxel_type_ids_binding,
            buffer_allocator,
        );

        (
            VoxelLOD {
                metadata: VoxelLODMetadata { size, voxels_per_tlc },
                bitmask_layer: MemoryGridLayer::new_raw(
                    MemoryGridLayerMetadata::new(
                        start_tlc,
                        params.size,
                        (),
                    ),
                    bitmask,
                ),
                voxel_type_id_layer: match voxel_ids {
                    None => None,
                    Some(vids) => Some(MemoryGridLayer::new_raw(
                        MemoryGridLayerMetadata::new(
                            start_tlc,
                            params.size,
                            (),
                        ),
                        vids,
                    )),
                },
                updated_bitmask_regions_layer: MemoryGridLayer::new_raw(
                    MemoryGridLayerMetadata::new(
                        start_tlc,
                        params.size,
                        (),
                    ),
                    vec![vec![]; cubed(params.size)]
                ),
            },
            lod,
        )
    }

    /// Aggregate the values from updated_bitmask_regions_layer into absolute regions in the
    /// bitmask buffer, generate regions for updating the voxel type ID buffer, and reset these
    /// tracked regions if clear_regions=true.
    pub fn aggregate_updated_regions(&mut self, clear_regions: bool) -> VoxelLODUpdateRegions {
        let mut bitmask_updated_regions = vec![];
        let mut voxel_id_updated_regions = match self.voxel_type_id_layer { None => None, Some(_) => vec![]};

        for (i, regions) in self.updated_bitmask_regions_layer.borrow_mem_mut().iter_mut().enumerate() {
            let bm_offset = i * self.metadata.voxels_per_tlc / VoxelBitmask::BITS_PER_VOXEL;
            for region in regions {
                bitmask_updated_regions.push(
                    BufferCopy {
                        src_offset: region.src_offset + bm_offset,
                        dst_offset: region.dst_offset + bm_offset,
                        size: region.size,
                        ..Default::default()
                    }
                )
            }

            match &mut voxel_id_updated_regions {
                None => {},
                Some(vi_regions) => {
                    let scale = VoxelTypeIDs::BITS_PER_VOXEL / VoxelBitmask::BITS_PER_VOXEL;
                    for region in regions {
                        vi_regions.push(
                            BufferCopy {
                                src_offset: (region.src_offset + bm_offset) * scale,
                                dst_offset: (region.dst_offset + bm_offset) * scale,
                                size: region.size * scale,
                                ..Default::default()
                            }
                        )
                    }
                }
            }

            if clear_regions { regions.clear(); }
        }

        VoxelLODUpdateRegions { bitmask_updated_regions, voxel_id_updated_regions }
    }
}
impl<VE: VoxelTypeEnum> PhysicalMemoryGrid<VirtualizedVoxelLOD> for VoxelLOD {
    fn shift_offsets(&mut self, shift: Vector3<i64>) {
        self.bitmask_layer.shift_offsets(shift);
        self.voxel_type_id_layer.shift_offsets(shift);
    }

    fn size(&self) -> usize {
        self.metadata.size
    }

    fn as_virtual(&mut self) -> Self::Virtual {
        self.as_virtual_for_size(self.metadata.size)
    }

    fn as_virtual_for_size<'a>(&mut self, grid_size: usize) -> VirtualizedVoxelLOD<'a> {
        let bitmask_vgrid: VirtualMemoryGridForLayer<'a, VoxelBitmask, ChunkBitmaskData, ()> =
            self.bitmask_layer.as_virtual_for_size(grid_size);
        let chunk_bitmasks = bitmask_vgrid.deconstruct();

        let voxel_id_vgrid: VirtualMemoryGridForLayer<'a, VoxelTypeIDs, ChunkVoxelIDData, ()> =
            self.voxel_type_id_layer.as_virtual_for_size(grid_size);
        let chunk_voxel_ids = voxel_id_vgrid.deconstruct();

        let chunk_update_regions =
            self.updated_bitmask_regions_layer.as_virtual_for_size(grid_size).deconstruct();

        VirtualizedVoxelLOD {
            chunks: chunk_bitmasks.into_iter()
                .zip(chunk_voxel_ids.into_iter())
                .zip(chunk_update_regions.into_iter())
                .map(|((bitmask, voxel_type_ids), updated_bitmask_regions)|
                    VoxelLODChunkData {
                        bitmask: bitmask.unwrap(),
                        voxel_type_ids,
                        updated_bitmask_regions: updated_bitmask_regions.unwrap(),
                    }
                ).collect(),
        }
    }
}

impl<'a> VirtualizedVoxelLOD<'a> {
    pub fn deconstruct(self) -> Vec<Option<VoxelLODChunkData<'a>>> {
        self.chunks
    }
}


impl VirtualGridChunkData<VoxelBitmask> for ChunkBitmaskData {
    fn new(slice: &mut [VoxelBitmask], loaded: bool) -> Self {
        ChunkBitmaskData { bitmask: ChunkBitmask::from(slice), loaded }
    }
}

impl VirtualGridChunkData<VoxelTypeIDs> for ChunkVoxelIDData {
    fn new(slice: &mut [VoxelTypeIDs], loaded: bool) -> Self {
        ChunkVoxelIDData { voxel_ids: ChunkVoxelIDs::from(slice), loaded }
    }
}