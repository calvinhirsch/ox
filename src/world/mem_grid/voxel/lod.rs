use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::component::voxels::lod::RendererVoxelLOD;
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::world::mem_grid::layer::{MemoryGridLayerChunkData, MemoryGridLayerData, MemoryGridLayerMetadata, PhysicalMemoryGridLayer};
use crate::world::mem_grid::utils::cubed;
use crate::world::mem_grid::voxel::gpu_defs::{ChunkBitmask, ChunkVoxelIDs};
use crate::world::mem_grid::{FromVirtual, MemoryGridMetadata, PhysicalMemoryGrid, PhysicalMemoryGridStruct, ToVirtual, VirtualMemoryGridStruct};
use crate::world::{TLCPos, TLCVector};
use std::sync::Arc;
use derive_more::Deref;
use itertools::Itertools;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;
use crate::world::loader::ChunkLoadQueueItem;


pub struct VoxelLODCreateParams {
    pub size: usize,
    pub bitmask_binding: u32,
    pub voxel_type_ids_binding: Option<u32>,
}

#[derive(Deref)]
pub struct VoxelLOD(PhysicalMemoryGridStruct<VoxelLODData, VoxelLODMetadata>);
pub type VirtualVoxelLOD = VirtualMemoryGridStruct<VoxelLODChunkData, VirtualVoxelLODMetadata>;

pub struct VoxelLODData {
    bitmask_layer: PhysicalMemoryGridLayer<Vec<VoxelBitmask>, ()>,
    voxel_type_id_layer: Option<PhysicalMemoryGridLayer<Vec<VoxelTypeIDs>, ()>>,
    updated_bitmask_regions_layer: PhysicalMemoryGridLayer<Vec<BufferCopy>, ()>,
}

pub struct VoxelLODMetadata {
    size: usize,
    voxels_per_tlc: usize,
}
impl MemoryGridMetadata for VoxelLODMetadata {
    fn size(&self) -> usize { self.size }
}

pub struct VirtualVoxelLODMetadata {
    this: VoxelLODMetadata,
    bitmask: MemoryGridLayerMetadata<()>,
    voxel_type_ids: Option<MemoryGridLayerMetadata<()>>,
    updated_regions: MemoryGridLayerMetadata<()>,
}
impl MemoryGridMetadata for VirtualVoxelLODMetadata {
    fn size(&self) -> usize { self.this.size }
}

pub struct VoxelLODChunkData {
    bitmask: MemoryGridLayerChunkData<ChunkBitmask>,
    voxel_type_ids: Option<MemoryGridLayerChunkData<ChunkVoxelIDs>>,
    updated_bitmask_regions: MemoryGridLayerChunkData<Vec<BufferCopy>>,
}


impl VoxelLOD {
    pub fn new(
        params: VoxelLODCreateParams,
        voxels_per_tlc: usize,
        start_tlc: TLCPos<i64>,
        tlc_size: usize,
        buffer_allocator: Arc<dyn MemoryAllocator>,
    ) -> (Self, RendererVoxelLOD) {
        let bitmask = vec![VoxelBitmask::new_vec(voxels_per_tlc); cubed(params.size)];
        let voxel_ids = match params.voxel_type_ids_binding {
            Some(_) => Some(vec![VoxelTypeIDs::new_vec(voxels_per_tlc); cubed(params.size)]),
            None => None,
        };
        let lod = RendererVoxelLOD::new(
            bitmask.iter().flatten().copied().collect::<Vec<_>>().into_iter(),  // ENHANCEMENT: Do this better (and below)
            match &voxel_ids {
                None => None,
                Some(ids) => Some(ids.iter().flatten().copied().collect::<Vec<_>>().into_iter()),
            },
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
                        bitmask_layer: PhysicalMemoryGridLayer::new(
                            PhysicalMemoryGridStruct {
                                metadata: common_layer_meta.clone(),
                                data: MemoryGridLayerData::new(bitmask),
                            }
                        ),
                        voxel_type_id_layer: match voxel_ids {
                            None => None,
                            Some(vids) => Some(PhysicalMemoryGridLayer::new(
                                PhysicalMemoryGridStruct {
                                    data: MemoryGridLayerData::new(vids),
                                    metadata: common_layer_meta.clone(),
                                }
                            )),
                        },
                        updated_bitmask_regions_layer: PhysicalMemoryGridLayer::new(
                            PhysicalMemoryGridStruct {
                                data: MemoryGridLayerData::new(vec![]),
                                metadata: common_layer_meta,
                            }
                        )
                    },
                    VoxelLODMetadata {
                        size: params.size,
                        voxels_per_tlc,
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

        for (chunk_i, regions) in self.data
            .updated_bitmask_regions_layer
            .borrow_mem_mut()
            .iter_mut()
            .enumerate()
        {
            let mut bitmask_updated_regions = vec![];
            let mut voxel_id_updated_regions = match self.data.voxel_type_id_layer {
                None => None,
                Some(_) => Some(vec![]),
            };
            let bm_offset = chunk_i * self.metadata.voxels_per_tlc / VoxelBitmask::BITS_PER_VOXEL;

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
                    bitmask: &self.data.bitmask_layer.borrow_mem()[chunk_i],
                    voxel_type_ids: self.data.voxel_type_id_layer.and_then(|layer| Some(&layer.borrow_mem()[chunk_i])),
                    bitmask_updated_regions,
                    voxel_id_updated_regions,
                }
            )
        }

        updates
    }
}

impl PhysicalMemoryGrid for VoxelLOD {
    type Data = VoxelLODData;
    type Metadata = VoxelLODMetadata;
    type ChunkLoadQueueItemData = ();

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<()>> {
        // Because the queues for all three of the layers will be the same size, only need to get one.
        self.data.bitmask_layer.queue_load_all()
    }

    fn shift(&mut self, shift: TLCVector<i32>, load_in_from_edge: TLCVector<i32>, load_buffer: [bool; 3]) -> Vec<ChunkLoadQueueItem<()>> {
        // Because all three of these queues will be the same size, only need to track one.
        self.data.voxel_type_id_layer.and_then(
            |mut layer|
                Some(layer.shift(shift, load_in_from_edge, load_buffer))
        );
        self.data.updated_bitmask_regions_layer.shift(shift, load_in_from_edge, load_buffer);
        self.data.bitmask_layer.shift(shift, load_in_from_edge, load_buffer)
    }
}


impl ToVirtual<VoxelLODChunkData, VirtualVoxelLODMetadata> for VoxelLOD {
    fn to_virtual_for_size(self, grid_size: usize) -> VirtualVoxelLOD {
        // ENHANCEMENT: Make this call .to_virtual_for_size(self.size()) on children instead of
        // use grid_size and then add the necessary padding in this function.
        let (data, metadata) = self.deconstruct();

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
                .zip(voxel_id_chunks.unwrap_or(vec![None; chunk_regions.len()]).into_iter())
                .zip(chunk_regions.into_iter())
                .map(
                    |(
                         (
                             bitmask,
                             voxel_type_ids
                         ),
                         updated_bitmask_regions
                     )|
                        match bitmask {
                            None => None,
                            Some(bm) => Some(VoxelLODChunkData {
                                bitmask: bm,
                                voxel_type_ids,
                                updated_bitmask_regions: updated_bitmask_regions.unwrap(),
                            }),
                        }
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


impl FromVirtual<VoxelLODChunkData, VirtualVoxelLODMetadata> for VoxelLOD {
    fn from_virtual_for_size(virtual_grid: VirtualMemoryGridStruct<VoxelLODChunkData, VirtualVoxelLODMetadata>, grid_size: usize) -> Self {
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
                    bitmask_layer: PhysicalMemoryGridLayer::from_virtual_for_size(
                        VirtualMemoryGridStruct::new(
                            bitmask_grid,
                            metadata.bitmask,
                        ),
                        grid_size,
                    ),
                    voxel_type_id_layer: match metadata.voxel_type_ids {
                        None => None,
                        Some(m) => Some(
                            PhysicalMemoryGridLayer::from_virtual_for_size(
                                VirtualMemoryGridStruct::new(
                                    voxel_id_grid,
                                    m,
                                ),
                                grid_size,
                            )
                        )
                    },
                    updated_bitmask_regions_layer: PhysicalMemoryGridLayer::from_virtual_for_size(
                        VirtualMemoryGridStruct::new(
                            update_grid,
                            metadata.updated_regions,
                        ),
                        grid_size,
                    ),
                },
                metadata: virtual_grid.metadata.this,
            }
        )
    }
}