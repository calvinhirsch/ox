use crate::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use crate::renderer::component::voxels::lod::RendererVoxelLOD;
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::world::mem_grid::layer::{LayerChunkLoadingQueue, MemoryGridLayerChunkData, MemoryGridLayerMetadata, PhysicalMemoryGridLayer, VirtualMemoryGridLayer};
use crate::world::mem_grid::utils::cubed;
use crate::world::mem_grid::voxel::gpu_defs::{ChunkBitmask, ChunkVoxelIDs};
use crate::world::mem_grid::{FromVirtual, PhysicalMemoryGrid, PhysicalMemoryGridStruct, ToVirtual, VirtualMemoryGridStruct};
use crate::world::{TLCPos, TLCVector};
use std::sync::Arc;
use derive_more::Deref;
use itertools::Itertools;
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;
use winit::platform::x11::XVisualID;

pub struct VoxelLODCreateParams {
    pub size: usize,
    pub bitmask_binding: u32,
    pub voxel_type_ids_binding: Option<u32>,
}

#[derive(Deref)]
pub struct VoxelLOD(PhysicalMemoryGridStruct<VoxelLODData, VoxelLODMetadata>);
#[derive(Deref)]
pub struct VirtualVoxelLOD(VirtualMemoryGridStruct<VoxelLODChunkData, VirtualVoxelLODMetadata>);

pub struct VoxelLODData {
    bitmask_layer: PhysicalMemoryGridLayer<Vec<VoxelBitmask>, ()>,
    voxel_type_id_layer: Option<PhysicalMemoryGridLayer<Vec<VoxelTypeIDs>, ()>>,
    updated_bitmask_regions_layer: PhysicalMemoryGridLayer<Vec<BufferCopy>, ()>,
}

pub struct VoxelLODMetadata {
    size: usize,
    voxels_per_tlc: usize,
}

pub struct VirtualVoxelLODMetadata {
    this: VoxelLODMetadata,
    bitmask: MemoryGridLayerMetadata<()>,
    voxel_type_ids: Option<MemoryGridLayerMetadata<()>>,
    updated_regions: MemoryGridLayerMetadata<()>,
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
        let bitmask = VoxelBitmask::new_vec(voxels_per_tlc * cubed(params.size));
        let voxel_ids = match params.voxel_type_ids_binding {
            Some(_) => Some(ChunkVoxelIDs::new_vec(voxels_per_tlc * cubed(params.size))),
            None => None,
        };
        let lod = RendererVoxelLOD::new(
            bitmask.iter().copied(),
            match voxel_ids {
                None => None,
                Some(ids) => ids.iter().copied(),
            },
            params.bitmask_binding,
            params.voxel_type_ids_binding,
            buffer_allocator,
        );

        let common_layer_meta = MemoryGridLayerMetadata::new(start_tlc, params.size, tlc_size, ());

        (
            VoxelLOD(
                PhysicalMemoryGridStruct::new(
                    VoxelLODData {
                        bitmask_layer: PhysicalMemoryGridLayer::new(
                            common_layer_meta.clone(),
                            bitmask,
                        ),
                        voxel_type_id_layer: match voxel_ids {
                            None => None,
                            Some(vids) => Some(PhysicalMemoryGridLayer::new(
                                common_layer_meta.clone(),
                                vids,
                            )),
                        },
                        updated_bitmask_regions_layer: PhysicalMemoryGridLayer::new(
                            common_layer_meta,
                            vec![],
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
                Some(_) => vec![],
            };

            for chunk_regions in regions.iter_mut() {
                let bm_offset = chunk_i * self.metadata.voxels_per_tlc / VoxelBitmask::BITS_PER_VOXEL;
                for region in chunk_regions {
                    bitmask_updated_regions.push(BufferCopy {
                        src_offset: region.src_offset,
                        dst_offset: region.dst_offset + bm_offset,
                        size: region.size,
                        ..Default::default()
                    })
                }

                match &mut voxel_id_updated_regions {
                    None => {}
                    Some(vi_regions) => {
                        let scale = VoxelTypeIDs::BITS_PER_VOXEL / VoxelBitmask::BITS_PER_VOXEL;
                        for region in chunk_regions {
                            vi_regions.push(BufferCopy {
                                src_offset: (region.src_offset) * scale,
                                dst_offset: (region.dst_offset + bm_offset) * scale,
                                size: region.size * scale,
                                ..Default::default()
                            })
                        }
                    }
                }

                if clear_regions {
                    chunk_regions.clear();
                }
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

impl PhysicalMemoryGrid<Vec<VoxelBitmask>, VoxelLODMetadata> for VoxelLOD {
    type ChunkLoadQueue = LayerChunkLoadingQueue;

    fn queue_load_all(&mut self) -> Self::ChunkLoadQueue {
        // Because the queues for all three of the layers will be the same size, only need to get one.
        self.data.bitmask_layer.queue_load_all()
    }

    fn shift(&mut self, shift: TLCVector<i32>, load: TLCVector<i32>) -> Self::ChunkLoadQueue {
        // Because all three of these queues will be the same size, only need to track one.
        self.data.voxel_type_id_layer.and_then(|mut layer| Some(layer.shift(shift, load)));
        self.data.updated_bitmask_regions_layer.shift(shift, load);
        self.data.bitmask_layer.shift_offsets(shift, load)
    }
}


impl ToVirtual<VoxelLODChunkData, VirtualVoxelLODMetadata> for VoxelLOD {
    fn to_virtual_for_size(self, grid_size: usize) -> VirtualVoxelLOD {
        let (data, metadata) = self.deconstruct();

        let (bitmask_chunks, bitmask_meta) = data.bitmask_layer
            .to_virtual_for_size(grid_size)
            .deconstruct();

        let (voxel_id_chunks, voxel_id_meta) = match data.voxel_type_id_layer {
            None => (None, None),
            Some(voxel_id_layer) => voxel_id_layer
                .to_virtual_for_size(grid_size)
                .deconstruct(),
        };

        let (chunk_regions, regions_meta) = data.updated_bitmask_regions_layer
            .to_virtual_for_size(grid_size)
            .deconstruct();

        VirtualVoxelLOD(
            VirtualMemoryGridStruct::new(
                bitmask_chunks
                    .into_iter()
                    .zip(voxel_id_chunks.into_iter())
                    .zip(chunk_regions.into_iter())
                    .map(
                        |(
                             (
                                 bitmask,
                                 voxel_type_ids),
                             updated_bitmask_regions
                         ) | VoxelLODChunkData {
                            bitmask: bitmask.unwrap(),
                            voxel_type_ids,
                            updated_bitmask_regions: updated_bitmask_regions.unwrap(),
                        },
                    )
                    .collect(),
                VirtualVoxelLODMetadata {
                    this: self.0.metadata,
                    bitmask: bitmask_meta,
                    voxel_type_ids: voxel_id_meta.unwrap_or(None),
                    updated_regions: regions_meta,
                }
            )
        )
    }
}


impl FromVirtual<VoxelLODChunkData, VirtualVoxelLODMetadata> for VoxelLOD {
    fn from_virtual_for_size(virtual_grid: VirtualMemoryGridStruct<VoxelLODChunkData, VirtualVoxelLODMetadata>, grid_size: usize) -> Self {
        let (data, metadata) = virtual_grid.deconstruct();
        let (bitmask_grid, voxel_id_grid, update_grid) = data.into_iter().map(
            |chunk_o| match chunk_o {
                None => (None, None, None),
                Some(chunk) => (chunk.bitmask, chunk.voxel_type_ids, chunk.updated_bitmask_regions),
            }
        ).collect();

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
                        Some(m) => {
                            PhysicalMemoryGridLayer::from_virtual_for_size(
                                VirtualMemoryGridStruct::new(
                                    voxel_id_grid,
                                    m,
                                ),
                                grid_size,
                            )
                        }
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