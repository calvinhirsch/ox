use cgmath::{Point3, Vector3};
use derive_new::new;
use itertools::Itertools;
use vulkano::buffer::Subbuffer;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::camera::Camera;
use crate::world::mem_grid::layer::{MemoryGridLayer, MemoryGridLayerMetadata};
use crate::world::mem_grid::{PhysicalMemoryGrid, VirtualMemoryGrid};
use crate::world::mem_grid::rendering::gpu_defs::{BlockBitmask, ChunkBitmask, ChunkVoxelIDs, VoxelTypeIDs};
use crate::world::TLCPos;

pub struct BitmaskLayerMetadata {
    staging_buffer: Subbuffer<[BlockBitmask]>,
}

pub struct VoxelTypeIdLayerMetadata {
    staging_buffer: Subbuffer<[VoxelTypeIDs]>,
}

struct ChunkBitmaskData {
    bitmask: ChunkBitmask,
    loaded: bool,
}
struct ChunkVoxelIDData {
    voxel_ids: ChunkVoxelIDs,
    loaded: bool,
}

#[derive(new)]
pub struct RenderingGridLOD {
    metadata: RenderingGridLODMetadata,
    bitmask_layer: MemoryGridLayer<ChunkBitmaskData, BitmaskLayerMetadata>,
    voxel_type_id_layer: Option<MemoryGridLayer<ChunkVoxelIDData, VoxelTypeIdLayerMetadata>>,
}

pub struct RenderingGridLODMetadata {
    size: usize,
}

pub struct RenderingGridLODChunkData {
    bitmask: ChunkBitmaskData,
    voxel_type_ids: Option<ChunkVoxelIDData>,
}

pub struct VirtualizedRenderingGridLOD {
    metadata: RenderingGridLODMetadata,
    bitmask_meta: MemoryGridLayerMetadata<BitmaskLayerMetadata>,
    voxel_type_id_meta: Option<MemoryGridLayerMetadata<VoxelTypeIdLayerMetadata>>,
    chunks: Vec<Option<RenderingGridLODChunkData>>,
}

impl<VE: VoxelTypeEnum> PhysicalMemoryGrid<VE> for RenderingGridLOD {
    type Virtual = VirtualizedRenderingGridLOD;

    fn move_camera(&mut self, camera: &mut Camera, tlc_size: usize) {
        let move_vector = Self::move_camera_for_size(camera, tlc_size, self.metadata.size);
        self.shift_offsets(move_vector);
    }

    fn shift_offsets(&mut self, shift: Vector3<i64>) {
        self.bitmask_layer.shift_offsets(shift);
        self.voxel_type_id_layer.shift_offsets(shift);
    }

    fn to_virtual(self) -> Self::Virtual {
        self.to_virtual_for_size(self.metadata.size)
    }

    fn to_virtual_for_size(self, grid_size: usize) -> Self::Virtual {
        let (bitmask_meta, chunk_bitmasks) = self.bitmask_layer
            .to_virtual_for_size(grid_size)
            .destroy();
        let (voxel_type_id_meta, chunk_voxel_ids) = self.voxel_type_id_layer
            .to_virtual_for_size(grid_size)
            .destroy();

        VirtualizedRenderingGridLOD {
            chunks: chunk_bitmasks.into_iter().zip(chunk_voxel_ids.into_iter())
                .map(|(bitmask, voxel_type_ids)| RenderingGridLODChunkData { bitmask, voxel_type_ids })
                .collect(),
            metadata: self.metadata,
            bitmask_meta,
            voxel_type_id_meta,
        }
    }
}

impl<VE: VoxelTypeEnum> VirtualMemoryGrid<VE> for VirtualizedRenderingGridLOD {
    type Physical = RenderingGridLOD;

    fn load_or_generate_tlc(&self, voxel_output: &mut ChunkVoxelIDs, tlc: TLCPos<i64>) {
        todo!()
    }

    fn reload_all(&mut self) {
        todo!()
    }

    fn set_voxel(&mut self, position: Point3<usize>, voxel_type: VE) -> Option<()> {
        todo!()
    }

    fn lock(self) -> Option<Self::Physical> {
        let (chunk_bitmasks, chunk_voxel_ids): (Vec<ChunkBitmaskData>, Option<Vec<ChunkVoxelIDData>>) = {
            match self.voxel_type_id_meta {
                None => (self.chunks.filter_map_ok(|chunk| Some(chunk?.bitmask)), None).collect(),
                Some(_) => self.chunks.filter_map_ok(|chunk| Some((chunk?.bitmask, Some(chunk?.voxel_type_ids.unwrap())))).unzip()
            }
        };

        Some(
            RenderingGridLOD::new(
                self.metadata,
                MemoryGridLayer::new_raw(self.bitmask_meta, chunk_bitmasks),
                match chunk_voxel_ids {
                    None => None,
                    Some(chunks) => MemoryGridLayer::new_raw(self.voxel_type_id_meta.unwrap(), chunks),
                },
            )
        )
    }
}