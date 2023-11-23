use cgmath::{Point3, Vector3};
use itertools::Itertools;
use syn::__private::quote::__private::ext::RepToTokensExt;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::camera::Camera;
use crate::world::mem_grid::layer::{MemoryGridLayer, MemoryGridLayerMetadata, VirtualMemoryGridForLayer};
use crate::world::mem_grid::{PhysicalMemoryGrid, VirtualMemoryGrid};
use crate::world::mem_grid::rendering::gpu_defs::ChunkVoxelIDs;
use crate::world::TLCPos;
use super::lod::{BitmaskLayerMetadata, RenderingGridLOD, RenderingGridLODChunkData, RenderingGridLODMetadata, VoxelTypeIdLayerMetadata};

struct RenderingMemoryGrid {
    metadata: RenderingMemoryGridMetadata,
    lods: Vec<Vec<RenderingGridLOD>>
}

struct RenderingMemoryGridMetadata {
    size: usize,
    n_lvls: usize,
    n_lods: usize,
}

struct RenderingMemoryGridChunkData {
    lods: Vec<Vec<RenderingGridLODChunkData>>
}

struct NestedRenderingGridLODMetadata {
    lod_meta: RenderingGridLODMetadata,
    bitmask_meta: MemoryGridLayerMetadata<BitmaskLayerMetadata>,
    voxel_type_id_meta: MemoryGridLayerMetadata<VoxelTypeIdLayerMetadata>,
}

struct VirtualRenderingMemoryGrid {
    metadata: RenderingMemoryGridMetadata,
    lod_metadata: Vec<Vec<NestedRenderingGridLODMetadata>>,
    chunks: Vec<RenderingMemoryGridChunkData>,
}

impl<VE: VoxelTypeEnum> PhysicalMemoryGrid<VE> for RenderingMemoryGrid {
    type Virtual = VirtualRenderingMemoryGrid;

    fn move_camera(&mut self, camera: &mut Camera, tlc_size: usize) {
        let move_vector = Self::move_camera_for_size(camera, tlc_size, self.metadata.size);
        self.shift_offsets(move_vector);
    }

    fn shift_offsets(&mut self, shift: Vector3<i64>) {
        for lod in self.lods.iter_mut().flatten() {
            lod.shift_offsets(shift);
        }
    }

    fn to_virtual(self) -> Self::Virtual {
        self.to_virtual_for_size(self.metadata.size)
    }

    fn to_virtual_for_size(self, grid_size: usize) -> Self::Virtual {
        let (lod_meta, v_lods) = self.lods.map(|lvl_lods|
            lvl_lods.map(|lod_o| {
                match lod_o {
                    None => (None, None),
                    Some(lod) => (Some(lod.meta), Some(lod.to_virtual_for_size(grid_size).into_iter())),
                }
            }).unzip()
        ).unzip();

        VirtualRenderingMemoryGrid {
            metadata: self.metadata,
            lod_metadata: lod_meta,
            chunks: LODSplitter(v_lods).into_iter().collect(),
        }

    }
}

impl<VE: VoxelTypeEnum> VirtualMemoryGrid<VE> for VirtualRenderingMemoryGrid {
    type Physical = RenderingMemoryGrid;

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
        // ENHANCEMENT: Probably a nicer way to do this
        let mut grid = vec![vec![vec![None; self.chunks.len()]; self.metadata.n_lvls]; self.metadata.n_lods];
        for (i, chunk) in self.chunks.enumerate() {
            for (lvl, lvl_lods) in chunk.lods.enumerate() {
                for (lod, data) in lvl_lods.enumerate() {
                    grid[lvl][lod][i] = data;
                }
            }
        }

        Some(
            RenderingMemoryGrid {
                metadata: self.metadata,
                lods: self.lod_metadata.into_iter().zip(grid.into_iter())
                    .map(|(lvl_metas, lvl_lods)|
                    lvl_metas.into_iter().zip(lvl_lods.into_iter())
                        .map(|(meta, data)| {
                            let (chunk_bitmasks, chunk_voxel_ids) = data.into_iter().filter_map_ok(|d|
                                Some((d?.bitmask, d?.voxel_id_types))
                            ).unzip();
                            RenderingGridLOD::new(
                                meta.lod_meta,
                                MemoryGridLayer::new_raw(meta.bitmask_meta, chunk_bitmasks),
                                match chunk_voxel_ids {
                                    None => None,
                                    Some(chunks) => MemoryGridLayer::new_raw(meta.voxel_type_id_meta, chunks),
                                },
                            )
                        }
                    ).collect()
                ).collect(),
            }
        )
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