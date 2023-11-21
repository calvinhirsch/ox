use cgmath::{Array, EuclideanSpace, Point3, Vector3};
use derive_new::new;
use vulkano::command_buffer::BufferCopy;
use super::{
    MemoryGridMetadata, MemoryGridLayerMetadata,
    utils::{cubed, pos_index, pos_for_index},
    rendering::{
        RenderingLayerChunkData,
    },
    physical_grid::{MemoryGrid, MemoryGridLayer}
};
use crate::{
    world::{
        VoxelPos, TLCPos
    },
    voxel_type::{VoxelTypeEnum},
};
use crate::world::mem_grid::MemoryGridLayerChunkData;
use crate::world::mem_grid::physical_grid::MemoryGridLayerSet;
use crate::world::mem_grid::rendering::gpu_defs::{ChunkBitmask, ChunkVoxelIDs};
use crate::world::mem_grid::rendering::{ChunkRenderingData, RenderingLayerSet, RenderingLayerSetMetadata};
use crate::world::TLCVector;


pub trait MemoryGridChunkData: Sized {
    fn new_empty() -> Self;
    fn new_blank(chunk_size: usize) -> Self;
}


#[derive(Clone, new)]
pub struct TopLevelChunk<VE: VoxelTypeEnum, CD: MemoryGridChunkData> {
    data: CD,
    rendering_data: ChunkRenderingData<VE>,
}

impl<VE: VoxelTypeEnum, CD: MemoryGridChunkData> TopLevelChunk<VE, CD> {
    // TODO: These should do something to self.data also

    fn set_raw_voxel_ids(&mut self, voxel_ids: ChunkVoxelIDs, grid_meta: &MemoryGridMetadata) {
        self.update_higher_lods_from(&voxel_ids, grid_meta);
        self.rendering_data[0][0].unwrap().data.set_from_voxels(voxel_ids, 0, 0, grid_meta);
    }

    fn update_higher_lods_from(&mut self, voxel_ids: &ChunkVoxelIDs, grid_meta: &MemoryGridMetadata) {
        for lvl in 0..grid_meta.n_chunk_lvls {
            for lod in 0..grid_meta.n_lods {
                if (lvl, lod) == (0, 0) { continue; }
                match &mut self.rendering_data[lvl][lod] {
                    None => {},
                    Some(rdata) => {
                        rdata.data.calc_from_lower_lod_voxels(
                            voxel_ids,
                            lod,
                            lvl,
                            0,
                            0,
                            grid_meta,
                        );
                    }
                }
            }
        }
    }
}

#[derive(Clone, new)]
pub struct VirtualMemoryGrid<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> {
    top_level_chunks: Vec<TopLevelChunk<VE, DL::ChunkData>>,
    rendering_layer_metadata: RenderingLayerSetMetadata,
    data_layer_metadata: DL::LayerSetMetadata,
    gen_func: fn(VoxelPos<i64>) -> VE,
    meta: MemoryGridMetadata,
}

impl<VE: VoxelTypeEnum, DL: MemoryGridLayerSet> VirtualMemoryGrid<VE, DL> {
    fn virtual_grid_pos_for_layer(&self, pos: &TLCPos<usize>, layer_meta: &MemoryGridLayerMetadata<DL::LayerExtraMetadata>) -> TLCPos<usize> {
        TLCPos(pos.0 - Vector3::<usize>::from_value(
            if self.meta.size > layer_meta.size { (self.meta.size - (layer_meta.size-1)) / 2 }
            else { 0 }
        ))
    }

    pub fn load_or_generate_tlc(&self, voxel_output: &mut ChunkVoxelIDs, tlc: TLCPos<i64>) {
        // TODO: Load if already visited this chunk

        let tlc_start_pt = tlc.0 * self.meta.tlc_size as i64;

        // Loop through TLC voxels and call world gen function on each
        let mut index: Vec<usize> = vec![0; self.meta.n_chunk_lvls+1];
        let mut pos: Vec<Vector3<usize>> = vec![Vector3::<usize>::from_value(0); self.meta.n_chunk_lvls+1];
        while index[self.meta.n_chunk_lvls] < cubed(self.meta.chunk_size) {
            let mut vox_pos = Vector3::<usize>::from_value(0);
            let mut vox_index = 0;
            for lvl in (0..=self.meta.n_chunk_lvls).rev() {
                vox_pos *= self.meta.chunk_size;
                vox_pos += pos[lvl];
                vox_index *= cubed(self.meta.chunk_size);
                vox_index += index[lvl];
            }

            voxel_output[vox_index] = (self.gen_func)(VoxelPos(tlc_start_pt + vox_pos.cast::<i64>().unwrap())).into();

            index[0] += 1;
            pos[0] = pos_for_index(index[0], self.meta.chunk_size);
            let mut lvl = 0;
            while index[lvl] > cubed(self.meta.chunk_size) && lvl < self.meta.n_chunk_lvls {
                index[lvl] = 0;
                pos[lvl] = Vector3::<usize>::from_value(0);
                index[lvl+1] += 1;
                pos[lvl+1] = pos_for_index(index[lvl+1], self.meta.chunk_size);

                lvl += 1;
            }
        }
    }

    pub fn reload_all(&mut self) {
        // TODO: This loads all the LOD 0 voxels first which is unnecessary but easy
        let mut tlc_voxel_ids = vec![ChunkVoxelIDs::new(cubed(self.meta.tlc_size))];
        for x in 0..self.meta.size as i64 {
            for z in 0..self.meta.size as i64 {
                for y in 0..self.meta.size as i64 {
                    let tlc = TLCPos(self.meta.start_tlc.0 + Vector3 { x, y, z });
                    self.load_or_generate_tlc(
                        &mut tlc_voxel_ids[pos_index(tlc.0.cast::<usize>().unwrap().to_vec(), self.meta.size)],
                        tlc,
                    );
                }
            }
        }

        for (mut tlc, raw_voxels) in self.top_level_chunks.iter().zip(tlc_voxel_ids.into_iter()) {
            tlc.set_raw_voxel_ids(raw_voxels, &self.meta);
        }
    }

    pub fn set_voxel(&mut self, position: Point3<usize>, voxel_type: VE) -> Option<()> {
        let tlc = position / self.meta.tlc_size;
        let mut did_something = false;
        for lod in 0..self.meta.n_lods {
            match &self.rendering_layer_metadata[0][lod] {
                None => {},
                Some(meta) => {
                    let layer_tlc = self.virtual_grid_pos_for_layer(&TLCPos(tlc), meta);
                    match &mut self.top_level_chunks[pos_index(layer_tlc.0.to_vec(), meta.size)].rendering_data[0][lod] {
                        None => {},
                        Some(layer_chunk) => {
                            did_something = true;

                            let voxel_size = 2usize.pow(lod as u32);
                            let pos_in_tlc = (position % self.meta.tlc_size) / voxel_size;
                            let index_in_tlc = pos_index(pos_in_tlc.to_vec(), self.meta.tlc_size / voxel_size);

                            layer_chunk.set_voxel(index_in_tlc, voxel_type);
                        }
                    }
                }
            }
        }

        // TODO: call set_voxel on data layers

        Some(())
    }

    pub fn lock(self) -> Option<MemoryGrid<VE, DL>> {
        let (rendering_layer_chunks, data_layer_chunks) = self.top_level_chunks.into_iter().map(|tlc| {
            (tlc.rendering_data, tlc.data)
        }).unzip();

        Some(
            MemoryGrid::new_raw(
                DL::from_virtual(data_layer_chunks, self.data_layer_metadata, &self.meta)?,
                RenderingLayerSet::from_virtual(rendering_layer_chunks, self.rendering_layer_metadata, &self.meta)?,
                self.gen_func,
                MemoryGridMetadata {
                    size: self.meta.size + 1,
                    ..self.meta
                },
            )
        )
    }
}