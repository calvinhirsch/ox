use cgmath::{Array, EuclideanSpace, Point3, Vector3};
use derive_new::new;
use vulkano::command_buffer::BufferCopy;
use super::{
    MemoryGridMetadata, MemoryGridLayer, MemoryGridLayerMetadata,
    utils::{cubed, pos_index, pos_for_index},
    rendering_data::{
        MemoryGridChunkRenderingData,
        gpu_defs::{VoxelTypeIDs},
    },
};
use crate::{
    world::{
        VoxelPos, TLCPos
    },
    voxel_type::{VoxelTypeEnum},
};

pub struct TrackedMemoryGridChunkRenderingData<VE: VoxelTypeEnum> {
    data: MemoryGridChunkRenderingData<VE>,
    bitmask_updated_regions: Vec<BufferCopy>,
    voxel_ids_updated_regions: Vec<BufferCopy>,
}


#[derive(Clone, new)]
pub struct TopLevelChunk<VE: VoxelTypeEnum> {
    // TODO: Include user defined data here
    rendering_layer_data: Vec<Vec<Option<TrackedMemoryGridChunkRenderingData<VE>>>>,
}

impl<VE: VoxelTypeEnum> TopLevelChunk<VE> {
    fn update_higher_lods_from(&mut self, data: MemoryGridChunkRenderingData<VE>) {

    }
}

#[derive(Clone)]
pub struct VirtualMemoryGrid<VE: VoxelTypeEnum> {
    data: Vec<TopLevelChunk<VE>>,
    layer_metadata: Vec<Vec<Option<MemoryGridLayerMetadata>>>,
    gen_func: fn(VoxelPos<i64>) -> VE,
    meta: MemoryGridMetadata,
}

impl<VE: VoxelTypeEnum> VirtualMemoryGrid<VE> {
    fn virtual_grid_pos_for_layer(&self, pos: &TLCPos<usize>, layer_meta: &MemoryGridLayerMetadata) -> TLCPos<usize> {
        TLCPos(pos.0 - Vector3::<usize>::from_value(
            if self.meta.size > layer_meta.size { (self.meta.size - (layer_meta.size-1)) / 2 }
            else { 0 }
        ))
    }

    pub fn load_or_generate_tlc(&self, voxel_output: &mut Vec<VoxelTypeIDs>, tlc: TLCPos<i64>) {
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
            voxel_output[vox_index / (128 / 8)].indices[vox_index % (128 / 8)] =
                (self.gen_func)(VoxelPos(tlc_start_pt + vox_pos.cast::<i64>().unwrap())).into();

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
        let mut chunk_voxels: Vec<Vec<VoxelTypeIDs>> = Vec::with_capacity(cubed(self.meta.size));
        let mut i = 0;
        for x in 0..self.meta.size as i64 {
            for z in 0..self.meta.size as i64 {
                for y in 0..self.meta.size as i64 {
                    let tlc = TLCPos(self.meta.start_tlc.0 + Vector3 { x, y, z });
                    self.load_or_generate_tlc(
                        &mut self.data[pos_index(tlc.0.cast::<usize>().unwrap().to_vec(), self.meta.size)].rendering_layer_data[0][0],
                        tlc,
                    );
                    i += 1;
                }
            }
        }

        for (chunk_voxels, chunk_layers) in chunk_voxels.iter().zip(self.data.iter_mut()){
            let (last_lvl, last_lod) = (0, 0);
            for lvl in 0..self.meta.n_chunk_lvls {
                for lod in 0..self.meta.n_lods {
                    if (lvl, lod) == (0, 0) { continue; }
                    match chunk_layers.rendering_layer_data[lvl][lod] {
                        None => {},
                        Some(rdata) => {
                            rdata.calc_from_lower_lod_voxels()
                        }
                    }
                    (last_lvl, last_lod) = (lvl, lod);
                    .
                }
            }
        }
    }

    pub fn set_voxel(&mut self, position: Point3<usize>, voxel_type: VE) -> Option<()> {
        let tlc = position / self.meta.tlc_size;
        let mut did_something = false;
        for lod in 0..self.meta.n_lods {
            match &self.layer_metadata[0][lod] {
                None => {},
                Some(meta) => {
                    let layer_tlc = self.virtual_grid_pos_for_layer(&TLCPos(tlc), meta);
                    match &mut self.data[pos_index(layer_tlc.0, meta.size)].rendering_layer_data[0][lod] {
                        None => {},
                        Some(layer_chunk) => {
                            did_something = true;

                            let voxel_size = 2.pow(lod as u32) as usize;
                            let pos_in_tlc = (position % self.meta.tlc_size) / voxel_size;
                            let index_in_tlc = pos_index(pos_in_tlc, self.meta.tlc_size / voxel_size);

                            layer_chunk.set_block(index_in_tlc, &voxel_type);
                        }
                    }
                }
            }
        }

        Some(())
    }

    pub fn consolidate_and_start_transfer(self) -> Option<(MemoryGrid<VE>, Vec<Vec<Vec<BufferCopy>>>)> {
        let mut layer_chunks: Vec<Vec<Option<Vec<Option<MemoryGridChunkRenderingData<VE>>>>>> =
            self.layer_metadata.iter().map(|lods| {
                lods.iter().map(|data| {
                    match data {
                        None => None,
                        Some(meta) => { Some(vec![None; meta.size]) }
                    }
                }).collect()
            }).collect();

        let mut updated_regions: Vec<Vec<Vec<BufferCopy>>> = vec![vec![vec![]; self.meta.n_lods]; self.meta.n_chunk_lvls+1];

        for (i, chunk) in self.data.into_iter().enumerate() {
            for (lvl, lvl_data) in chunk.rendering_layer_data.into_iter().enumerate() {
                for (lod, data) in lvl_data.into_iter().enumerate() {
                    let meta = self.layer_metadata[lvl][lod]?;
                    layer_chunks[lvl][lod]?[
                            meta.index_for_virtual_grid_pos(
                                TLCPos(pos_for_index(i, self.meta.size) - Vector3::<usize>::from_value(
                                    if self.meta.size > meta.size { (self.meta.size - (meta.size-1)) / 2 }
                                    else { 0 }
                                ))
                            )
                        ] = data;
                }
            }
        }

        // TODO: Aggregate all updated regions, copy to staging buffers, and then return them in updated_regions

        Some((
             MemoryGrid {
                rendering_layers: layer_chunks.into_iter().zip(self.layer_metadata).map(|(chunks_lvl, meta_lvl)| {
                    chunks_lvl.into_iter().zip(meta_lvl).map(|(chunks, meta)| {
                        match chunks {
                            None => None,
                            Some(data) => Some(MemoryGridLayer {
                                chunks: data.try_into().unwrap(),
                                meta: meta?,
                            }),
                        }
                    }).collect()
                }).collect(),
                gen_func: self.gen_func,
                meta: MemoryGridMetadata {
                    size: self.meta.size + 1,
                    ..self.meta
                }
            },
            updated_regions,
        ))
    }
}