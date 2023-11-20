use cgmath::{Array, Point3, Vector3};
use derive_new::new;
use num_traits::Zero;
use vulkano::buffer::Subbuffer;
use super::{
    MemoryGridMetadata, MemoryGridLayer, MemoryGridLayerMetadata,
    utils::{cubed, pos_index},
    rendering_data::{
        MemoryGridChunkRenderingData,
        gpu_defs::{BlockBitmask, VoxelTypeIDs},
    },
    virtual_grid::{VirtualMemoryGrid, TopLevelChunk},
};
use crate::{
    world::{
        VoxelPos, TLCPos, WorldCreateParams,
        camera::Camera,
    },
    voxel_type::{VoxelTypeEnum},
};


#[derive(Clone, new)]
pub struct MemoryGrid<VE: VoxelTypeEnum> {
    // data_layers: Vec<MemoryGridLayer<dyn MemoryGridChunkData>>,
    rendering_layers: Vec<Vec<Option<MemoryGridLayer<MemoryGridChunkRenderingData<VE>>>>>,
    gen_func: fn(VoxelPos<i64>) -> VE,
    meta: MemoryGridMetadata,
}

impl<VE: VoxelTypeEnum> MemoryGrid<VE> {
    pub fn new<const N_CHUNK_LVLS: usize, const N_LODS: usize>(
        params: WorldCreateParams<N_CHUNK_LVLS, N_LODS, VE>,
        staging_buffers: [[Option<(Subbuffer<[BlockBitmask]>, Subbuffer<[VoxelTypeIDs]>)>; N_LODS]; N_CHUNK_LVLS],
    ) -> Option<MemoryGrid<VE>> {
        let max_size: usize = *(params.render_area_sizes.iter().map(|sizes| sizes.iter().max()).max()??.as_ref())?;
        let start_tlc = TLCPos(params.curr_tlc.0 - Vector3::<i64>::from_value((max_size/2) as i64));

        // TODO: This inits and fills all the memory with zeroes which is unnecessary slowness, but easy

        let meta = MemoryGridMetadata {
            size: max_size,
            n_chunk_lvls: N_CHUNK_LVLS,
            tlc_size: params.chunk_size.pow(N_CHUNK_LVLS as u32),
            n_lods: N_LODS,
            start_tlc,
            chunk_size: params.chunk_size,
            load_thresh_dist: params.load_thresh_dist,
            lod_block_fill_thresh: params.lod_block_fill_thresh,
        };

        Some(MemoryGrid {
            rendering_layers: params.render_area_sizes.into_iter().zip(staging_buffers.iter())
                .map(|(sizes, buffers)| {
                    sizes.into_iter().zip(buffers.iter())
                        .map(|(size, staging_buffer_pair)| {
                        match (size, staging_buffer_pair) {
                            (None, &None) => None,
                            (Some(s), &Some((bm_sb, vid_sb))) => Some(
                                MemoryGridLayer::<MemoryGridChunkRenderingData<VE>>::new(
                                    MemoryGridLayerMetadata {
                                        size: s,
                                        offsets: MemoryGridLayerMetadata::calc_offsets(start_tlc, s),
                                        loaded_upper_chunks: true,  // doesn't matter
                                    },
                                    bm_sb.clone(),
                                    vid_sb.clone(),
                                    &meta,
                                )
                            ),
                            _ => panic!("Need to provide staging buffers and render area sizes for the same set of memory grid layers."),
                        }
                    }).collect::<Vec<_>>()
                }).collect(),
            gen_func: params.gen_func,
            meta,
        })
    }

    pub fn move_grid(&mut self, camera: &mut Camera) {
        let move_vector = (
            camera.position
            / (self.meta.chunk_size.pow(self.meta.n_chunk_lvls as u32) as f32)
        )
            .map(|a| a.floor() as i64)
            - Point3::<i64>::from_value(((self.meta.size-1)/2) as i64);

        if !move_vector.is_zero() {
            camera.position -= (move_vector * self.meta.chunk_size.pow(self.meta.n_chunk_lvls as u32) as i64)
                .cast::<f32>().unwrap();

            // Shift offsets and possibly start loading new chunks
            todo!();
        }
    }

    pub fn to_virtual(self) -> VirtualMemoryGrid<VE> {
        let mut grid = vec![
            TopLevelChunk::new(
                vec![vec![None; self.meta.n_lods]; self.meta.n_chunk_lvls+1],
                vec![vec![vec![]; self.meta.n_lods]; self.meta.n_chunk_lvls+1]
            );
            cubed(self.meta.size-1)
        ];
        let mut meta = vec![vec![None; self.meta.n_lods]; self.meta.n_chunk_lvls+1];
        for (lvl, lod_layers) in self.rendering_layers.into_iter().enumerate() {
            for (lod, layer_o) in lod_layers.into_iter().enumerate() {
                match layer_o {
                    None => (),
                    Some(layer) => {
                        meta[lvl][lod] = Some(layer.meta);
                        for (chunk_i, chunk) in layer.chunks.into_iter().enumerate() {
                            // If this layer is smaller than full grid, add padding to virtual position so it
                            // is centered
                            let virtual_pos = meta[lvl][lod].unwrap().virtual_grid_pos_for_index(chunk_i).0 +
                                Vector3::<usize>::from_value(
                                    if meta[lvl][lod].unwrap().size < self.meta.size {
                                        (self.meta.size - meta[lvl][lod].unwrap().size) / 2
                                    } else { 0 }
                                );

                            grid[pos_index(virtual_pos, self.meta.size-1)].rendering_layer_data[lvl][lod] = chunk;
                        }
                    }
                }
            }
        }

        VirtualMemoryGrid::new(
            grid,
            meta,
            self.gen_func,
            MemoryGridMetadata {
                size: self.meta.size - 1,
                ..self.meta
            },
        )
    }
}
