use std::collections::HashMap;
use std::marker::PhantomData;
use vulkano::buffer::{BufferContents, Subbuffer};

pub mod gpu_defs;

use gpu_defs::{BufferContentsVec, ChunkBitmask, ChunkVoxelIDs, BlockBitmask, VoxelTypeIDs};
use crate::{
    world::{
        mem_grid::{
            MemoryGridMetadata, MemoryGridChunkData,
            utils::{cubed},
        }
    },
    voxel_type::{VoxelTypeEnum},
};


#[derive(Clone)]
pub struct ChunkDataWithStagingBuffer<D: BufferContents, V: BufferContentsVec<Contents = D>> {
    pub data: V,
    pub staging_buffer: Subbuffer<[D]>,
}

#[derive(Clone)]
pub struct MemoryGridChunkRenderingData<VE: VoxelTypeEnum> {
    pub bitmask: ChunkDataWithStagingBuffer<BlockBitmask, ChunkBitmask>,
    pub voxel_type_ids: ChunkDataWithStagingBuffer<VoxelTypeIDs, ChunkVoxelIDs>,
    pub _phantom: PhantomData<VE>,  // temporary until VoxelTypeIDs is generic
}

impl<VE: VoxelTypeEnum> MemoryGridChunkRenderingData<VE> {
    fn set_from_voxels(
        &mut self,
        voxels: ChunkVoxelIDs,
        lvl: usize,
        lod: usize,
        grid_meta: &MemoryGridMetadata,
    ) {
        self.voxel_type_ids.data = voxels;
        self.update_full_bitmask(lvl, lod, grid_meta);
    }

    fn calc_from_lower_lod_voxels(
        &mut self,
        voxels: &ChunkVoxelIDs,
        curr_lod: usize,
        curr_lvl: usize,
        grid_meta: &MemoryGridMetadata,
    ) {
        let (lower_lod, lower_lvl) = if curr_lod == 0 { (grid_meta.n_lods - 1, curr_lvl - 1) } else { (curr_lod - 1, curr_lvl) };
        let lower_lod_z_incr = grid_meta.chunk_size >> lower_lod;
        let lower_lod_y_incr = lower_lod_z_incr*lower_lod_z_incr;

        for vox_index in 0..cubed((grid_meta.chunk_size*(grid_meta.n_chunk_lvls - curr_lvl)).to_le() >> curr_lod) {
            // Index of the lower corner of the 2x2x2 area in the lower LOD data we want to look at
            let lower_lod_index = vox_index.to_le() << 3;
            let mut visible_count = 0;
            let mut type_counts = HashMap::<u8, u8>::new();

            let voxels_to_check = [
                voxels[lower_lod_index],
                voxels[lower_lod_index + 1],
                voxels[lower_lod_index + lower_lod_z_incr],
                voxels[lower_lod_index + 1 + lower_lod_z_incr],
                voxels[lower_lod_index + lower_lod_y_incr],
                voxels[lower_lod_index + 1 + lower_lod_y_incr],
                voxels[lower_lod_index + lower_lod_z_incr + lower_lod_y_incr],
                voxels[lower_lod_index + 1 + lower_lod_z_incr + lower_lod_y_incr],
            ];
            for vox in voxels_to_check {
                let vox_type = VE::try_from(vox).unwrap();
                if vox_type.def().is_visible {
                    visible_count += 1;
                    match type_counts.get_mut(&vox) {
                        None => { type_counts.insert(vox, 1); },
                        Some(c) => { *c += 1; },
                    }
                }
            }

            if visible_count >= grid_meta.lod_block_fill_thresh {
                self.voxel_type_ids.data[vox_index] = type_counts.into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1)).map(|(k, v)| k).unwrap();
            }
            else {
                self.voxel_type_ids.data[vox_index] = 0;
            }
        }

        // let mut index: Vec<usize> = vec![0; grid_meta.n_chunk_lvls - curr_lvl + 1];
        // let mut lod_per_lvl = vec![0; grid_meta.n_chunk_lvls - curr_lvl + 1];
        // lod_per_lvl[0] = curr_lod;
        // while index[grid_meta.n_chunk_lvls - curr_lvl] < cubed(grid_meta.chunk_size >> lod_per_lvl[grid_meta.n_chunk_lvls - curr_lvl]) {
        //     let mut vox_index = 0;
        //     for lvl in (0..=grid_meta.n_chunk_lvls - curr_lvl).rev() {
        //         vox_index *= cubed(grid_meta.chunk_size >> lod_per_lvl[lvl]);
        //         vox_index += index[lvl];
        //     }
        //
        //     // Index of the lower corner of the 2x2x2 area in the lower LOD data we want to look at
        //     let lower_lod_index = vox_index.to_le() << 3;
        //     let mut visible_count = 0;
        //     let mut type_counts = HashMap::<u8, u8>::new();
        //
        //     let voxels_to_check = [
        //         voxels[lower_lod_index],
        //         voxels[lower_lod_index + 1],
        //         voxels[lower_lod_index + lower_lod_z_incr],
        //         voxels[lower_lod_index + 1 + lower_lod_z_incr],
        //         voxels[lower_lod_index + lower_lod_y_incr],
        //         voxels[lower_lod_index + 1 + lower_lod_y_incr],
        //         voxels[lower_lod_index + lower_lod_z_incr + lower_lod_y_incr],
        //         voxels[lower_lod_index + 1 + lower_lod_z_incr + lower_lod_y_incr],
        //     ];
        //     for vox in voxels_to_check {
        //         let vox_type = VE::try_from(vox).unwrap();
        //         if vox_type.def().is_visible {
        //             visible_count += 1;
        //             match type_counts.get_mut(&vox) {
        //                 None => { type_counts.insert(vox, 1); },
        //                 Some(c) => { *c += 1; },
        //             }
        //         }
        //     }
        //
        //     if visible_count >= grid_meta.lod_block_fill_thresh {
        //         self.voxel_type_ids.data[vox_index] = type_counts.into_iter()
        //             .max_by(|a, b| a.1.cmp(&b.1)).map(|(k, v)| k).unwrap();
        //     }
        //     else {
        //         self.voxel_type_ids.data[vox_index] = 0;
        //     }
        //
        //     index[0] += 1;
        //     let mut lvl = 0;
        //     while index[lvl] > cubed(grid_meta.chunk_size >> lod_per_lvl[lvl]) && lvl <= grid_meta.n_chunk_lvls - curr_lvl {
        //         index[lvl] = 0;
        //         index[lvl+1] += 1;
        //
        //         lvl += 1;
        //     }
        // }

        self.update_full_bitmask(curr_lvl, curr_lod, grid_meta);
    }

    fn update_full_bitmask(&mut self, lvl: usize, lod: usize, grid_meta: &MemoryGridMetadata) {
        for vox_index in 0..cubed((grid_meta.chunk_size*(grid_meta.n_chunk_lvls - lvl)).to_le() >> lod) {
            if VE::try_from(self.voxel_type_ids.data[vox_index]).unwrap().def().is_visible {
                self.bitmask.data.set_block_true(vox_index);
            }
            else {
                self.bitmask.data.set_block_false(vox_index);
            }
        }
    }
}

impl<VE: VoxelTypeEnum> MemoryGridChunkData for MemoryGridChunkRenderingData<VE> {
    type VoxelType = VE;

    fn set_block(&mut self, index: usize, voxel_type: VE) -> Option<()> {
        // Set bitmask
        if voxel_type.def().is_visible {
            self.bitmask.data.set_block_true(index);
        }
        else {
            self.bitmask.data.set_block_false(index);
        }

        // Set voxel type IDs
        self.voxel_type_ids.data[index] = voxel_type.into().to_le();

        Some(())
    }
}