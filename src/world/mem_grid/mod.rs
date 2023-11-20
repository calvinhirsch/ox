use std::marker::PhantomData;
use cgmath::{EuclideanSpace, Vector3};
use vulkano::buffer::Subbuffer;
use vulkano::DeviceSize;

use crate::voxel_type::VoxelTypeEnum;
use crate::world::{
    TLCVector, TLCPos,
};

mod utils;
use utils::{pos_index, amod, pos_for_index, cubed};
mod rendering_data;
use rendering_data::{
    MemoryGridChunkRenderingData, ChunkDataWithStagingBuffer,
    gpu_defs::{BlockBitmask, VoxelTypeIDs, ChunkBitmask, ChunkVoxelIDs},
};

pub mod physical_grid;
pub mod virtual_grid;


#[derive(Clone)]
struct MemoryGridLayerMetadata {
    // TODO: Make these mutexes or something
    size: usize, // Size of grid in number of TLCs, 1 more than the rendered area size in each dimension for pre-loading
    offsets: Vector3<usize>,
    loaded_upper_chunks: bool,
}

impl MemoryGridLayerMetadata {
    fn grid_pos_for_virtual_grid_pos(&self, tlc_pos: TLCVector<usize>) -> TLCVector<usize> {
        TLCVector((tlc_pos.0 + self.offsets) % self.size)
    }

    fn index_for_virtual_grid_pos(&self, pos: TLCVector<usize>) -> usize {
        pos_index(self.grid_pos_for_virtual_grid_pos(pos).0, self.size)
    }

    fn virtual_grid_pos_for_grid_pos(&self, pos: TLCVector<usize>) -> TLCVector<usize> {
        TLCVector(amod(pos.0.cast::<i64>().unwrap() - self.offsets.cast::<i64>().unwrap(), self.size))
    }

    fn virtual_grid_pos_for_index(&self, index: usize) -> TLCVector<usize> {
        self.virtual_grid_pos_for_grid_pos(TLCVector(pos_for_index(index, self.size)))
    }
    
    fn calc_offsets(start_tlc: TLCPos<i64>, size: usize) -> Vector3<usize> {
        amod(start_tlc.0.to_vec(), size)
    }
}

trait MemoryGridChunkData {
    type VoxelType: VoxelTypeEnum;

    fn set_block(&mut self, index: usize, voxel_type: Self::VoxelType) -> Option<()>;
}

#[derive(Clone)]
struct MemoryGridLayer<D: MemoryGridChunkData> {
    chunks: Vec<Option<D>>,
    meta: MemoryGridLayerMetadata,
}

impl<VE: VoxelTypeEnum> MemoryGridLayer<MemoryGridChunkRenderingData<VE>> {
    fn new(
        meta: MemoryGridLayerMetadata,
        mut bitmask_staging_buffer: Subbuffer<[BlockBitmask]>,
        mut voxel_id_staging_buffer: Subbuffer<[VoxelTypeIDs]>,
        grid_meta: &MemoryGridMetadata,
    ) -> MemoryGridLayer<MemoryGridChunkRenderingData<VE>> {
        let mut chunks = vec![None; cubed(meta.size)];
        let bitmask_bytes_per_tlc = cubed(grid_meta.tlc_size) / 8;
        let voxel_ids_bytes_per_tlc = cubed(grid_meta.tlc_size);
        for index in 0..cubed(meta.size) {
            let (bm_sb, b) = bitmask_staging_buffer.split_at(DeviceSize::from(bitmask_bytes_per_tlc as u64));
            bitmask_staging_buffer = b;

            let (vid_sb, b) = voxel_id_staging_buffer.split_at(DeviceSize::from(voxel_ids_bytes_per_tlc as u64));
            voxel_id_staging_buffer = b;

            chunks[index] = Some(MemoryGridChunkRenderingData::<VE> {
                bitmask: ChunkDataWithStagingBuffer {
                    data: ChunkBitmask(vec![BlockBitmask { mask: 0 }; cubed(grid_meta.tlc_size) / 128]),
                    staging_buffer: bm_sb,
                },
                voxel_type_ids: ChunkDataWithStagingBuffer {
                    data: ChunkVoxelIDs(vec![VoxelTypeIDs { indices: [0u8; 128 / 8] }; cubed(grid_meta.tlc_size) * 8 / 128]),
                    staging_buffer: vid_sb,
                },
                _phantom: PhantomData,
            })
        }

        MemoryGridLayer {
            meta,
            chunks,
        }
    }
}

#[derive(Clone)]
struct MemoryGridMetadata {
    size: usize,
    chunk_size: usize,
    tlc_size: usize,
    load_thresh_dist: usize,
    n_chunk_lvls: usize,
    n_lods: usize,
    start_tlc: TLCPos<i64>,
    lod_block_fill_thresh: u8,  // up to 8
}
