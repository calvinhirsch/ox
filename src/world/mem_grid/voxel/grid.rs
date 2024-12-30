use super::lod::{
    BorrowedVoxelLODChunkEditor, VoxelLODChunkEditor, VoxelLODCreateParams, VoxelMemoryGridLOD,
};
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::renderer::component::voxels::VoxelData;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::{ChunkLoadQueueItem, LayerChunkState};
use crate::world::mem_grid::utils::{cubed, index_for_pos_in_tlc, IteratorWithIndexing};
use crate::world::mem_grid::voxel::gpu_defs::ChunkVoxels;
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor, MemoryGridEditorChunk};
use crate::world::{TLCPos, TLCVector, VoxelPos};
use cgmath::{Array, EuclideanSpace, Vector3};
use getset::{CopyGetters, Getters};
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;
use unzip_array_of_tuple::unzip_array_of_tuple;
use vulkano::memory::allocator::MemoryAllocator;

#[derive(Debug, Getters)]
pub struct VoxelMemoryGrid<const N: usize> {
    lods: [VoxelMemoryGridLOD; N],
    #[get = "pub"]
    metadata: VoxelMemoryGridMetadata,
}

#[derive(Clone, Copy, Debug)]
pub struct LodId {
    lvl: usize,
    sublvl: usize,
}

#[derive(Clone, Debug, CopyGetters)]
pub struct VoxelMemoryGridMetadata {
    #[get_copy = "pub"]
    largest_lod: LodId, // which (lvl, sublvl) has the largest grid size
    #[get_copy = "pub"]
    chunk_size: usize,
    #[get_copy = "pub"]
    n_lvls: usize,
    #[get_copy = "pub"]
    n_sublvls: usize,
    #[get_copy = "pub"]
    lod_block_fill_thresh: f32,
}

#[derive(Clone, Debug)]
pub struct VoxelChunkLoadQueueItemData<const N: usize> {
    pub lods: [bool; N],
}

impl VoxelMemoryGridMetadata {
    pub fn tlc_size(&self) -> usize {
        self.chunk_size.pow(self.largest_lod.lvl as u32)
            * 2usize.pow(self.largest_lod.sublvl as u32)
    }
}

impl<const N: usize> VoxelMemoryGrid<N> {
    /// Size of top level chunks in units of an LOD's voxels, where the LOD is specified by lvl and sublvl
    fn lod_tlc_size(chunk_size: usize, n_lvls: usize, lvl: usize, sublvl: usize) -> usize {
        chunk_size.pow((n_lvls - lvl) as u32).to_le() >> sublvl
    }

    fn lod(&self, lvl: usize, sublvl: usize) -> Option<&VoxelMemoryGridLOD> {
        self.lods
            .iter()
            .filter(|lod| {
                lod.metadata().extra().lvl == lvl && lod.metadata().extra().sublvl == sublvl
            })
            .next()
    }

    pub fn new(
        lod_params: [VoxelLODCreateParams; N],
        memory_allocator: Arc<dyn MemoryAllocator>,
        chunk_size: usize,
        start_tlc: TLCPos<i64>,
        lod_block_fill_thresh: f32,
    ) -> (Self, VoxelData<N>) {
        for p in lod_params.iter() {
            p.validate(chunk_size);
        }
        debug_assert!(
            lod_params.len()
                == lod_params
                    .iter()
                    .map(|lod| (lod.lvl, lod.sublvl))
                    .collect::<HashSet<_>>()
                    .len(),
            "LOD params contained duplicate LODs (lvl and sublvl are the same)"
        );

        let (largest_lvl, largest_sublvl, size) = lod_params
            .iter()
            .map(|lod| (lod.lvl, lod.sublvl, lod.render_area_size))
            .max()
            .unwrap();
        let n_lvls = largest_lvl + 1;
        let n_sublvls = (chunk_size.ilog2() + 1) as usize;

        let (grid_lods, lods) = unzip_array_of_tuple(lod_params.map(|params| {
            let lod_tlc_size = Self::lod_tlc_size(chunk_size, n_lvls, params.lvl, params.sublvl);
            let start_tlc = TLCPos(
                start_tlc.0 + Vector3::from_value(((size - params.render_area_size) / 2) as i64),
            );
            VoxelMemoryGridLOD::new_voxel_lod(
                params,
                start_tlc,
                lod_tlc_size,
                Arc::clone(&memory_allocator),
            )
        }));

        let grid = VoxelMemoryGrid {
            lods: grid_lods,
            metadata: VoxelMemoryGridMetadata {
                largest_lod: LodId {
                    lvl: largest_lvl,
                    sublvl: largest_sublvl,
                },
                chunk_size,
                n_lvls,
                n_sublvls,
                lod_block_fill_thresh,
            },
        };

        debug_assert!(
            (0..n_lvls).map(|i| grid.lod(i, 0)).all(|x| x.is_some()),
            "Every tier should have an LOD at subtier 0"
        );

        (grid, VoxelData::new(lods))
    }

    pub fn get_updates(&mut self) -> [Vec<VoxelLODUpdate>; N] {
        self.lods.each_mut().map(|lod| lod.aggregate_updates(true))
    }

    fn apply_and_queue<F: FnMut(&mut VoxelMemoryGridLOD) -> Vec<ChunkLoadQueueItem<()>>>(
        &mut self,
        mut to_apply: F,
    ) -> Vec<ChunkLoadQueueItem<VoxelChunkLoadQueueItemData<N>>> {
        let mut chunks = HashMap::new();

        for (i, lod) in self.lods.iter_mut().enumerate() {
            for item in to_apply(lod) {
                let e = chunks.entry(item.pos.0).or_insert(ChunkLoadQueueItem {
                    pos: item.pos,
                    data: VoxelChunkLoadQueueItemData { lods: [false; N] },
                });
                e.data.lods[i] = true;
            }
        }

        chunks.into_values().collect()
    }

    fn largest_lod(&self) -> &VoxelMemoryGridLOD {
        self.lod(
            self.metadata().largest_lod.lvl,
            self.metadata().largest_lod.sublvl,
        )
        .unwrap()
    }
}

impl<const N: usize> MemoryGrid for VoxelMemoryGrid<N> {
    type ChunkLoadQueueItemData = VoxelChunkLoadQueueItemData<N>;

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        self.apply_and_queue(|lod| lod.queue_load_all())
    }

    fn shift(
        &mut self,
        shift: TLCVector<i32>,
        load_in_from_edge: TLCVector<i32>,
        load_buffer: [bool; 3],
    ) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        self.apply_and_queue(|lod| lod.shift(shift, load_in_from_edge, load_buffer))
    }

    fn size(&self) -> usize {
        self.largest_lod().size()
    }
    fn start_tlc(&self) -> TLCPos<i64> {
        self.largest_lod().start_tlc()
    }
}

impl<'a, const N: usize, VE: VoxelTypeEnum>
    MemoryGridEditorChunk<'a, VoxelMemoryGrid<N>, &'a mut VoxelMemoryGridMetadata>
    for ChunkVoxelEditor<'a, VE, N>
{
    fn edit_grid_with_size(
        mem_grid: &'a mut VoxelMemoryGrid<N>,
        grid_size: usize,
    ) -> MemoryGridEditor<ChunkVoxelEditor<'a, VE, N>, &'a mut VoxelMemoryGridMetadata> {
        let start_tlc = mem_grid.start_tlc();
        let VoxelMemoryGrid { lods, metadata } = mem_grid;
        MemoryGridEditor {
            chunks: crate::util::zip(lods.each_mut().map(|lod| {
                Option::<VoxelLODChunkEditor<_>>::edit_grid_with_size(lod, grid_size)
                    .chunks
                    .into_iter()
            }))
            .map(|chunk_lods| ChunkVoxelEditor { lods: chunk_lods })
            .collect(),
            size: grid_size,
            start_tlc,
            metadata,
        }
    }
}

#[derive(Debug)]
pub struct ChunkVoxelEditor<'a, VE: VoxelTypeEnum, const N: usize> {
    lods: [Option<VoxelLODChunkEditor<'a, VE>>; N], // When this chunk is too far away for an LOD to have data, it is `None` here
}

pub enum SetVoxelErr {
    LODDoesNotExist,
    LODNotLoaded,
    LODVoxelsNotLoaded,
}

impl<'a, VE: VoxelTypeEnum, const N: usize> ChunkVoxelEditor<'a, VE, N> {
    pub unsafe fn mark_all_lods_invalid(&mut self) -> Result<(), ()> {
        let mut r = Ok(());
        for lod_o in self.lods.iter_mut() {
            if let Some(lod) = lod_o {
                r = r.or(lod.data_mut().set_invalid());
            }
        }
        r
    }

    pub fn set_voxel(
        &mut self,
        index_in_tlc: usize,
        voxel_typ: VE,
        meta: &VoxelMemoryGridMetadata,
    ) {
        for lod in self.lods.iter_mut().filter_map(|x| x.as_mut()) {
            let (lvl, sublvl) = (lod.lvl(), lod.sublvl());
            if let Some(mut data) = lod
                .data_mut()
                .get_mut()
                .map(|d| d.with_voxel_ids_mut())
                .flatten()
            {
                let block_size = meta.chunk_size.pow(lvl as u32) * 2usize.pow(sublvl as u32);
                data.set_voxel(index_in_tlc / cubed(block_size), voxel_typ);
            }
        }
    }

    pub fn set_voxel_pos(
        &mut self,
        pos: VoxelPos<u32>,
        voxel_typ: VE,
        meta: &VoxelMemoryGridMetadata,
    ) {
        self.set_voxel(
            index_for_pos_in_tlc(pos.0, meta.chunk_size, meta.n_lvls, 0, 0),
            voxel_typ,
            meta,
        );
    }

    pub fn no_missing_lods(&self) -> bool {
        for lod in self.lods.iter() {
            if let Some(lod) = lod {
                if matches!(lod.data().state(), LayerChunkState::Missing) {
                    return false;
                }
            }
        }
        true
    }
}

#[derive(Debug)]
pub struct BorrowedChunkVoxelEditor<VE: VoxelTypeEnum, const N: usize> {
    lods: [Option<BorrowedVoxelLODChunkEditor<VE>>; N], // When this chunk is too far away for an LOD to have data, it is `None` here
}

impl<VE: VoxelTypeEnum, const N: usize> BorrowedChunkVoxelEditor<VE, N> {
    pub fn new(ce: &mut ChunkVoxelEditor<VE, N>) -> Self {
        Self {
            lods: ce.lods.each_mut().map(|lod_o| {
                lod_o
                    .as_mut()
                    .map(|lod| BorrowedVoxelLODChunkEditor::new(lod))
            }),
        }
    }

    /// Load a chunk using `gen_func` to generate the voxel data where needed.
    /// Unsafe because it will access the chunk's data when it is in "missing" state.
    /// Presumably, this function is being called in a chunk loading thread having borrowed
    /// the chunk data.
    pub unsafe fn load_new<F: Fn(TLCPos<i64>, usize, usize, &mut ChunkVoxels, usize, usize)>(
        &mut self,
        pos: TLCPos<i64>,
        lods_to_load: [bool; N],
        gen_func: F,
        chunk_size: usize,
        tlc_size: usize,
        n_chunk_lvls: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        // Last lvl/sublvl that contained voxel ID info
        let (mut last_vox_lvl, mut last_vox_sublvl, mut last_vox_idx) = (None, None, None);
        // Last lvl/sublvl that contained bitmask info
        let (mut last_bitmask_lvl, mut last_bitmask_sublvl, mut last_bitmask_idx) =
            (None, None, None);

        let len = self.lods.len();
        IteratorWithIndexing::new(&mut self.lods, len).apply(|i, lod, lods_to_index| {
            let load = lods_to_load[i];

            if let Some(lod_data) = lod {
                let lvl = lod_data.lvl() as usize;
                let sublvl = lod_data.sublvl() as usize;
                if load {
                    debug_assert!(
                        matches!(
                            unsafe { &**lod_data.data() }.state(),
                            LayerChunkState::Invalid
                        ),
                        "Trying to load into already loaded chunk {:?} ({}, {})",
                        pos,
                        lod_data.lvl(),
                        lod_data.sublvl(),
                    ); // Should have been marked invalid before loading

                    let data = unsafe { (&mut **lod_data.data_mut()).get_mut_for_loading() };

                    // Need to load the info in this chunk
                    if let Some(mut data) = data.with_voxel_ids_mut() {
                        if let (Some(l_lvl), Some(l_sublvl), Some(l_i)) =
                            (last_vox_lvl, last_vox_sublvl, last_vox_idx)
                        {
                            // Load voxels based on higher fidelity LOD that is already loaded
                            data.calc_from_lower_lod_voxels::<VE>(
                                unsafe {
                                    (&**lods_to_index[l_i].as_ref().unwrap().data())
                                        .get_for_loading()
                                        .with_voxel_ids()
                                        .unwrap()
                                },
                                lvl,
                                sublvl,
                                l_lvl,
                                l_sublvl,
                                chunk_size,
                                n_chunk_lvls,
                                n_lods,
                                lod_block_fill_thresh,
                            );
                        } else {
                            // Generate voxels
                            gen_func(
                                pos,
                                lvl,
                                sublvl,
                                data.overwrite::<VE>().voxel_ids,
                                tlc_size,
                                n_chunk_lvls,
                            );
                        }

                        last_vox_lvl = Some(lvl);
                        last_vox_sublvl = Some(sublvl);
                        last_vox_idx = Some(i);
                    } else {
                        // If this chunk only has a bitmask, update from previous LOD bitmask
                        data.update_bitmask_from_lower_lod(
                            unsafe {
                                (&**lods_to_index[last_bitmask_idx.unwrap()]
                                    .as_ref()
                                    .unwrap()
                                    .data())
                                    .get_for_loading()
                            },
                            lvl,
                            sublvl,
                            last_bitmask_lvl.unwrap(),
                            last_bitmask_sublvl.unwrap(),
                            chunk_size,
                            n_chunk_lvls,
                            n_lods,
                            lod_block_fill_thresh,
                        )
                    }

                    last_bitmask_lvl = Some(lvl);
                    last_bitmask_sublvl = Some(sublvl);
                    last_bitmask_idx = Some(i);
                }
            }
        });
    }

    pub unsafe fn set_all_lods_valid(&mut self) {
        for lod_o in self.lods.iter_mut() {
            if let Some(lod) = lod_o {
                unsafe {
                    (&mut **lod.data_mut()).set_valid();
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct GlobalVoxelPos {
    pub tlc: TLCPos<i64>,
    pub voxel_index: usize,
}
impl GlobalVoxelPos {
    pub fn new(global_pos: VoxelPos<i64>, chunk_size: usize, n_chunk_lvls: usize) -> Self {
        let tlc_size = chunk_size.pow(n_chunk_lvls as u32);
        let global_tlc = global_pos.0 / tlc_size as i64;
        let pos_in_tlc = (global_pos.0 - (global_tlc * tlc_size as i64).to_vec())
            .cast::<u32>()
            .unwrap();

        GlobalVoxelPos {
            tlc: TLCPos(global_tlc),
            voxel_index: index_for_pos_in_tlc(pos_in_tlc, chunk_size, n_chunk_lvls, 0, 0),
        }
    }
}
