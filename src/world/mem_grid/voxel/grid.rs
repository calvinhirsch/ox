use super::lod::{VoxelLODChunkEditor, VoxelLODCreateParams, VoxelMemoryGridLOD};
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::renderer::component::voxels::VoxelData;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::ChunkLoadQueueItem;
use crate::world::mem_grid::utils::{cubed, index_for_pos_in_tlc, IteratorWithIndexing};
use crate::world::mem_grid::voxel::gpu_defs::ChunkVoxels;
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor, MemoryGridEditorChunk};
use crate::world::{TLCPos, TLCVector, VoxelPos};
use cgmath::{Array, EuclideanSpace, Vector3};
use getset::Getters;
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

#[derive(Clone, Debug, Getters)]
pub struct VoxelMemoryGridMetadata {
    #[getset(get_copy = "pub")]
    largest_lod: LodId, // which (lvl, sublvl) has the largest grid size
    #[get = "pub"]
    chunk_size: usize,
    #[get = "pub"]
    n_lvls: usize,
    #[get = "pub"]
    n_sublvls: usize,
    #[get = "pub"]
    lod_block_fill_thresh: f32,
}

#[derive(Clone, Debug)]
pub struct VoxelChunkLoadQueueItemData<const N: usize> {
    pub lods: [bool; N],
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
        lod_params.iter().map(|p| p.validate(chunk_size));
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
            .max_by_key(|x| x)
            .unwrap();
        let n_lvls = largest_lvl + 1;
        let n_sublvls = chunk_size.log2() + 1;

        let (grid_lods, lods) = unzip_array_of_tuple(lod_params.map(|params| {
            VoxelMemoryGridLOD::new_voxel_lod(
                params,
                TLCPos(
                    start_tlc.0
                        + Vector3::from_value(((size - params.render_area_size) / 2) as i64),
                ),
                Self::lod_tlc_size(chunk_size, n_lvls, params.lvl, params.sublvl),
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

    pub fn get_updates(&mut self) -> Vec<Vec<VoxelLODUpdate>> {
        self.lods
            .iter_mut()
            .map(|lod| lod.aggregate_updates(true))
            .collect()
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
    MemoryGridEditorChunk<'a, VoxelMemoryGrid<N>, VoxelMemoryGridMetadata>
    for ChunkVoxelEditor<'a, VE>
{
    fn edit_grid_with_size(
        mem_grid: &'a mut VoxelMemoryGrid<N>,
        grid_size: usize,
    ) -> MemoryGridEditor<ChunkVoxelEditor<'a, VE>, VoxelMemoryGridMetadata> {
        let start_tlc = mem_grid.start_tlc();
        let VoxelMemoryGrid { lods, metadata } = mem_grid;
        MemoryGridEditor {
            chunks: crate::util::zip(
                lods.into_iter()
                    .map(|lod| {
                        VoxelLODChunkEditor::edit_grid_with_size(lod, grid_size)
                            .chunks
                            .into_iter()
                    })
                    .collect::<[std::vec::IntoIter<_>; N]>(),
            )
            .collect(),
            size: grid_size,
            start_tlc,
            metadata,
        }
    }
}

struct LODSplitter<I: Iterator>(Vec<Vec<Option<I>>>);

impl<I: Iterator> Iterator for LODSplitter<I> {
    type Item = Vec<Vec<Option<I::Item>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_items: Vec<Vec<_>> = self
            .0
            .iter_mut()
            .map(|iters| {
                iters
                    .iter_mut()
                    .map(|iter| iter.as_mut().map(Iterator::next))
                    .collect()
            })
            .collect();

        // If any of the iterators are Some (meaning the LOD exists) but the item in it is None, then stop iterating
        if next_items
            .iter()
            .any(|iters| iters.iter().any(|iter| matches!(iter, Some(None))))
        {
            // If one iterator is depleted, make sure all others are too
            debug_assert!(
                next_items.iter().any(|iters| iters
                    .iter()
                    .any(|iter| matches!(iter, Some(None)) || iter.is_none())),
                "LOD chunk iterators passed to LODSplitter have different numbers of chunks."
            );
            return None;
        }

        Some(
            next_items
                .into_iter()
                .map(|items| {
                    items
                        .into_iter()
                        .map(|item| item.map(|i| i.unwrap()))
                        .collect()
                })
                .collect(),
        )
    }
}

#[derive(Debug)]
pub struct ChunkVoxelEditor<'a, VE: VoxelTypeEnum> {
    lods: Vec<Vec<Option<VoxelLODChunkEditor<'a, VE>>>>,
}

pub enum SetVoxelErr {
    LODDoesNotExist,
    LODNotLoaded,
    LODVoxelsNotLoaded,
}

impl<'a, VE: VoxelTypeEnum> ChunkVoxelEditor<'a, VE> {
    pub fn load_new<F: Fn(TLCPos<i64>, usize, usize, &mut ChunkVoxels)>(
        &'a mut self,
        pos: TLCPos<i64>,
        lods_to_load: Vec<Vec<bool>>,
        gen_func: F,
        chunk_size: usize,
        n_chunk_lvls: usize,
        n_lods: usize,
        lod_block_fill_thresh: f32,
    ) {
        println!("\npos {:?}", pos);
        // Last lvl/LOD that contained voxel ID info
        let (mut last_vox_lvl, mut last_vox_lod) = (None, None);
        // Last lvl/LOD that contained bitmask info
        let (mut last_bitmask_lvl, mut last_bitmask_lod) = (None, None);

        let lods: Vec<_> = self.lods.iter_mut().flatten().collect();

        let len = lods.len();
        IteratorWithIndexing::new(lods, len).apply(|i, lod, lods_to_index| {
            let lvl_i = i / n_lods;
            let lod_i = i % n_lods;
            let load = lods_to_load[lvl_i][lod_i];

            if let Some(lod_data) = lod {
                if load {
                    debug_assert!(
                        !lod_data.bitmask().loaded,
                        "Trying to load into already loaded chunk {:?} ({}, {})",
                        pos,
                        lvl_i,
                        lod_i
                    ); // Should have been marked not loaded before loading
                    lod_data.set_loaded();

                    // Need to load the info in this chunk
                    if lod_data.has_voxel_ids() {
                        if let (Some(l_lvl), Some(l_lod)) = (last_vox_lvl, last_vox_lod) {
                            // Load voxels based on higher fidelity LOD that is already loaded
                            (*lod_data).calc_from_lower_lod_voxels(
                                lods_to_index[l_lvl * n_lods + l_lod]
                                    .as_ref()
                                    .unwrap()
                                    .voxels()
                                    .as_ref()
                                    .unwrap(),
                                lvl_i,
                                lod_i,
                                l_lvl,
                                l_lod,
                                chunk_size,
                                n_chunk_lvls,
                                n_lods,
                                lod_block_fill_thresh,
                            );
                        } else {
                            // Generate voxels
                            gen_func(pos, lvl_i, lod_i, lod_data.overwrite().voxels);
                        }

                        last_vox_lvl = Some(lvl_i);
                        last_vox_lod = Some(lod_i);
                    } else {
                        // If this chunk only has a bitmask, update from previous LOD bitmask
                        lod_data.update_bitmask_from_lower_lod(
                            lods_to_index
                                [last_bitmask_lvl.unwrap() * n_lods + last_bitmask_lod.unwrap()]
                            .as_ref()
                            .unwrap()
                            .bitmask(),
                            lvl_i,
                            lod_i,
                            last_bitmask_lvl.unwrap(),
                            last_bitmask_lod.unwrap(),
                            chunk_size,
                            n_chunk_lvls,
                            n_lods,
                            lod_block_fill_thresh,
                        )
                    }

                    last_bitmask_lvl = Some(lvl_i);
                    last_bitmask_lod = Some(lod_i);
                } else if lod_data.bitmask().loaded {
                    last_bitmask_lod = Some(lod_i);
                    last_bitmask_lvl = Some(lvl_i);

                    // Note: voxels should only ever be loaded when the bitmask is
                    if let Some(true) = lod_data.voxels().as_ref().map(|v| v.loaded) {
                        last_vox_lvl = Some(lvl_i);
                        last_vox_lod = Some(lod_i);
                    }
                }
            }
        });
    }

    pub fn set_voxel(
        &mut self,
        index_in_tlc: usize,
        voxel_typ: VE,
        meta: &VoxelMemoryGridMetadata,
    ) {
        for lvl in 0..meta.n_lvls {
            for lod in 0..meta.n_lods {
                if let Some(data) = self.lods[lvl][lod].as_mut() {
                    if let Some(voxels) = data.voxels().as_ref() {
                        if voxels.loaded {
                            let block_size =
                                meta.chunk_size.pow(lvl as u32) * 2usize.pow(lod as u32);
                            data.set_voxel(index_in_tlc / cubed(block_size), voxel_typ);
                        }
                    }
                }
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
