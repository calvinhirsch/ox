use super::lod::{
    BorrowedLodChunk, LodChunkEditorMaybeUnloaded, VoxelLODCreateParams, VoxelMemoryGridLod,
};
use crate::loader::{BorrowChunkForLoading, BorrowedChunk, ChunkLoadQueueItem};
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::renderer::component::voxels::VoxelData;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::mem_grid::layer::MemoryGridLayer;
use crate::world::mem_grid::utils::{ChunkSize, IteratorWithIndexing, VoxelPosInLod};
use crate::world::mem_grid::voxel::gpu_defs::ChunkVoxels;
use crate::world::mem_grid::voxel::lod::{
    update_bitmask_from_lower_lod_untracked, LodChunkDataVariant, LodChunkDataVariantMut,
    LodChunkEditorVariantMut, UpdateRegion,
};
use crate::world::mem_grid::{EditMemoryGridChunk, MemoryGrid, MemoryGridLoadChunks};
use crate::world::{TlcPos, VoxelPos};
use cgmath::{Array, EuclideanSpace, Vector3};
use getset::{CopyGetters, Getters};
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;
use unzip_array_of_tuple::unzip_array_of_tuple;
use vulkano::memory::allocator::MemoryAllocator;

#[derive(Debug, Getters)]
pub struct VoxelMemoryGrid<const N: usize> {
    #[get = "pub"]
    lods: [VoxelMemoryGridLod; N],
    #[get = "pub"]
    metadata: VoxelMemoryGridMetadata,
}

#[derive(CopyGetters, Clone, Copy, Debug)]
pub struct LodId {
    #[get_copy = "pub"]
    lvl: u8,
    #[get_copy = "pub"]
    sublvl: u8,
}

#[derive(Clone, Debug, CopyGetters)]
pub struct VoxelMemoryGridMetadata {
    #[get_copy = "pub"]
    largest_lod: LodId, // which (lvl, sublvl) has the largest grid size
    #[get_copy = "pub"]
    chunk_size: ChunkSize,
    #[get_copy = "pub"]
    lod_block_fill_thresh: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoxelChunkLoadQueueItemData<const N: usize> {
    pub lods: [bool; N],
}

impl VoxelMemoryGridMetadata {
    pub fn tlc_size(&self) -> usize {
        self.chunk_size.size().pow(self.largest_lod.lvl as u32)
            * 2usize.pow(self.largest_lod.sublvl as u32)
    }
}

/// Size (on one side) of top level chunks in units of an LOD's voxels, where the LOD is specified by lvl and sublvl.
/// This number can be cubed to get the number of voxels.
pub fn lod_tlc_size(chunk_size: ChunkSize, largest_lvl: u8, lvl: u8, sublvl: u8) -> usize {
    1usize << (chunk_size.exp() * (largest_lvl - lvl) - sublvl)
}

impl<const N: usize> VoxelMemoryGrid<N> {
    fn lod(&self, lvl: u8, sublvl: u8) -> Option<&VoxelMemoryGridLod> {
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
        chunk_size: ChunkSize,
        start_tlc: TlcPos<i64>,
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
        assert!(
            largest_sublvl == 0,
            "Largest lvl LOD (lowest fidelity) should have sublvl 0"
        );

        let (grid_lods, lods) = unzip_array_of_tuple(lod_params.map(|params| {
            let lod_tlc_size = lod_tlc_size(chunk_size, largest_lvl, params.lvl, params.sublvl);
            let start_tlc = TlcPos(
                start_tlc.0 + Vector3::from_value(((size - params.render_area_size) / 2) as i64),
            );
            VoxelMemoryGridLod::new_voxel_lod(
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
                lod_block_fill_thresh: 0.00000001,
            },
        };

        debug_assert!(
            (0..=largest_lvl)
                .map(|i| grid.lod(i, 0))
                .all(|x| x.is_some()),
            "Every tier should have an LOD at subtier 0"
        );

        (grid, VoxelData::new(lods))
    }

    pub fn get_updates(&mut self) -> [Vec<VoxelLODUpdate>; N] {
        self.lods.each_mut().map(|lod| lod.aggregate_updates(true))
    }

    fn apply_to_lods_and_queue_chunks_mut<
        F: FnMut(&mut VoxelMemoryGridLod) -> Vec<ChunkLoadQueueItem<()>>,
    >(
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

    fn largest_lod(&self) -> &VoxelMemoryGridLod {
        self.lod(
            self.metadata().largest_lod.lvl,
            self.metadata().largest_lod.sublvl,
        )
        .unwrap()
    }
}

impl<const N: usize> MemoryGridLoadChunks for VoxelMemoryGrid<N> {
    type ChunkLoadQueueItemData = VoxelChunkLoadQueueItemData<N>;

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        self.apply_to_lods_and_queue_chunks_mut(|lod| lod.queue_load_all())
    }

    fn shift(
        &mut self,
        shift: &crate::world::mem_grid::MemGridShift,
    ) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        let r = self.apply_to_lods_and_queue_chunks_mut(|lod| lod.shift(shift));
        r
    }
}

impl<const N: usize> MemoryGrid for VoxelMemoryGrid<N> {
    fn size(&self) -> usize {
        self.largest_lod().size()
    }

    fn start_tlc(&self) -> TlcPos<i64> {
        self.largest_lod().start_tlc()
    }
}

#[derive(Debug, Getters)]
pub struct ChunkVoxelEditor<'a, VE: VoxelTypeEnum, const N: usize> {
    #[getset(get = "pub")]
    lods: [Option<LodChunkEditorMaybeUnloaded<'a, VE>>; N], // When this chunk is too far away for an LOD to have data, it is `None` here
}

impl<VE: VoxelTypeEnum, const N: usize> EditMemoryGridChunk<VE> for VoxelMemoryGrid<N> {
    type ChunkEditor<'a> = ChunkVoxelEditor<'a, VE, N>
        where
            Self: 'a;

    fn edit_chunk(
        &mut self,
        pos: TlcPos<i64>,
        buffer_chunk_states: [crate::world::BufferChunkState; 3],
    ) -> Option<Self::ChunkEditor<'_>> {
        let e = ChunkVoxelEditor {
            lods: self.lods.each_mut().map(|lod| {
                <MemoryGridLayer<_, _, _> as EditMemoryGridChunk<VE>>::edit_chunk(
                    lod,
                    pos,
                    buffer_chunk_states,
                )
            }),
        };
        if e.lods.iter().all(|lod| lod.is_none()) {
            None
        } else {
            Some(e)
        }
    }
}

impl<'a, const N: usize, VE: VoxelTypeEnum>
    BorrowChunkForLoading<BorrowedChunkVoxelEditor<VE, N>, VoxelChunkLoadQueueItemData<N>>
    for ChunkVoxelEditor<'a, VE, N>
{
    fn should_still_load(&self, queue_item: &VoxelChunkLoadQueueItemData<N>) -> bool {
        for (queued_lod, lod) in queue_item.lods.iter().zip(self.lods.iter()) {
            match *queued_lod {
                true => {
                    if lod.is_none() {
                        return false;
                    }
                }
                false => {}
            }
        }
        true
    }

    fn take_data_for_loading(
        &mut self,
        queue_item: &VoxelChunkLoadQueueItemData<N>,
    ) -> BorrowedChunkVoxelEditor<VE, N> {
        for (i, (queued_lod, lod)) in queue_item.lods.iter().zip(self.lods.iter_mut()).enumerate() {
            match *queued_lod {
                true => {
                    debug_assert!(
                        lod.is_some(),
                        "Queued chunk to load with LOD {} that should be present but was not. Chunk data:\n{:?}",
                        i,
                        queue_item
                    );
                }
                // If we aren't loading a LOD, mark that LOD as none in this borrowed
                // chunk data so that we only load the correct ones.
                false => *lod = None,
            }
        }
        BorrowedChunkVoxelEditor::new(self).unwrap()
    }

    fn mark_invalid(&mut self) -> Result<(), ()> {
        unsafe { self.mark_all_lods_invalid() }
    }
}

impl<'a, VE: VoxelTypeEnum, const N: usize> ChunkVoxelEditor<'a, VE, N> {
    pub unsafe fn mark_all_lods_invalid(&mut self) -> Result<(), ()> {
        let mut r = Ok(());
        for lod_o in self.lods.iter_mut() {
            if let Some(lod) = lod_o {
                r = r.and(lod.data_mut().set_invalid());
            }
        }
        r
    }

    /// Requires that this TLC has full LOD. Requires both position and index of the voxel.
    pub fn set_voxel(
        &mut self,
        pos: VoxelPos<u32>,
        index: usize,
        voxel_typ: VE,
        meta: &VoxelMemoryGridMetadata,
    ) -> Result<(), ()> {
        // first make sure all LODs are loaded
        if self.lods.iter_mut().any(|lod| match lod {
            None => false,
            Some(lod) => lod.as_loaded().is_none(),
        }) {
            return Err(());
        }

        let mut iter = self.lods.iter_mut();
        let mut first_lod = iter
            .next()
            .unwrap()
            .as_mut()
            .expect("Tried to set_voxel in a chunk where full LOD was not loaded")
            .as_loaded()
            .unwrap();
        let mut first_lod = match first_lod.with_voxel_ids_mut() {
            LodChunkEditorVariantMut::WithVoxels(lod) => lod,
            LodChunkEditorVariantMut::WithoutVoxels(_) => panic!(),
        };
        first_lod.set_voxel(index, voxel_typ);
        let first_lod = first_lod.data();

        for lod in iter.filter_map(|x| x.as_mut()) {
            let (lvl, sublvl) = (lod.lvl(), lod.sublvl());
            let lod_pos = VoxelPosInLod {
                pos: pos.0,
                lvl: 0,
                sublvl: 0,
            }
            .in_other_lod(lvl, sublvl, meta.chunk_size);
            lod.as_loaded().unwrap().update_voxel_from_lower_lod::<VE>(
                lod_pos,
                lod_pos.index(meta.chunk_size, meta.largest_lod.lvl),
                &first_lod,
                0,
                0,
                meta.chunk_size,
                meta.largest_lod.lvl,
                meta.lod_block_fill_thresh,
            );
        }

        Ok(())
    }
}

#[derive(Getters, Debug)]
pub struct BorrowedChunkVoxelEditor<VE: VoxelTypeEnum, const N: usize> {
    #[get = "pub"]
    lods: [Option<BorrowedLodChunk<VE>>; N], // When this chunk is too far away for an LOD to have data, it is `None` here
}

impl<VE: VoxelTypeEnum, const N: usize> BorrowedChunk for BorrowedChunkVoxelEditor<VE, N> {
    type MemoryGrid = VoxelMemoryGrid<N>;

    fn return_data(self, grid: &mut VoxelMemoryGrid<N>) {
        self.queue_to_sync_to_gpu(grid);
        for (lod, editor_lod) in grid.lods.iter_mut().zip(self.lods) {
            if let Some(elod) = editor_lod {
                elod.return_data(lod);
            }
        }
    }
}

impl<VE: VoxelTypeEnum, const N: usize> BorrowedChunkVoxelEditor<VE, N> {
    pub fn new(ce: &mut ChunkVoxelEditor<VE, N>) -> Result<Self, ()> {
        let lods = ce.lods.each_mut().map(|lod_o| match lod_o.as_mut() {
            None => Ok(None),
            Some(lod) => BorrowedLodChunk::new(lod).map_or(Err(()), |e| Ok(Some(e))),
        });
        if lods.iter().any(|l| l.is_err()) {
            Err(())
        } else {
            Ok(Self {
                lods: lods.map(|l| l.unwrap()),
            })
        }
    }

    /// Load a chunk using `gen_func` to generate the voxel data where needed.
    /// Unsafe because it will access the chunk's data when it is in "missing" state.
    /// Presumably, this function is being called in a chunk loading thread having borrowed
    /// the chunk data. This will load all non-`None` LODs in `self`, so if a LOD exists
    /// but shouldn't be loaded, the reference to that LOD should be set to `None` in `self`.
    pub unsafe fn load_new<F: Fn(TlcPos<i64>, u8, u8, &mut ChunkVoxels, usize, u8)>(
        &mut self,
        pos: TlcPos<i64>,
        gen_func: F,
        metadata: &VoxelMemoryGridMetadata,
    ) {
        struct LodId {
            index: usize,
            lvl: u8,
            sublvl: u8,
        }

        // Last lvl/sublvl that contained voxel ID info
        let mut last_vox_lod: Option<LodId> = None;
        // Last lvl/sublvl that contained bitmask info
        let mut first_bitmask_lod: Option<LodId> = None;

        let len = self.lods.len();
        IteratorWithIndexing::new(&mut self.lods, len).apply(|i, lod, lods_to_index| {
            if let Some(lod_data) = lod {
                let lvl = lod_data.lvl();
                let sublvl = lod_data.sublvl();
                let data = lod_data.data_mut();

                // Need to load the info in this chunk
                match data.check_voxel_ids_mut() {
                    LodChunkDataVariantMut::WithVoxels(mut data) => {
                        if let Some(last_vox_lod) = last_vox_lod.as_ref() {
                            // Load voxels based on higher fidelity LOD that is already loaded
                            let last_vox_data = {
                                match lods_to_index[last_vox_lod.index]
                                    .as_ref()
                                    .unwrap()
                                    .data()
                                    .check_voxel_ids()
                                {
                                    LodChunkDataVariant::WithoutVoxels(_) => unreachable!(),
                                    LodChunkDataVariant::WithVoxels(data) => data,
                                }
                            };
                            data.update_from_lower_lod_voxels_untracked::<VE>(
                                last_vox_data,
                                lvl,
                                sublvl,
                                last_vox_lod.lvl,
                                last_vox_lod.sublvl,
                                metadata.chunk_size,
                                metadata.largest_lod().lvl,
                                metadata.lod_block_fill_thresh(),
                            );
                        } else {
                            // Generate voxels
                            gen_func(
                                pos,
                                lvl,
                                sublvl,
                                data.overwrite::<VE>().chunk.raw_voxel_ids_mut(),
                                metadata.tlc_size(),
                                metadata.largest_lod().lvl,
                            );
                        }

                        last_vox_lod = Some(LodId {
                            lvl,
                            sublvl,
                            index: i,
                        });
                    }
                    LodChunkDataVariantMut::WithoutVoxels(data) => {
                        // If this chunk only has a bitmask, update from previous LOD bitmask
                        update_bitmask_from_lower_lod_untracked(
                            data,
                            lods_to_index[first_bitmask_lod.as_ref().unwrap().index]
                                .as_ref()
                                .unwrap()
                                .data()
                                .bitmask(),
                            lvl,
                            sublvl,
                            first_bitmask_lod.as_ref().unwrap().lvl,
                            first_bitmask_lod.as_ref().unwrap().sublvl,
                            metadata.chunk_size(),
                            metadata.largest_lod().lvl,
                            0.,
                        )
                    }
                }

                if first_bitmask_lod.is_none() {
                    first_bitmask_lod = Some(LodId {
                        lvl,
                        sublvl,
                        index: i,
                    });
                }
            }
        });
    }

    // pub unsafe fn set_all_lods_valid(&mut self) {
    //     for lod_o in self.lods.iter_mut() {
    //         if let Some(lod) = lod_o {
    //             unsafe {
    //                 (&mut **lod.data_mut()).set_valid();
    //             }
    //         }
    //     }
    // }

    // For each LOD of this chunk, add a region to the `updated_regions` covering this
    // chunk's data so that it is sync'd to GPU
    pub fn queue_to_sync_to_gpu(&self, grid: &mut VoxelMemoryGrid<N>) {
        for (editor_lod_o, lod) in self.lods.iter().zip(grid.lods.iter_mut()) {
            if let Some(editor_lod) = editor_lod_o {
                let n_voxels = lod.metadata().extra().voxels_per_tlc;
                lod.state_mut().updated_regions.push(UpdateRegion {
                    chunk_idx: editor_lod.chunk_idx(),
                    voxel_idx: 0,
                    n_voxels,
                });
            }
        }
    }
}

/// Given a global full LOD voxel position, return the top level chunk it is
/// in and the position within that chunk.
pub fn voxel_pos_in_tlc_from_global_pos(
    global_pos: VoxelPos<i64>,
    chunk_size: ChunkSize,
    largest_chunk_lvl: u8,
) -> (TlcPos<i64>, VoxelPos<u32>) {
    let tlc_size = chunk_size.size().pow(largest_chunk_lvl as u32);
    let global_tlc = global_pos.0 / tlc_size as i64;
    (
        TlcPos(global_tlc),
        VoxelPos(
            (global_pos.0 - (global_tlc * tlc_size as i64).to_vec())
                .cast::<u32>()
                .unwrap(),
        ),
    )
}

pub fn global_voxel_pos_from_pos_in_tlc(
    tlc: TlcPos<i64>,
    pos: VoxelPos<u32>,
    chunk_size: ChunkSize,
    largest_chunk_lvl: u8,
) -> VoxelPos<i64> {
    let tlc_size = chunk_size.size().pow(largest_chunk_lvl as u32) as i64;
    VoxelPos(tlc.0 * tlc_size + pos.0.map(|a| a as i64).to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::Point3;
    use enum_iterator::Sequence;
    use num_derive::{FromPrimitive, ToPrimitive};

    use crate::{
        loader::LayerChunk,
        renderer::test_context::TestContext,
        voxel_type::{Material, VoxelTypeDefinition},
        world::{camera::Camera, mem_grid::voxel::ChunkBitmask, World},
    };

    const CHUNK_SIZE: ChunkSize = ChunkSize::new(3);

    #[derive(Debug, Sequence, Clone, Copy, FromPrimitive, ToPrimitive, PartialEq, Eq, Hash)]
    pub enum Block {
        AIR,
        SOLID,
    }

    impl VoxelTypeEnum for Block {
        type VoxelAttributes = ();

        fn def(&self) -> VoxelTypeDefinition<Self::VoxelAttributes> {
            use Block::*;
            match *self {
                AIR => VoxelTypeDefinition {
                    material: Material::default(),
                    is_visible: false,
                    attributes: (),
                },
                SOLID => VoxelTypeDefinition {
                    material: Material {
                        color: [1., 0., 0.],
                        emission_color: [1., 0., 0.],
                        emission_strength: 1.2,
                        ..Default::default()
                    },
                    is_visible: true,
                    attributes: (),
                },
            }
        }

        fn empty() -> Block {
            Block::AIR
        }
    }

    #[test]
    fn test_edit_voxel_grid() {
        let renderer_context = TestContext::new();
        let start_tlc = TlcPos(Point3::<i64> {
            x: -6,
            y: -6,
            z: -6,
        });
        let (mg, _) = VoxelMemoryGrid::new(
            [
                VoxelLODCreateParams {
                    voxel_resolution: 1,
                    lvl: 0,
                    sublvl: 0,
                    render_area_size: 1,
                    bitmask_binding: 8,
                    voxel_ids_binding: Some(4),
                },
                VoxelLODCreateParams {
                    voxel_resolution: 2,
                    lvl: 0,
                    sublvl: 1,
                    render_area_size: 3,
                    bitmask_binding: 9,
                    voxel_ids_binding: Some(5),
                },
                VoxelLODCreateParams {
                    voxel_resolution: 4,
                    lvl: 0,
                    sublvl: 2,
                    render_area_size: 7,
                    bitmask_binding: 10,
                    voxel_ids_binding: Some(6),
                },
                VoxelLODCreateParams {
                    voxel_resolution: 8,
                    lvl: 1,
                    sublvl: 0,
                    render_area_size: 15,
                    bitmask_binding: 11,
                    voxel_ids_binding: Some(7),
                },
                VoxelLODCreateParams {
                    voxel_resolution: 64,
                    lvl: 2,
                    sublvl: 0,
                    render_area_size: 15,
                    bitmask_binding: 12,
                    voxel_ids_binding: None,
                },
            ],
            Arc::clone(&renderer_context.memory_allocator) as Arc<dyn MemoryAllocator>,
            CHUNK_SIZE,
            start_tlc,
        );
        let v = 2; // this doesn't matter
        let size = mg.size();
        let mut world = World::new(mg, Camera::new(v, size), v, v as u32);

        let pos = TlcPos(Point3 { x: 2, y: 0, z: 0 });

        {
            for lod in 1..=2 {
                let mut editor = world.edit_chunk::<Block>(pos).unwrap();
                let chunk = editor.lods[lod].as_mut().unwrap().data_mut();
                **chunk = LayerChunk::new_valid(chunk.take().unwrap());
            }
            {
                match world.edit_chunk::<Block>(pos).unwrap().lods[1]
                    .as_mut()
                    .unwrap()
                    .as_loaded()
                    .unwrap()
                    .with_voxel_ids_mut()
                {
                    LodChunkEditorVariantMut::WithVoxels(mut chunk) => {
                        chunk.set_voxel(0, Block::SOLID);
                    }
                    _ => panic!(),
                }
            }
            let editor = world.edit_chunk::<Block>(pos).unwrap();
            let l1_bitmask = editor.lods[1]
                .as_ref()
                .unwrap()
                .data()
                .get()
                .unwrap()
                .bitmask();
            let true_l1_bitmask = {
                let mut bm = ChunkBitmask::new_blank(l1_bitmask.n_voxels());
                bm.set_block_true(0);
                bm
            };
            assert_eq!(*l1_bitmask, true_l1_bitmask);

            let l2_bitmask = editor.lods[2]
                .as_ref()
                .unwrap()
                .data()
                .get()
                .unwrap()
                .bitmask();
            let true_l2_bitmask = ChunkBitmask::new_blank(l2_bitmask.n_voxels());
            assert_eq!(*l2_bitmask, true_l2_bitmask);
        }

        assert!(
            world.mem_grid.lods[1].chunks()[2].get().is_some(),
            "Chunk indexed was not valid; i.e. not indexed properly somewhere in this process",
        );
        assert!(world.mem_grid.lods[1].state().updated_regions.len() == 1);
        assert!(world.mem_grid.lods[1].chunks()[2]
            .get()
            .unwrap()
            .bitmask()
            .get(0));
        assert!(!world.mem_grid.lods[1].chunks()[2]
            .get()
            .unwrap()
            .bitmask()
            .get(1));
        assert!(!world.mem_grid.lods[2].chunks()[2]
            .get()
            .unwrap()
            .bitmask()
            .get(0));
    }
}
