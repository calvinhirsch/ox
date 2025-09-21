use super::lod::{
    BorrowedVoxelLODChunkEditor, VoxelLODChunkEditor, VoxelLODCreateParams, VoxelMemoryGridLOD,
};
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::renderer::component::voxels::VoxelData;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::{
    BorrowChunkForLoading, BorrowedChunk, ChunkLoadQueueItem, LayerChunkState,
};
use crate::world::mem_grid::utils::{ChunkSize, IteratorWithIndexing, VoxelPosInLod};
use crate::world::mem_grid::voxel::gpu_defs::ChunkVoxels;
use crate::world::mem_grid::voxel::lod::LODLayerDataWithVoxelIDs;
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor, MemoryGridEditorChunk};
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
    lods: [VoxelMemoryGridLOD; N],
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
    fn lod(&self, lvl: u8, sublvl: u8) -> Option<&VoxelMemoryGridLOD> {
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
        F: FnMut(&mut VoxelMemoryGridLOD) -> Vec<ChunkLoadQueueItem<()>>,
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
        self.apply_to_lods_and_queue_chunks_mut(|lod| lod.queue_load_all())
    }

    fn shift(
        &mut self,
        shift: &crate::world::mem_grid::MemGridShift,
    ) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        self.apply_to_lods_and_queue_chunks_mut(|lod| lod.shift(shift))
    }

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
    lods: [Option<VoxelLODChunkEditor<'a, VE>>; N], // When this chunk is too far away for an LOD to have data, it is `None` here
}

impl<'a, const N: usize, VE: VoxelTypeEnum>
    MemoryGridEditorChunk<'a, VoxelMemoryGrid<N>, VoxelMemoryGridMetadata>
    for ChunkVoxelEditor<'a, VE, N>
{
    fn edit_grid_with_size(
        mem_grid: &'a mut VoxelMemoryGrid<N>,
        grid_size: usize,
    ) -> MemoryGridEditor<ChunkVoxelEditor<'a, VE, N>, VoxelMemoryGridMetadata> {
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
            metadata: metadata.clone(),
        }
    }
}

unsafe impl<'a, const N: usize, VE: VoxelTypeEnum>
    BorrowChunkForLoading<BorrowedChunkVoxelEditor<VE, N>> for ChunkVoxelEditor<'a, VE, N>
{
    fn take_data_for_loading(&mut self) -> BorrowedChunkVoxelEditor<VE, N> {
        unsafe {
            self.mark_all_lods_missing();
        }
        BorrowedChunkVoxelEditor::new(self)
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

    pub unsafe fn mark_all_lods_missing(&mut self) {
        for lod_o in self.lods.iter_mut() {
            if let Some(lod) = lod_o {
                unsafe { (&mut **lod.data_mut()).set_missing() }
            }
        }
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
        if self.lods.iter().any(|lod| match lod {
            None => false,
            Some(lod) => lod.data().get().is_none(),
        }) {
            return Err(());
        }

        let mut iter = self.lods.iter_mut();
        let mut first_lod = iter
            .next()
            .unwrap()
            .as_mut()
            .expect("Tried to set_voxel in a chunk where full LOD was not loaded")
            .data_mut()
            .get_mut()
            .unwrap()
            .with_voxel_ids_mut()
            .unwrap();
        first_lod.set_voxel(index, voxel_typ);
        let first_lod: LODLayerDataWithVoxelIDs = first_lod.into();

        for lod in iter.filter_map(|x| x.as_mut()) {
            let (lvl, sublvl) = (lod.lvl(), lod.sublvl());
            let lod_pos = VoxelPosInLod {
                pos: pos.0,
                lvl: 0,
                sublvl: 0,
            }
            .in_other_lod(lvl, sublvl, meta.chunk_size);
            lod.data_mut()
                .get_mut()
                .unwrap()
                .update_voxel_from_lower_lod::<VE>(
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

    pub fn all_lods_valid(&self) -> bool {
        self.lods.iter().all(|l| {
            l.as_ref()
                .map(|l| matches!(l.data().state(), LayerChunkState::Valid))
                .unwrap_or(true)
        })
    }
}

#[derive(Getters, Debug)]
pub struct BorrowedChunkVoxelEditor<VE: VoxelTypeEnum, const N: usize> {
    #[get = "pub"]
    lods: [Option<BorrowedVoxelLODChunkEditor<VE>>; N], // When this chunk is too far away for an LOD to have data, it is `None` here
}

unsafe impl<VE: VoxelTypeEnum, const N: usize> Send for BorrowedChunkVoxelEditor<VE, N> {}

unsafe impl<VE: VoxelTypeEnum, const N: usize> BorrowedChunk for BorrowedChunkVoxelEditor<VE, N> {
    unsafe fn mark_valid(&mut self) {
        unsafe {
            self.set_all_lods_valid();
        }
    }
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
    pub unsafe fn load_new<F: Fn(TlcPos<i64>, u8, u8, &mut ChunkVoxels, usize, u8)>(
        &mut self,
        pos: TlcPos<i64>,
        lods_to_load: [bool; N],
        gen_func: F,
        chunk_size: ChunkSize,
        tlc_size: usize,
        largest_chunk_lvl: u8,
        lod_block_fill_thresh: f32,
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
            let load = lods_to_load[i];

            if let Some(lod_data) = lod {
                let lvl = lod_data.lvl();
                let sublvl = lod_data.sublvl();
                if load {
                    debug_assert!(
                        matches!(
                            unsafe { &**lod_data.data() }.state(),
                            LayerChunkState::Missing
                        ),
                        "Trying to load into already loaded chunk {:?} ({}, {})",
                        pos,
                        lod_data.lvl(),
                        lod_data.sublvl(),
                    ); // Should have been marked missing before loading

                    let data = unsafe { (&mut **lod_data.data_mut()).get_mut_for_loading() };

                    // Need to load the info in this chunk
                    if let Some(mut data) = data.with_voxel_ids_mut() {
                        if let Some(last_vox_lod) = last_vox_lod.as_ref() {
                            // Load voxels based on higher fidelity LOD that is already loaded
                            data.update_from_lower_lod_voxels::<VE>(
                                unsafe {
                                    (&**lods_to_index[last_vox_lod.index].as_ref().unwrap().data())
                                        .get_for_loading()
                                        .with_voxel_ids()
                                        .unwrap()
                                },
                                lvl,
                                sublvl,
                                last_vox_lod.lvl,
                                last_vox_lod.sublvl,
                                chunk_size,
                                largest_chunk_lvl,
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
                                largest_chunk_lvl,
                            );
                        }

                        last_vox_lod = Some(LodId {
                            lvl,
                            sublvl,
                            index: i,
                        });
                    } else {
                        // If this chunk only has a bitmask, update from previous LOD bitmask
                        data.update_bitmask_from_lower_lod(
                            unsafe {
                                (&**lods_to_index[first_bitmask_lod.as_ref().unwrap().index]
                                    .as_ref()
                                    .unwrap()
                                    .data())
                                    .get_for_loading()
                            },
                            lvl,
                            sublvl,
                            first_bitmask_lod.as_ref().unwrap().lvl,
                            first_bitmask_lod.as_ref().unwrap().sublvl,
                            chunk_size,
                            largest_chunk_lvl,
                            0.,
                        )
                    }

                    if first_bitmask_lod.is_none() {
                        first_bitmask_lod = Some(LodId {
                            lvl,
                            sublvl,
                            index: i,
                        });
                    }
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
        renderer::test_context::TestContext,
        voxel_type::{Material, VoxelTypeDefinition},
        world::mem_grid::voxel::ChunkBitmask,
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
        let (mut grid, _) = VoxelMemoryGrid::new(
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

        {
            let mut editor = ChunkVoxelEditor::<Block, 5>::edit_grid(&mut grid);
            for lod in 1..=2 {
                let chunk = editor.chunks[1640].lods[lod].as_mut().unwrap().data_mut();
                unsafe {
                    chunk.set_missing();
                    chunk.set_valid();
                }
            }
            {
                editor.chunks[1640].lods[1]
                    .as_mut()
                    .unwrap()
                    .data_mut()
                    .get_mut()
                    .unwrap()
                    .with_voxel_ids_mut()
                    .unwrap()
                    .overwrite::<Block>()
                    .voxel_ids[0] = Block::SOLID as u8;
            }
            let l1_bitmask = editor.chunks[1640].lods[1]
                .as_mut()
                .unwrap()
                .data_mut()
                .get_mut()
                .unwrap()
                .bitmask();
            let true_l1_bitmask = {
                let mut bm = ChunkBitmask::new_blank(l1_bitmask.n_voxels());
                bm.set_block_true(0);
                bm
            };
            assert_eq!(*l1_bitmask, true_l1_bitmask);

            let l2_bitmask = editor.chunks[1640].lods[2]
                .as_mut()
                .unwrap()
                .data_mut()
                .get_mut()
                .unwrap()
                .bitmask();
            let true_l2_bitmask = ChunkBitmask::new_blank(l2_bitmask.n_voxels());
            assert_eq!(*l2_bitmask, true_l2_bitmask);
        }

        assert!(grid.lods[1].chunks()[2].get().expect(
            "Chunk indexed was not valid; i.e. not indexed properly somewhere in this process",
        ).updated_bitmask_regions().regions.len() == 1);
        assert!(grid.lods[1].chunks()[2].get().unwrap().bitmask().get(0));
        assert!(!grid.lods[1].chunks()[2].get().unwrap().bitmask().get(1));
        assert!(!grid.lods[2].chunks()[2].get().unwrap().bitmask().get(0));
    }
}
