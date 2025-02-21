use super::lod::{
    BorrowedVoxelLODChunkEditor, VoxelLODChunkEditor, VoxelLODCreateParams, VoxelMemoryGridLOD,
};
use crate::renderer::component::voxels::lod::VoxelLODUpdate;
use crate::renderer::component::voxels::VoxelData;
use crate::voxel_type::VoxelTypeEnum;
use crate::world::loader::{
    BorrowChunkForLoading, BorrowedChunk, ChunkLoadQueueItem, LayerChunkState,
};
use crate::world::mem_grid::utils::{cubed, ChunkSize, IteratorWithIndexing, VoxelPosInLOD};
use crate::world::mem_grid::voxel::gpu_defs::ChunkVoxels;
use crate::world::mem_grid::{MemoryGrid, MemoryGridEditor, MemoryGridEditorChunk};
use crate::world::{TLCPos, VoxelPos};
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
        assert!(
            largest_sublvl == 0,
            "Largest lvl LOD (lowest fidelity) should have sublvl 0"
        );

        let (grid_lods, lods) = unzip_array_of_tuple(lod_params.map(|params| {
            let lod_tlc_size = lod_tlc_size(chunk_size, largest_lvl, params.lvl, params.sublvl);
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
                lod_block_fill_thresh,
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

    fn apply_to_lods_and_queue_chunks<
        F: FnMut(&VoxelMemoryGridLOD) -> Vec<ChunkLoadQueueItem<()>>,
    >(
        &self,
        mut to_apply: F,
    ) -> Vec<ChunkLoadQueueItem<VoxelChunkLoadQueueItemData<N>>> {
        let mut chunks = HashMap::new();

        for (i, lod) in self.lods.iter().enumerate() {
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

    fn load_buffer_chunks(
        &self,
        cfg: &crate::world::mem_grid::LoadBufferChunks,
    ) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        self.apply_to_lods_and_queue_chunks(|lod| lod.load_buffer_chunks(cfg))
    }

    fn size(&self) -> usize {
        self.largest_lod().size()
    }

    fn start_tlc(&self) -> TLCPos<i64> {
        self.largest_lod().start_tlc()
    }
}

#[derive(Debug)]
pub struct ChunkVoxelEditor<'a, VE: VoxelTypeEnum, const N: usize> {
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
                let block_size = meta.chunk_size.size().pow(lvl as u32) * 2usize.pow(sublvl as u32);
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
            VoxelPosInLOD::in_full_lod(pos).index(meta.chunk_size, meta.largest_lod.lvl),
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

    pub fn all_lods_valid(&self) -> bool {
        self.lods.iter().all(|l| {
            l.as_ref()
                .map(|l| matches!(l.data().state(), LayerChunkState::Valid))
                .unwrap_or(true)
        })
    }
}

#[derive(Debug)]
pub struct BorrowedChunkVoxelEditor<VE: VoxelTypeEnum, const N: usize> {
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
    pub unsafe fn load_new<F: Fn(TLCPos<i64>, u8, u8, &mut ChunkVoxels, usize, u8)>(
        &mut self,
        pos: TLCPos<i64>,
        lods_to_load: [bool; N],
        gen_func: F,
        chunk_size: ChunkSize,
        tlc_size: usize,
        largest_chunk_lvl: u8,
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
                            largest_chunk_lvl,
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

#[derive(Debug, Clone, Copy)]
pub struct GlobalVoxelPos {
    pub tlc: TLCPos<i64>,
    pub voxel_index: usize,
}
impl GlobalVoxelPos {
    pub fn new(global_pos: VoxelPos<i64>, chunk_size: ChunkSize, largest_chunk_lvl: u8) -> Self {
        let tlc_size = chunk_size.size().pow(largest_chunk_lvl as u32);
        let global_tlc = global_pos.0 / tlc_size as i64;
        let pos_in_tlc = (global_pos.0 - (global_tlc * tlc_size as i64).to_vec())
            .cast::<u32>()
            .unwrap();

        GlobalVoxelPos {
            tlc: TLCPos(global_tlc),
            voxel_index: VoxelPosInLOD::in_full_lod(VoxelPos(pos_in_tlc))
                .index(chunk_size, largest_chunk_lvl),
        }
    }
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
    };

    const CHUNK_SIZE: ChunkSize = ChunkSize::new(3);

    #[derive(Debug, Sequence, Clone, Copy, FromPrimitive, ToPrimitive)]
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

        fn empty() -> u8 {
            0
        }
    }

    #[test]
    fn test_edit_voxel_grid() {
        let renderer_context = TestContext::new();
        let start_tlc = TLCPos(Point3::<i64> {
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
            0.26,
        );

        {
            let mut editor = ChunkVoxelEditor::<Block, 5>::edit_grid(&mut grid);
            let chunk = editor.chunks[1640].lods[1].as_mut().unwrap().data_mut();
            unsafe {
                chunk.set_missing();
                chunk.set_valid();
            }
            chunk
                .get_mut()
                .unwrap()
                .with_voxel_ids_mut()
                .unwrap()
                .update_full_buffer_gpu();
        }

        assert!(
            grid.lods[1].chunks()[2]
                .get()
                .expect("Chunk indexed was not valid; i.e. not indexed properly somewhere in this process")
                .updated_bitmask_regions()
                .regions
                .len()
                == 1
        );
    }
}
