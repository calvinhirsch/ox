use crate::blocks::Block;
use cgmath::{Array, Point3, Vector3};
use hashbrown::HashMap;
use ox::ray::ChunkEditorVoxels;
use ox::world::loader::{BorrowChunkForLoading, BorrowedChunk, ChunkLoadQueueItem, LayerChunk};
use ox::world::mem_grid::layer::MemoryGridLayer;
use ox::world::mem_grid::utils::{cubed, ChunkSize, VoxelPosInLod};
use ox::world::mem_grid::voxel::grid::{
    BorrowedChunkVoxelEditor, ChunkVoxelEditor, VoxelChunkLoadQueueItemData,
    VoxelMemoryGridMetadata,
};
use ox::world::mem_grid::voxel::{ChunkVoxels, VoxelMemoryGrid};
use ox::world::mem_grid::{MemoryGrid, MemoryGridEditor, MemoryGridEditorChunk};
use ox::world::{TlcPos, VoxelPos};

pub const CHUNK_SIZE: ChunkSize = ChunkSize::new(3);

#[derive(Debug, Clone)]
pub struct Entity {
    pub position: Point3<f32>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Entities {
    pub entities: Vec<Entity>,
}
impl Default for Entities {
    fn default() -> Self {
        Entities { entities: vec![] }
    }
}

pub type EntityMemoryGrid = MemoryGridLayer<Entities>;

#[derive(Debug)]
pub struct WorldMemoryGrid<const N: usize> {
    pub voxel: VoxelMemoryGrid<N>,
    pub entity: EntityMemoryGrid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldChunkLoadQueueItemData<const N: usize> {
    voxel: Option<VoxelChunkLoadQueueItemData<N>>,
    entity: Option<()>,
}

impl<const N: usize> WorldMemoryGrid<N> {
    pub fn new(
        voxel_mem_grid: VoxelMemoryGrid<N>,
        start_tlc: TlcPos<i64>,
        entity_render_area_size: usize,
    ) -> Self {
        let vox_size = voxel_mem_grid.size();
        let entity_grid_size = entity_render_area_size + 1;
        WorldMemoryGrid {
            voxel: voxel_mem_grid,
            entity: EntityMemoryGrid::new(
                (0..cubed(entity_grid_size))
                    .map(|_| LayerChunk::new(Entities { entities: vec![] }))
                    .collect(),
                TlcPos(start_tlc.0 + Vector3::from_value((vox_size - entity_grid_size) as i64 / 2)),
                entity_grid_size,
                (),
            ),
        }
    }
}

impl<const N: usize> MemoryGrid for WorldMemoryGrid<N> {
    type ChunkLoadQueueItemData = WorldChunkLoadQueueItemData<N>;

    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        let mut queue = HashMap::new();

        for item in self.voxel.queue_load_all() {
            let existing_item = queue.insert(
                item.pos.0,
                ChunkLoadQueueItem {
                    pos: item.pos,
                    data: WorldChunkLoadQueueItemData {
                        voxel: Some(item.data),
                        entity: None,
                    },
                },
            );

            // Make sure no duplicates
            debug_assert!(existing_item.is_none());
        }

        for item in self.entity.queue_load_all() {
            let e = queue.entry(item.pos.0).or_insert(ChunkLoadQueueItem {
                pos: item.pos,
                data: WorldChunkLoadQueueItemData {
                    voxel: None,
                    entity: None,
                },
            });
            e.data.entity = Some(());
        }

        queue.into_values().collect()
    }

    fn shift(
        &mut self,
        shift: &ox::world::mem_grid::MemGridShift,
    ) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>> {
        let mut queue = HashMap::new();

        for item in self.voxel.shift(shift) {
            if item.data.lods[0] {
                dbg!(item.pos);
            }
            debug_assert!(queue
                .insert(
                    item.pos.0,
                    ChunkLoadQueueItem {
                        pos: item.pos,
                        data: WorldChunkLoadQueueItemData {
                            voxel: Some(item.data),
                            entity: None,
                        }
                    }
                )
                .is_none())
        }

        for item in self.entity.shift(shift) {
            let e = queue.entry(item.pos.0).or_insert(ChunkLoadQueueItem {
                pos: item.pos,
                data: WorldChunkLoadQueueItemData {
                    voxel: None,
                    entity: None,
                },
            });
            e.data.entity = Some(());
        }

        queue.into_values().collect()
    }

    fn size(&self) -> usize {
        self.voxel.size()
    }

    fn start_tlc(&self) -> TlcPos<i64> {
        self.voxel.start_tlc()
    }
}

#[derive(Debug, Clone)]
pub struct WorldEditorMetadata {
    pub voxel: VoxelMemoryGridMetadata,
    pub entity: (),
}

#[derive(Debug)]
pub struct WorldChunkEditor<'a, const N: usize> {
    pub voxel: ChunkVoxelEditor<'a, Block, N>,
    pub entity: Option<&'a mut LayerChunk<Entities>>,
}

impl<'a, const N: usize> ChunkEditorVoxels<Block, N> for WorldChunkEditor<'a, N> {
    fn voxels(&self) -> &ChunkVoxelEditor<'_, Block, N> {
        &self.voxel
    }
}

#[derive(Debug)]
pub struct BorrowedWorldChunkEditor<const N: usize> {
    voxel: BorrowedChunkVoxelEditor<Block, N>,
    entity: Option<*mut LayerChunk<Entities>>,
}

impl<'a, const N: usize> MemoryGridEditorChunk<'a, WorldMemoryGrid<N>, WorldEditorMetadata>
    for WorldChunkEditor<'a, N>
{
    fn edit_grid_with_size(
        mem_grid: &'a mut WorldMemoryGrid<N>,
        grid_size: usize,
    ) -> MemoryGridEditor<WorldChunkEditor<'a, N>, WorldEditorMetadata> {
        let size = mem_grid.size();
        let start_tlc = mem_grid.start_tlc();

        let voxel_editor = ChunkVoxelEditor::edit_grid_with_size(&mut mem_grid.voxel, grid_size);
        let entity_editor = Option::<&'a mut LayerChunk<Entities>>::edit_grid_with_size(
            &mut mem_grid.entity,
            grid_size,
        );
        let metadata = WorldEditorMetadata {
            voxel: (*voxel_editor.metadata()).clone(),
            entity: (),
        };

        MemoryGridEditor::new(
            voxel_editor
                .chunks
                .into_iter()
                .zip(entity_editor.chunks.into_iter())
                .map(|(voxels_o, entities_o)| WorldChunkEditor {
                    voxel: voxels_o,
                    entity: entities_o,
                })
                .collect(),
            size,
            start_tlc,
            metadata,
        )
    }
}

unsafe impl<'a, const N: usize> BorrowChunkForLoading<BorrowedWorldChunkEditor<N>>
    for WorldChunkEditor<'a, N>
{
    fn take_data_for_loading(&mut self) -> BorrowedWorldChunkEditor<N> {
        if let Some(e) = self.entity.as_mut() {
            unsafe { e.set_missing() };
        }
        BorrowedWorldChunkEditor {
            entity: self.entity.as_mut().map(|e| *e as *mut _),
            voxel: self.voxel.take_data_for_loading(),
        }
    }

    fn mark_invalid(&mut self) -> Result<(), ()> {
        let mut r = Ok(());
        if let Some(entity_data) = self.entity.as_mut() {
            r = r.and(unsafe { entity_data.set_invalid() });
        }
        r = r.and(self.voxel.mark_invalid());
        r
    }
}

unsafe impl<const N: usize> Send for BorrowedWorldChunkEditor<N> {}

unsafe impl<const N: usize> BorrowedChunk for BorrowedWorldChunkEditor<N> {
    unsafe fn mark_valid(&mut self) {
        if self.voxel.lods()[0].is_some() {
            dbg!(
                "VALID",
                (&**self.voxel.lods()[0].as_ref().unwrap().data())
                    .get_for_loading()
                    .bitmask()
                    .bitmask
                    .iter()
                    .map(|bm| (bm.mask > 0) as u32)
                    .sum::<u32>()
            );
        }
        if let Some(e) = self.entity {
            unsafe {
                (&mut *e).set_valid();
            }
        }
        self.voxel.mark_valid();
    }
}

pub fn load_chunk<const N: usize>(
    editor: &mut BorrowedWorldChunkEditor<N>,
    chunk: ChunkLoadQueueItem<WorldChunkLoadQueueItemData<N>>,
    metadata: WorldEditorMetadata,
) {
    unsafe {
        editor.voxel.load_new(
            chunk.pos,
            chunk.data.voxel.unwrap().lods,
            generate_chunk,
            metadata.voxel.chunk_size(),
            metadata.voxel.tlc_size(),
            metadata.voxel.largest_lod().lvl(),
            metadata.voxel.lod_block_fill_thresh(),
        );
    }
    if let Some(e) = editor.entity.as_mut() {
        unsafe { (&mut **e).get_mut_for_loading() }.entities.clear();
    }
}

fn generate_chunk(
    chunk_pos: TlcPos<i64>,
    lvl: u8,
    sublvl: u8,
    voxel_ids_out: &mut ChunkVoxels,
    tlc_size: usize,
    largest_chunk_lvl: u8,
) {
    let voxel_size = CHUNK_SIZE.size().pow(lvl as u32) * 2usize.pow(sublvl as u32);
    let chunk_start_pt: VoxelPos<i64> = VoxelPos(chunk_pos.0 * (tlc_size / voxel_size) as i64);
    for x in 0..(tlc_size / voxel_size) as u32 {
        for y in 0..(tlc_size / voxel_size) as u32 {
            for z in 0..(tlc_size / voxel_size) as u32 {
                let idx = VoxelPosInLod {
                    pos: Point3 { x, y, z },
                    lvl,
                    sublvl,
                }
                .index(CHUNK_SIZE, largest_chunk_lvl);

                // world coords
                let x = (x as i64 + chunk_start_pt.0.x) * voxel_size as i64;
                let y = (y as i64 + chunk_start_pt.0.y) * voxel_size as i64;
                let z = (z as i64 + chunk_start_pt.0.z) * voxel_size as i64;

                // strips
                // voxel_ids_out[idx] = if x % 8 == 0 && (y == 64 * 7) {
                //     Block::Dirt
                // } else {
                //     Block::Air
                // } as u8;

                // inverted pyranmid
                // voxel_ids_out[idx] =
                //     if y < 64 * 7 + (x - (64 * 7 + 32)).abs() + (z - (64 * 7 + 32)).abs() {
                //         Block::Dirt
                //     } else {
                //         Block::Air
                //     } as u8;

                // flat world
                let within_main_tlc = x >= 64 * 7 && x < 64 * 8 && z >= 64 * 7 && z < 64 * 8;
                voxel_ids_out[idx] = if y < 64 * 7 + 8 && within_main_tlc {
                    Block::GrayCarpet
                } else {
                    Block::Air
                } as u8;
                if y == 64 * 7 + 8 && x % 8 == 0 && z % 8 == 0 && within_main_tlc {
                    voxel_ids_out[idx] = Block::RedLight as u8;
                }
                if y == 64 * 7 + 8 && x % 8 == 4 && z % 8 == 4 && within_main_tlc {
                    voxel_ids_out[idx] = Block::GreenLight as u8;
                }
                if y == 64 * 7 + 8 && x % 8 == 4 && z % 8 == 0 && within_main_tlc {
                    voxel_ids_out[idx] = Block::BlueLight as u8;
                }
                if y >= 64 * 7 + 8 && y < 64 * 7 + 11 && x % 8 == 0 && z % 8 == 4 && within_main_tlc
                {
                    voxel_ids_out[idx] = Block::Mirror as u8;
                }
            }
        }
    }
}
