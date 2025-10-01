use crate::blocks::Block;
use cgmath::{Array, Point3, Vector3};
use hashbrown::HashMap;
use ox::loader::{BorrowChunkForLoading, BorrowedChunk, ChunkLoadQueueItem, LayerChunk};
use ox::ray::ChunkEditorVoxels;
use ox::world::mem_grid::layer::{
    DefaultBorrowedLayerChunk, DefaultLayerChunkEditor, MemoryGridLayer,
};
use ox::world::mem_grid::utils::{cubed, ChunkSize, VoxelPosInLod};
use ox::world::mem_grid::voxel::grid::{
    BorrowedChunkVoxelEditor, ChunkVoxelEditor, VoxelChunkLoadQueueItemData,
    VoxelMemoryGridMetadata,
};
use ox::world::mem_grid::voxel::{ChunkVoxels, VoxelMemoryGrid};
use ox::world::mem_grid::{EditMemoryGridChunk, MemoryGrid, MemoryGridLoadChunks};
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
                (),
            ),
        }
    }
}

impl<const N: usize> MemoryGridLoadChunks for WorldMemoryGrid<N> {
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

        dbg!(shift);

        for item in self.voxel.shift(shift) {
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

        dbg!(&queue);

        for item in self.entity.shift(shift) {
            // let e = queue
            //     .get_mut(&item.pos.0)
            //     .expect(format!("{:?}", &item.pos.0).as_str());
            let e = queue.entry(item.pos.0).or_insert(ChunkLoadQueueItem {
                pos: item.pos,
                data: WorldChunkLoadQueueItemData {
                    voxel: None,
                    entity: None,
                },
            });
            e.data.entity = Some(());
        }

        dbg!(&queue);
        // panic!();

        queue.into_values().collect()
    }
}
impl<const N: usize> MemoryGrid for WorldMemoryGrid<N> {
    fn size(&self) -> usize {
        self.voxel.size()
    }

    fn start_tlc(&self) -> TlcPos<i64> {
        self.voxel.start_tlc()
    }
}

#[derive(Debug)]
pub struct WorldChunkEditor<'a, const N: usize> {
    pub voxel: ChunkVoxelEditor<'a, Block, N>,
    pub entity: Option<DefaultLayerChunkEditor<'a, Entities>>,
}

impl<'a, const N: usize> ChunkEditorVoxels<Block, N> for WorldChunkEditor<'a, N> {
    fn voxels(&self) -> &ChunkVoxelEditor<'_, Block, N> {
        &self.voxel
    }
}

#[derive(Debug)]
pub struct BorrowedWorldChunkEditor<const N: usize> {
    voxel: BorrowedChunkVoxelEditor<Block, N>,
    entity: Option<DefaultBorrowedLayerChunk<Entities>>,
}

impl<const N: usize> EditMemoryGridChunk for WorldMemoryGrid<N> {
    type ChunkEditor<'a> = WorldChunkEditor<'a, N>
        where
            Self: 'a;

    fn edit_chunk(
        &mut self,
        pos: TlcPos<i64>,
        buffer_chunk_states: [ox::world::BufferChunkState; 3],
    ) -> Option<Self::ChunkEditor<'_>> {
        Some(WorldChunkEditor {
            voxel: self.voxel.edit_chunk(pos, buffer_chunk_states)?,
            entity: self.entity.edit_chunk(pos, buffer_chunk_states),
        })
    }
}

impl<'a, const N: usize>
    BorrowChunkForLoading<BorrowedWorldChunkEditor<N>, WorldChunkLoadQueueItemData<N>>
    for WorldChunkEditor<'a, N>
{
    fn should_still_load(&self, queue_item: &WorldChunkLoadQueueItemData<N>) -> bool {
        if let Some(voxel) = queue_item.voxel.as_ref() {
            self.voxel.should_still_load(voxel)
        } else {
            true
        }
    }

    fn mark_invalid(&mut self) -> Result<(), ()> {
        let mut r = Ok(());
        if let Some(entity_data) = self.entity.as_mut() {
            r = r.and(entity_data.chunk.set_invalid());
        }
        r = r.and(self.voxel.mark_invalid());
        r
    }

    fn take_data_for_loading(
        &mut self,
        queue_item: &WorldChunkLoadQueueItemData<N>,
    ) -> BorrowedWorldChunkEditor<N> {
        BorrowedWorldChunkEditor {
            entity: self.entity.as_mut().map(|e| e.take_data_for_loading(&())),
            voxel: self
                .voxel
                .take_data_for_loading(queue_item.voxel.as_ref().unwrap()),
        }
    }
}

impl<const N: usize> BorrowedChunk for BorrowedWorldChunkEditor<N> {
    type MemoryGrid = WorldMemoryGrid<N>;

    fn return_data(self, grid: &mut Self::MemoryGrid) {
        if let Some(e) = self.entity {
            e.return_data(&mut grid.entity);
        }

        self.voxel.return_data(&mut grid.voxel);
    }
}

pub fn load_chunk<const N: usize>(
    editor: &mut BorrowedWorldChunkEditor<N>,
    chunk: ChunkLoadQueueItem<WorldChunkLoadQueueItemData<N>>,
    params: VoxelMemoryGridMetadata,
) {
    unsafe {
        editor.voxel.load_new(chunk.pos, generate_chunk, &params);
    }
    if let Some(e) = editor.entity.as_mut() {
        e.chunk.entities.clear();
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

                // flat world with mirrors
                let tlc_size = CHUNK_SIZE.size().pow(2) as i64;
                let center_tlc = 15;
                let within_area = x >= tlc_size * (center_tlc - 2)
                    && x < tlc_size * (center_tlc + 2)
                    && z >= tlc_size * (center_tlc - 2)
                    && z < tlc_size * (center_tlc + 2);
                voxel_ids_out[idx] = if y < tlc_size * center_tlc + 8 && within_area {
                    Block::GrayCarpet
                } else {
                    Block::Air
                } as u8;
                if y == tlc_size * center_tlc + 8 && x % 8 == 0 && z % 8 == 0 && within_area {
                    voxel_ids_out[idx] = Block::RedLight as u8;
                }
                if y == tlc_size * center_tlc + 8 && x % 8 == 4 && z % 8 == 4 && within_area {
                    voxel_ids_out[idx] = Block::GreenLight as u8;
                }
                if y == tlc_size * center_tlc + 8 && x % 8 == 4 && z % 8 == 0 && within_area {
                    voxel_ids_out[idx] = Block::BlueLight as u8;
                }
                if y >= tlc_size * center_tlc + 8
                    && y < tlc_size * center_tlc + 11
                    && x % 8 == 0
                    && z % 8 == 4
                    && within_area
                {
                    voxel_ids_out[idx] = Block::Mirror as u8;
                }

                // hills
                // let avg_height = 64.0 * 14.0;
                // let amp = 24.0;
                // let period = 24.0;
                // if (y as f64)
                //     < ((x as f64 / period).sin() + (z as f64 / period).sin()) * amp + avg_height
                // {
                //     voxel_ids_out[idx] = Block::Grass as u8;
                // }
            }
        }
    }
}
