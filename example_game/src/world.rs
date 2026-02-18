use crate::blocks::Block;
use cgmath::{Array, InnerSpace, Point2, Point3, Vector2, Vector3};
use hashbrown::HashMap;
use ox::loader::{ChunkLoadQueueItem, LayerChunk, TakeChunkForLoading, TakenChunk};
use ox::ray::ChunkEditorVoxels;
use ox::world::mem_grid::layer::{
    DefaultLayerChunkEditor, DefaultTakenLayerChunk, MemoryGridLayer,
};
use ox::world::mem_grid::utils::{cubed, ChunkSize, VoxelPosInLod};
use ox::world::mem_grid::voxel::grid::{
    ChunkVoxelEditor, TakenChunkVoxelEditor, VoxelChunkLoadQueueItemData, VoxelMemoryGridMetadata,
};
use ox::world::mem_grid::voxel::{ChunkVoxels, VoxelMemoryGrid};
use ox::world::mem_grid::{EditMemoryGridChunk, MemoryGrid, MemoryGridLoadChunks};
use ox::world::{TlcPos, VoxelPos};

pub const CHUNK_SIZE: ChunkSize = ChunkSize::new(3);

#[derive(Debug, Clone)]
pub struct Entity {
    pub _position: Point3<f32>,
    pub _name: String,
}

/// This is what we will store in each chunk to track entities
#[derive(Debug, Clone)]
pub struct Entities {
    pub entities: Vec<Entity>,
}

/// Our custom memory grid
#[derive(Debug)]
pub struct WorldMemoryGrid<const N: usize> {
    pub voxel: VoxelMemoryGrid<N>,
    pub entity: MemoryGridLayer<Entities>,
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
        entity_loaded_area_size: usize,
    ) -> Self {
        let vox_size = voxel_mem_grid.size();
        let entity_grid_size = entity_loaded_area_size + 1;
        WorldMemoryGrid {
            voxel: voxel_mem_grid,
            entity: MemoryGridLayer::<Entities>::new(
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

        for item in self.voxel.shift(shift) {
            let r = queue.insert(
                item.pos.0,
                ChunkLoadQueueItem {
                    pos: item.pos,
                    data: WorldChunkLoadQueueItemData {
                        voxel: Some(item.data),
                        entity: None,
                    },
                },
            );
            debug_assert!(r.is_none())
        }

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
pub struct TakenWorldChunkEditor<const N: usize> {
    voxel: TakenChunkVoxelEditor<Block, N>,
    entity: Option<DefaultTakenLayerChunk<Entities>>,
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
    TakeChunkForLoading<TakenWorldChunkEditor<N>, WorldChunkLoadQueueItemData<N>>
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
    ) -> TakenWorldChunkEditor<N> {
        TakenWorldChunkEditor {
            entity: self.entity.as_mut().map(|e| e.take_data_for_loading(&())),
            voxel: self
                .voxel
                .take_data_for_loading(queue_item.voxel.as_ref().unwrap()),
        }
    }
}

impl<const N: usize> TakenChunk for TakenWorldChunkEditor<N> {
    type MemoryGrid = WorldMemoryGrid<N>;

    fn return_data(self, grid: &mut Self::MemoryGrid) {
        if let Some(e) = self.entity {
            e.return_data(&mut grid.entity);
        }

        self.voxel.return_data(&mut grid.voxel);
    }
}

pub fn load_chunk<const N: usize>(
    editor: &mut TakenWorldChunkEditor<N>,
    chunk: ChunkLoadQueueItem<WorldChunkLoadQueueItemData<N>>,
    params: VoxelMemoryGridMetadata,
) {
    editor.voxel.load_new(chunk.pos, generate_chunk, &params);
    if let Some(e) = editor.entity.as_mut() {
        e.chunk.entities.clear();
    }
}

/// www.pcg-random.org and www.shadertoy.com/view/XlGcRh
fn randi(inp: u32) -> u32 {
    let x = inp.wrapping_mul(747796405).wrapping_add(2891336453);
    let x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
    (x >> 22) ^ x
}
fn rand(inp: u32) -> f32 {
    randi(inp) as f32 / u32::MAX as f32
}

/// returns noise value and its derivatives
/// https://www.shadertoy.com/view/MdX3Rr
fn noised(pt: Point2<f64>) -> (f32, Vector2<f32>) {
    // this will be bad at high integer values
    let (xf, yf) = (pt.x.fract() as f32, pt.y.fract() as f32);
    let ux = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0);
    let uy = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0);
    let dux = 30.0 * xf * xf * (xf * (xf - 2.0) + 1.0);
    let duy = 30.0 * yf * yf * (yf * (yf - 2.0) + 1.0);

    let tile_x = pt.x.floor() as i64;
    let tile_y = pt.y.floor() as i64;
    fn tile_seed(x: i64, y: i64) -> u32 {
        (100_000_000 + x * 10_000 + y) as u32
    }
    let a = rand(tile_seed(tile_x, tile_y));
    let b = rand(tile_seed(tile_x + 1, tile_y));
    let c = rand(tile_seed(tile_x, tile_y + 1));
    let d = rand(tile_seed(tile_x + 1, tile_y + 1));

    let w = a - b - c + d;
    (
        a + (b - a) * ux + (c - a) * uy + w * ux * uy,
        Vector2 {
            x: dux * ((b - a) + w * uy),
            y: duy * ((c - a) + w * ux),
        },
    )
}

const CENTER_TLC: i64 = 11;
const N_NOISE_LAYERS: usize = 6;
const TILE_SIZE: u32 = 400;
const NOISE_SCALE: f32 = 200.0;
const BASE_TERRAIN_HEIGHT: f64 = 64.0 * (CENTER_TLC as f64 - 3.5);

fn generate_chunk(
    chunk_pos: TlcPos<i64>,
    lvl: u8,
    sublvl: u8,
    voxel_ids_out: &mut ChunkVoxels,
    tlc_size: usize,
    largest_chunk_lvl: u8,
) {
    let voxel_size = CHUNK_SIZE.size().pow(lvl as u32) * 2usize.pow(sublvl as u32);
    let chunk_start_pt: VoxelPos<i64> = VoxelPos(chunk_pos.0 * tlc_size as i64);
    let grid_size = tlc_size / voxel_size;

    for x_grid in 0..grid_size as u32 {
        // world coord
        let x = x_grid as i64 * voxel_size as i64 + chunk_start_pt.0.x;

        for z_grid in 0..grid_size as u32 {
            // world coord
            let z = z_grid as i64 * voxel_size as i64 + chunk_start_pt.0.z;

            // terrain height
            let (dh2, height) = {
                let mut h = 0.0;
                let mut dh = Vector2::new(0.0, 0.0);
                let mut dh2 = None;
                for noise_layer in 0..N_NOISE_LAYERS {
                    let tile_size_divisor = (1usize << noise_layer) as f64;
                    let tile_coords = Point2 {
                        x: tile_size_divisor * x as f64 / TILE_SIZE as f64,
                        y: tile_size_divisor * z as f64 / TILE_SIZE as f64,
                    };

                    let (v, dv) = noised(tile_coords);
                    dh += dv;
                    let noise_scale = 1.0 / (1usize << noise_layer) as f32;
                    h += (noise_scale * v / (1.0 + dh.dot(dh))) as f64;

                    if noise_layer == 2 {
                        dh2 = Some(dh);
                    }
                }
                (dh2.unwrap(), h * NOISE_SCALE as f64 + BASE_TERRAIN_HEIGHT)
            };

            for y_grid in 0..grid_size as u32 {
                // world coord
                let y = y_grid as i64 * voxel_size as i64 + chunk_start_pt.0.y;

                // index in voxel_ids_out
                let idx = VoxelPosInLod {
                    pos: Point3 {
                        x: x_grid,
                        y: y_grid,
                        z: z_grid,
                    },
                    lvl,
                    sublvl,
                }
                .index(CHUNK_SIZE, largest_chunk_lvl);

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
                let tlc_size_i = tlc_size as i64;
                if x >= tlc_size_i * (CENTER_TLC)
                    && x < tlc_size_i * (CENTER_TLC + 1)
                    && z >= tlc_size_i * (CENTER_TLC)
                    && z < tlc_size_i * (CENTER_TLC + 1)
                {
                    voxel_ids_out[idx] = if y < tlc_size_i * CENTER_TLC + 8 {
                        Block::GrayCarpet
                    } else {
                        Block::Air
                    } as u8;
                    if y == tlc_size_i * CENTER_TLC + 8 && x % 8 == 0 && z % 8 == 0 {
                        voxel_ids_out[idx] = Block::RedLight as u8;
                    }
                    if y == tlc_size_i * CENTER_TLC + 8 && x % 8 == 4 && z % 8 == 4 {
                        voxel_ids_out[idx] = Block::GreenLight as u8;
                    }
                    if y == tlc_size_i * CENTER_TLC + 8 && x % 8 == 4 && z % 8 == 0 {
                        voxel_ids_out[idx] = Block::BlueLight as u8;
                    }
                    if y >= tlc_size_i * CENTER_TLC + 8
                        && y < tlc_size_i * CENTER_TLC + 11
                        && x % 8 == 0
                        && z % 8 == 4
                    {
                        voxel_ids_out[idx] = Block::Mirror as u8;
                    }
                }

                // hills
                // let avg_height = 64.0 * 7.5;
                // let amp = 24.0;
                // let period = 24.0;
                // if (y as f64)
                //     < ((x as f64 / period).sin() + (z as f64 / period).sin()) * amp + avg_height
                // {
                //     voxel_ids_out[idx] = Block::Grass as u8;
                // }

                // terrain
                if (y as f64) < height {
                    voxel_ids_out[idx] = if (y as f64) < height - 5.0 {
                        Block::Rock as u8
                    } else if dh2.dot(dh2) > 0.8 {
                        Block::Rock as u8
                    } else if height > 64.0 * (CENTER_TLC as f64 - 2.0) {
                        Block::Snow as u8
                    } else {
                        Block::Grass as u8
                    }
                }
            }
        }
    }
}
