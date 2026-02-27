use std::sync::Arc;
use std::u128;

use cgmath::{Array, Point3, Vector3};
use enum_iterator::Sequence;
use hashbrown::{HashMap, HashSet};
use num_derive::{FromPrimitive, ToPrimitive};

use ox::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use ox::renderer::component::voxels::lod::{VoxelIDUpdate, VoxelLODUpdate};
use ox::voxel_type::{Material, VoxelTypeDefinition, VoxelTypeEnum};
use ox::world::camera::Camera;
use ox::world::mem_grid::utils::{cubed, squared, VoxelPosInLod};
use ox::world::mem_grid::voxel::{VoxelLODCreateParams, VoxelMemoryGrid};
use ox::world::{TlcPos, World};
use ox::{
    loader::{ChunkLoadQueueItem, ChunkLoader, ChunkLoaderParams},
    renderer::test_context::TestContext,
    world::mem_grid::{
        utils::ChunkSize,
        voxel::grid::{
            TakenChunkVoxelEditor, VoxelChunkLoadQueueItemData, VoxelMemoryGridMetadata,
        },
        MemoryGrid, MemoryGridLoadChunks,
    },
};
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;

const CHUNK_SIZE: ChunkSize = ChunkSize::new(3);

#[derive(Debug, Sequence, Clone, Copy, FromPrimitive, ToPrimitive, Hash, PartialEq, Eq)]
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

fn fill_chunk<const N: usize, VE: VoxelTypeEnum>(
    data: &mut TakenChunkVoxelEditor<VE, N>,
    chunk: ChunkLoadQueueItem<VoxelChunkLoadQueueItemData<N>>,
    metadata: VoxelMemoryGridMetadata,
) {
    data.load_new(
        chunk.pos,
        |_, lvl, sublvl, voxel_ids_out, tlc_size, largest_chunk_lvl| {
            let voxel_size = CHUNK_SIZE.size().pow(lvl as u32) * 2usize.pow(sublvl as u32);
            for x in 0..(tlc_size / voxel_size) as u32 {
                for y in 0..(tlc_size / voxel_size) as u32 {
                    for z in 0..(tlc_size / voxel_size) as u32 {
                        let idx = VoxelPosInLod {
                            pos: Point3 { x, y, z },
                            lvl,
                            sublvl,
                        }
                        .index(CHUNK_SIZE, largest_chunk_lvl);
                        voxel_ids_out[idx] = Block::SOLID as u8;
                    }
                }
            }
        },
        &metadata,
    );
}

#[derive(Debug, Hash, PartialEq, Eq)]
struct BufferCopyCmp {
    src_offset: u64,
    dst_offset: u64,
    size: u64,
}
impl From<BufferCopy> for BufferCopyCmp {
    fn from(
        BufferCopy {
            src_offset,
            dst_offset,
            size,
            _ne: _,
        }: BufferCopy,
    ) -> Self {
        Self {
            src_offset,
            dst_offset,
            size,
        }
    }
}

fn assert_updates_eq(u1: Vec<VoxelLODUpdate>, u2: Vec<VoxelLODUpdate>) {
    let bm_lookup1: HashMap<BufferCopyCmp, VoxelLODUpdate> = u1
        .iter()
        .cloned()
        .map(|u| (u.bitmask_updated_region.clone().into(), u))
        .collect();
    let bm_lookup2: HashMap<BufferCopyCmp, VoxelLODUpdate> = u2
        .iter()
        .cloned()
        .map(|u| (u.bitmask_updated_region.clone().into(), u))
        .collect();

    assert_eq!(
        bm_lookup1.keys().collect::<HashSet<_>>(),
        bm_lookup2.keys().collect::<HashSet<_>>()
    );
    for (k, v) in bm_lookup1.iter() {
        assert_eq!(
            *v.bitmask,
            *bm_lookup2.get(k).unwrap().bitmask,
            "for chunk {:?} (idx {})",
            k,
            k.dst_offset / k.size,
        );
    }

    let vox_lookup1: HashMap<BufferCopyCmp, VoxelIDUpdate> = u1
        .iter()
        .cloned()
        .filter_map(|u| u.id_update.map(|u| (u.updated_region.clone().into(), u)))
        .collect();
    let vox_lookup2: HashMap<BufferCopyCmp, VoxelIDUpdate> = u2
        .iter()
        .cloned()
        .filter_map(|u| u.id_update.map(|u| (u.updated_region.clone().into(), u)))
        .collect();
    assert_eq!(
        vox_lookup1.keys().collect::<HashSet<_>>(),
        vox_lookup2.keys().collect::<HashSet<_>>()
    );
    for (k, v) in vox_lookup1.iter() {
        assert_eq!(
            *v.ids,
            *vox_lookup2.get(k).unwrap().ids,
            "for chunk {:?} (idx {})",
            k,
            k.dst_offset / k.size,
        );
    }
}

#[test]
fn test_queue_load_all() {
    let renderer_context = TestContext::new();
    let start_tlc = TlcPos(Point3::<i64> { x: 0, y: 0, z: 0 } - Vector3::from_value(7));
    let (grid, _) = VoxelMemoryGrid::new(
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
    let mg_size = grid.size();
    let mut world = World::new(grid, Camera::new(v, mg_size), v, v as u32);

    let expected_queue: HashSet<_> = {
        let mut q = (-7..=7)
            .into_iter()
            .flat_map(|x| {
                (-7..=7).into_iter().flat_map(move |y| {
                    (-7..=7).into_iter().map(move |z| ChunkLoadQueueItem {
                        pos: TlcPos(Point3 { x, y, z }),
                        data: VoxelChunkLoadQueueItemData {
                            lods: [false, false, false, true, true],
                        },
                    })
                })
            })
            .map(|qi| (qi.pos.clone(), qi))
            .collect::<HashMap<_, _>>();

        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    q.get_mut(&TlcPos(Point3 { x, y, z })).unwrap().data.lods[2] = true;
                }
            }
        }
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    q.get_mut(&TlcPos(Point3 { x, y, z })).unwrap().data.lods[1] = true;
                }
            }
        }
        q.get_mut(&TlcPos(Point3 { x: 0, y: 0, z: 0 }))
            .unwrap()
            .data
            .lods[0] = true;

        q.into_values().collect()
    };

    let queue = world.mem_grid.queue_load_all();

    assert_eq!(
        queue.clone().into_iter().collect::<HashSet<_>>(),
        expected_queue,
        "grid.queue_load_all() returned incorrect chunks to queue"
    );

    // Load chunks
    {
        let mut loader =
            ChunkLoader::<_, TakenChunkVoxelEditor<Block, 5>>::new(ChunkLoaderParams {
                n_threads: 1,
            });
        let md = world.mem_grid.metadata().clone();
        for chunk in queue {
            let priority = world.mem_grid.chunk_loading_priority(chunk.pos);
            loader.enqueue(chunk, priority);
        }
        loader.sync(&mut world, &fill_chunk::<5, Block>, md.clone());
        assert!(loader.skipped_loading_last() == 0);

        while loader.active_loading_threads() > 0 {
            loader.sync(&mut world, &fill_chunk::<5, Block>, md.clone());
            assert!(loader.skipped_loading_last() == 0);
        }
    }

    // Examine updates that would be made to staging buffers for each LOD

    let [u_0_0, u_0_1, u_0_2, u_1_0, u_2_0] = world.mem_grid.get_updates();
    let dummy_bitmask = VoxelBitmask::new_vec(0);
    let dummy_ids = VoxelTypeIDs::new_vec(0);
    let mut u_0_0_bitmask = vec![VoxelBitmask::new_vec(cubed(64)); cubed(2)];
    let mut u_0_1_bitmask = vec![VoxelBitmask::new_vec(cubed(32)); cubed(4)];
    let mut u_0_2_bitmask = vec![VoxelBitmask::new_vec(cubed(16)); cubed(8)];
    let mut u_1_0_bitmask = vec![VoxelBitmask::new_vec(cubed(8)); cubed(16)];
    let mut u_2_0_bitmask = vec![VoxelBitmask::new_vec(1); cubed(16)];
    let mut u_0_0_voxels = vec![VoxelTypeIDs::new_vec(cubed(64)); cubed(2)];
    let mut u_0_1_voxels = vec![VoxelTypeIDs::new_vec(cubed(32)); cubed(4)];
    let mut u_0_2_voxels = vec![VoxelTypeIDs::new_vec(cubed(16)); cubed(8)];
    let mut u_1_0_voxels = vec![VoxelTypeIDs::new_vec(cubed(8)); cubed(16)];

    // LOD 0 0 --- offset = 0
    for i in 0..u_0_0_bitmask[0].len() {
        u_0_0_bitmask[0][i] = VoxelBitmask { mask: u128::MAX };
    }
    for i in 0..u_0_0_voxels[0].len() {
        u_0_0_voxels[0][i] = VoxelTypeIDs {
            indices: [Block::SOLID as u8; 128 / 8],
        };
    }
    assert_updates_eq(
        u_0_0,
        vec![VoxelLODUpdate {
            bitmask: &u_0_0_bitmask[0],
            bitmask_updated_region: BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: cubed(64) / 8,
                ..Default::default()
            },
            id_update: Some(VoxelIDUpdate {
                ids: &u_0_0_voxels[0],
                updated_region: BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: cubed(64),
                    ..Default::default()
                },
            }),
        }],
    );

    fn gen_update<'a>(
        x: u64,
        y: u64,
        z: u64,
        grid_size: u64,
        n_vox: u64,
        ids: bool,
        dummy_bitmask: &'a Vec<VoxelBitmask>,
        dummy_ids: &'a Vec<VoxelTypeIDs>,
    ) -> (usize, VoxelLODUpdate<'a>) {
        let chunk_idx = x + y * squared(grid_size) + z * grid_size;
        (
            chunk_idx as usize,
            VoxelLODUpdate {
                bitmask: &dummy_bitmask,
                bitmask_updated_region: BufferCopy {
                    src_offset: 0,
                    dst_offset: chunk_idx * (n_vox / 8).max(16),
                    size: (n_vox + 7) / 8,
                    ..Default::default()
                },
                id_update: if ids {
                    Some(VoxelIDUpdate {
                        ids: &dummy_ids,
                        updated_region: BufferCopy {
                            src_offset: 0,
                            dst_offset: chunk_idx * n_vox.max(16),
                            size: n_vox,
                            ..Default::default()
                        },
                    })
                } else {
                    None
                },
            },
        )
    }

    fn gen_updates<'a>(
        grid_size: u64,
        n_vox: u64,
        offset: u64,
        bitmasks: &'a mut Vec<Vec<VoxelBitmask>>,
        ids: &'a mut Option<&'a mut Vec<Vec<VoxelTypeIDs>>>,
        dummy_bitmask: &'a Vec<VoxelBitmask>,
        dummy_ids: &'a Vec<VoxelTypeIDs>,
    ) -> Vec<VoxelLODUpdate<'a>> {
        let offset_filter = |v: &u64| {
            if offset == 0 {
                *v != grid_size
            } else {
                *v != offset - 1
            }
        };
        let has_ids = ids.is_some();
        let mut updates = (0..grid_size)
            .filter(offset_filter)
            .flat_map(|x| {
                (0..grid_size).filter(offset_filter).flat_map(move |y| {
                    (0..grid_size).filter(offset_filter).map(move |z| {
                        gen_update(x, y, z, grid_size, n_vox, has_ids, dummy_bitmask, dummy_ids)
                    })
                })
            })
            .collect::<Vec<_>>();

        // set bitmask and voxel IDs in ground truth (we are filling chunks completely)
        for (idx, _) in updates.iter_mut() {
            let bitmask_val = VoxelBitmask {
                mask: if n_vox > 128 { u128::MAX } else { 1 },
            };
            for i in 0..bitmasks[*idx].len() {
                bitmasks[*idx][i] = bitmask_val.clone();
            }
            if let Some(ref mut ids) = ids {
                for i in 0..ids[*idx].len() {
                    ids[*idx][i] = VoxelTypeIDs {
                        indices: [Block::SOLID as u8; 128 / 8],
                    };
                }
            }
        }

        // put references to the correct ground truth in each update
        for (idx, update) in updates.iter_mut() {
            update.bitmask = &bitmasks[*idx];
            if let Some(ref ids) = ids {
                update.id_update.as_mut().unwrap().ids = &ids[*idx];
            }
        }

        updates.into_iter().map(|(_, u)| u).collect()
    }

    // LOD 0 1 --- offset = 3
    assert_updates_eq(
        u_0_1,
        gen_updates(
            4,
            cubed(32),
            3,
            &mut u_0_1_bitmask,
            &mut Some(&mut u_0_1_voxels),
            &dummy_bitmask,
            &dummy_ids,
        ),
    );

    // LOD 0 2 --- offset = 5
    assert_updates_eq(
        u_0_2,
        gen_updates(
            8,
            cubed(16),
            5,
            &mut u_0_2_bitmask,
            &mut Some(&mut u_0_2_voxels),
            &dummy_bitmask,
            &dummy_ids,
        ),
    );

    // LOD 1 0 --- offset = 9
    assert_updates_eq(
        u_1_0,
        gen_updates(
            16,
            cubed(8),
            9,
            &mut u_1_0_bitmask,
            &mut Some(&mut u_1_0_voxels),
            &dummy_bitmask,
            &dummy_ids,
        ),
    );

    // LOD 2 0 --- offset = 9
    assert_updates_eq(
        u_2_0,
        gen_updates(
            16,
            1,
            9,
            &mut u_2_0_bitmask,
            &mut None,
            &dummy_bitmask,
            &dummy_ids,
        ),
    );
}
