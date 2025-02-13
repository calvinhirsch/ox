use std::sync::Arc;

use cgmath::{Array, Point3, Vector3};
use enum_iterator::Sequence;
use hashbrown::{HashMap, HashSet};
use num_derive::{FromPrimitive, ToPrimitive};

use ox::renderer::component::voxels::data::{VoxelBitmask, VoxelTypeIDs};
use ox::renderer::component::voxels::lod::{VoxelIDUpdate, VoxelLODUpdate};
use ox::voxel_type::{Material, VoxelTypeDefinition, VoxelTypeEnum};
use ox::world::mem_grid::utils::{cubed, squared, VoxelPosInLOD};
use ox::world::mem_grid::voxel::grid::ChunkVoxelEditor;
use ox::world::mem_grid::voxel::{VoxelLODCreateParams, VoxelMemoryGrid};
use ox::world::mem_grid::{MemoryGrid, MemoryGridEditorChunk};
use ox::world::TLCPos;
use ox::{
    renderer::test_context::TestContext,
    world::{
        loader::{ChunkLoadQueueItem, ChunkLoader, ChunkLoaderParams},
        mem_grid::{
            utils::ChunkSize,
            voxel::grid::{
                BorrowedChunkVoxelEditor, VoxelChunkLoadQueueItemData, VoxelMemoryGridMetadata,
            },
        },
        BufferChunkState,
    },
};
use vulkano::command_buffer::BufferCopy;
use vulkano::memory::allocator::MemoryAllocator;

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

fn fill_chunk<const N: usize, VE: VoxelTypeEnum>(
    data: &mut BorrowedChunkVoxelEditor<VE, N>,
    chunk: ChunkLoadQueueItem<VoxelChunkLoadQueueItemData<N>>,
    metadata: VoxelMemoryGridMetadata,
) {
    unsafe {
        data.load_new(
            chunk.pos,
            chunk.data.lods,
            |_, lvl, sublvl, voxel_ids_out, tlc_size, largest_chunk_lvl| {
                let voxel_size = CHUNK_SIZE.size().pow(lvl as u32) * 2usize.pow(sublvl as u32);
                for x in 0..(tlc_size / voxel_size) as u32 {
                    for y in 0..(tlc_size / voxel_size) as u32 {
                        for z in 0..(tlc_size / voxel_size) as u32 {
                            let idx = VoxelPosInLOD {
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
            metadata.chunk_size(),
            metadata.tlc_size(),
            metadata.largest_lod().lvl(),
            metadata.lod_block_fill_thresh(),
        );
        if chunk.pos.0 == (Point3 { x: 1, y: -1, z: -1 }) {
            println!("Loading it!");
            let d = &mut **data.lods[1].as_mut().expect("cyz").data_mut();
            d.get_mut_for_loading().updated_bitmask_regions.regions = vec![BufferCopy {
                src_offset: 1111,
                dst_offset: 1111,
                size: 1111,
                ..Default::default()
            }]
        }
    }
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
    assert_eq!(
        u1.iter()
            .cloned()
            .map(|u| u
                .bitmask_updated_regions
                .into_iter()
                .map(|r| r.into())
                .collect())
            .collect::<HashSet<Vec<BufferCopyCmp>>>(),
        u2.iter()
            .cloned()
            .map(|u| u
                .bitmask_updated_regions
                .into_iter()
                .map(|r| r.into())
                .collect())
            .collect::<HashSet<Vec<BufferCopyCmp>>>(),
    );
    assert_eq!(
        u1.into_iter()
            .filter_map(|u| u.id_update.map(|iu| iu
                .updated_regions
                .into_iter()
                .map(|r| r.into())
                .collect()))
            .collect::<HashSet<Vec<BufferCopyCmp>>>(),
        u2.into_iter()
            .filter_map(|u| u.id_update.map(|iu| iu
                .updated_regions
                .into_iter()
                .map(|r| r.into())
                .collect()))
            .collect::<HashSet<Vec<BufferCopyCmp>>>(),
    );
}

#[test]
fn test_queue_load_all() {
    let renderer_context = TestContext::new();
    let start_tlc = TLCPos(Point3::<i64> { x: 0, y: 0, z: 0 } - Vector3::from_value(7));
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

    let expected_queue: HashSet<_> = {
        let mut q = (-7..=7)
            .into_iter()
            .flat_map(|x| {
                (-7..=7).into_iter().flat_map(move |y| {
                    (-7..=7).into_iter().map(move |z| ChunkLoadQueueItem {
                        pos: TLCPos(Point3 { x, y, z }),
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
                    q.get_mut(&TLCPos(Point3 { x, y, z })).unwrap().data.lods[2] = true;
                }
            }
        }
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    q.get_mut(&TLCPos(Point3 { x, y, z })).unwrap().data.lods[1] = true;
                }
            }
        }
        q.get_mut(&TLCPos(Point3 { x: 0, y: 0, z: 0 }))
            .unwrap()
            .data
            .lods[0] = true;

        q.into_values().collect()
    };

    let queue = grid.queue_load_all();

    assert_eq!(
        queue.clone().into_iter().collect::<HashSet<_>>(),
        expected_queue,
        "grid.queue_load_all() returned incorrect chunks to queue"
    );

    // Load chunks
    {
        let mut editor = ChunkVoxelEditor::<Block, 5>::edit_grid(&mut grid);

        unsafe {
            editor.chunks[1640].lods[1]
                .as_mut()
                .unwrap()
                .data_mut()
                .set_missing();
            editor.chunks[1640].lods[1]
                .as_mut()
                .unwrap()
                .data_mut()
                .get_mut_for_loading()
        }
        .updated_bitmask_regions
        .regions = vec![BufferCopy {
            src_offset: 2222,
            dst_offset: 2222,
            size: 2222,
            ..Default::default()
        }];
        unsafe {
            editor.chunks[1640].lods[1]
                .as_mut()
                .unwrap()
                .data_mut()
                .set_valid();
        }
        // let mut loader = ChunkLoader::new(ChunkLoaderParams { n_threads: 1 });
        // let all_unloaded = [
        //     BufferChunkState::Unloaded,
        //     BufferChunkState::Unloaded,
        //     BufferChunkState::Unloaded,
        // ];
        // loader.sync(
        //     start_tlc,
        //     &mut editor,
        //     queue,
        //     &all_unloaded,
        //     &fill_chunk::<5, Block>,
        // );
        // while loader.active_loading_threads() > 0 {
        //     loader.sync(
        //         start_tlc,
        //         &mut editor,
        //         vec![],
        //         &all_unloaded,
        //         &fill_chunk::<5, Block>,
        //     );
        // }
    }

    // Examine updates that would be made to staging buffers for each LOD

    let [u_0_0, u_0_1, u_0_2, u_1_0, u_2_0] = grid.get_updates();
    let dummy_bitmask = VoxelBitmask::new_vec(0);
    let dummy_ids = VoxelTypeIDs::new_vec(0);

    // LOD 0 0
    // assert_updates_eq(
    //     u_0_0,
    //     vec![VoxelLODUpdate {
    //         bitmask: &dummy_bitmask,
    //         bitmask_updated_regions: vec![BufferCopy {
    //             src_offset: 0,
    //             dst_offset: 0,
    //             size: cubed(64) / 8,
    //             ..Default::default()
    //         }],
    //         id_update: Some(VoxelIDUpdate {
    //             ids: &dummy_ids,
    //             updated_regions: vec![BufferCopy {
    //                 src_offset: 0,
    //                 dst_offset: 0,
    //                 size: cubed(64),
    //                 ..Default::default()
    //             }],
    //         }),
    //     }],
    // );

    let gen_update = |x: u64, y: u64, z: u64, grid_size: u64, n_vox: u64| VoxelLODUpdate {
        bitmask: &dummy_bitmask,
        bitmask_updated_regions: vec![BufferCopy {
            src_offset: 0,
            dst_offset: (x + y * squared(grid_size) + z * grid_size) * n_vox / 8,
            size: n_vox / 8,
            ..Default::default()
        }],
        id_update: Some(VoxelIDUpdate {
            ids: &dummy_ids,
            updated_regions: vec![BufferCopy {
                src_offset: 0,
                dst_offset: (x + y * squared(grid_size) + z * grid_size) * n_vox,
                size: n_vox,
                ..Default::default()
            }],
        }),
    };

    // LOD 0 1
    let n_vox = cubed(32);
    let grid_size = 4;
    assert_updates_eq(
        u_0_1,
        (0..grid_size - 1)
            .into_iter()
            .flat_map(|x| {
                (0..grid_size - 1).into_iter().flat_map(move |y| {
                    (0..grid_size - 1)
                        .into_iter()
                        .map(move |z| gen_update(x, y, z, grid_size, n_vox))
                })
            })
            .collect(),
    );

    // LOD 0 2
    let n_vox = cubed(16);
    let grid_size = 8;
    assert_updates_eq(
        u_0_2,
        (0..grid_size - 1)
            .into_iter()
            .flat_map(|x| {
                (0..grid_size - 1).into_iter().flat_map(move |y| {
                    (0..grid_size - 1)
                        .into_iter()
                        .map(move |z| gen_update(x, y, z, grid_size, n_vox))
                })
            })
            .collect(),
    );

    // LOD 1 0
    let n_vox = cubed(8);
    let grid_size = 16;
    assert_updates_eq(
        u_1_0,
        (0..grid_size - 1)
            .into_iter()
            .flat_map(|x| {
                (0..grid_size - 1).into_iter().flat_map(move |y| {
                    (0..grid_size - 1)
                        .into_iter()
                        .map(move |z| gen_update(x, y, z, grid_size, n_vox))
                })
            })
            .collect(),
    );

    // LOD 2 0
    let n_vox = cubed(8);
    let grid_size = 16;
    assert_updates_eq(
        u_2_0,
        (0..grid_size - 1)
            .into_iter()
            .flat_map(|x| {
                (0..grid_size - 1).into_iter().flat_map(move |y| {
                    (0..grid_size - 1)
                        .into_iter()
                        .map(move |z| gen_update(x, y, z, grid_size, n_vox))
                })
            })
            .collect(),
    );
}
