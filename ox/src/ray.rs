use cgmath::{Array, InnerSpace, Point3, Vector3};
use std::ops::AddAssign;

use crate::{
    voxel_type::VoxelTypeEnum,
    world::{
        mem_grid::{
            utils::{ChunkSize, VoxelPosInLod},
            voxel::grid::{ChunkVoxelEditor, GlobalVoxelPos},
            MemoryGridEditor,
        },
        TlcPos, VoxelPos,
    },
};

pub trait ChunkEditorVoxels<VE: VoxelTypeEnum, const N: usize> {
    fn voxels(&self) -> &ChunkVoxelEditor<'_, VE, N>;
}

pub struct VoxelFace {
    pub ax: u8,    // 0, 1, or 2
    pub dir: bool, // true for positive, false for negative
}

pub struct RayVoxelIntersect {
    pub voxel: GlobalVoxelPos,
    pub face: VoxelFace,
}

pub struct RayPos {
    // top level chunk
    tlc: TlcPos<i64>,
    // position within tlc
    pos: Point3<f32>,
    // which voxel is being examined during traversal
    ipos: Point3<i32>,
}

pub enum CastRayInTlcResult {
    Hit(RayVoxelIntersect),
    Miss(RayPos),
    /// Ray traversed out of rendered area with LOD 0 voxels
    OutOfArea,
}

const TRAVERSAL_SAFETY_LIMIT: u32 = 10_000;

pub fn cast_ray_in_tlc<VE: VoxelTypeEnum, const N: usize, CE: ChunkEditorVoxels<VE, N>, MD>(
    editor: &mut MemoryGridEditor<CE, MD>,
    RayPos { tlc, pos, mut ipos }: RayPos,
    ray_dir: Vector3<f32>,
    chunk_size: ChunkSize,
    largest_chunk_lvl: u8,
    last_crossed_ax: Option<u8>,
) -> Result<CastRayInTlcResult, ()> {
    let chunk_editor = editor.chunk(tlc).unwrap().voxels();
    let chunk_voxels = match &chunk_editor.lods()[0] {
        None => return Ok(CastRayInTlcResult::OutOfArea),
        Some(lod) => match lod.data().get() {
            None => return Err(()),
            Some(data) => data.voxel_ids().as_ref().unwrap(),
        },
    };

    dbg!(tlc);
    dbg!(pos);

    // ENHANCEMENT: a bunch of this gets repeated when called from cast_ray

    let ray_dir = ray_dir.normalize();
    let tlc_size = chunk_size.size().pow(largest_chunk_lvl as u32) as i32;

    // Identify axis (x, y, or z) the ray is most parallel to and set to axis A (with others set to B, C)
    let (ax_a, ax_b, ax_c) = if ray_dir.x > ray_dir.y {
        if ray_dir.x > ray_dir.z {
            (0, 1, 2)
        } else {
            (2, 0, 1)
        }
    } else {
        if ray_dir.y > ray_dir.z {
            (1, 2, 0)
        } else {
            (2, 0, 1)
        }
    };
    let crossed_ax = match last_crossed_ax {
        None => ax_a,
        Some(ax) => ax as usize,
    };

    let xyz_to_abc = {
        let mut v = Vector3::from_value(0);
        v[ax_a] = 0;
        v[ax_b] = 1;
        v[ax_c] = 2;
        v
    };

    // Put everything in abc coords
    // Shadowing the vars here so the XYZ versions are not accidentally used

    let ray_dir = Vector3 {
        x: ray_dir[ax_a],
        y: ray_dir[ax_b],
        z: ray_dir[ax_c],
    };
    let mut pos = Point3 {
        x: pos[ax_a],
        y: pos[ax_b],
        z: pos[ax_c],
    };
    let crossed_ax = xyz_to_abc[crossed_ax];

    // Bounds of traversal
    let min_pt = Vector3::from_value(-1i32);
    let max_pt = Vector3::from_value(tlc_size);

    let mut last_bi = ipos.y;
    let mut last_ci = ipos.z;
    let a_dir = (ray_dir.x > 0.0) as i32 * 2 - 1;

    // helpers

    let vox_idx = |ipos: &Point3<i32>| {
        dbg!("ipos to check", &ipos);
        VoxelPosInLod::in_full_lod(VoxelPos(ipos.cast::<u32>().unwrap()))
            .index(chunk_size, largest_chunk_lvl)
    };
    let hit = |tlc, voxel_index, crossed_ax_abc| {
        Ok(CastRayInTlcResult::Hit(RayVoxelIntersect {
            voxel: GlobalVoxelPos {
                tlc: tlc,
                voxel_index,
            },
            face: VoxelFace {
                ax: crossed_ax_abc as u8,
                dir: ray_dir[crossed_ax] < 0.0,
            },
        }))
    };

    let step_ray = |ipos: &mut Point3<i32>, pos: &mut Point3<f32>| {
        ipos.x += a_dir;
        pos.add_assign(ray_dir);
        ipos.y = pos.y.floor() as i32;
        ipos.z = pos.z.floor() as i32;
    };

    let pos_xyz = |pos_abc: Point3<f32>| {
        let mut r = Point3::from_value(0.0);
        r[ax_a] = pos_abc.x;
        r[ax_b] = pos_abc.y;
        r[ax_c] = pos_abc.z;
        r
    };

    let ipos_xyz = |ipos_abc: Point3<i32>| {
        let mut r = Point3::from_value(0);
        r[ax_a] = ipos_abc.x;
        r[ax_b] = ipos_abc.y;
        r[ax_c] = ipos_abc.z;
        r
    };

    // Check if the block we are starting in got hit

    let starting_block_idx = vox_idx(&ipos);
    if chunk_voxels[starting_block_idx] != VE::empty() {
        return hit(tlc, starting_block_idx, [ax_a, ax_b, ax_c][crossed_ax]);
    }

    // Find a step magnitude that will snap pos to the desired integer a value (ipos.x)
    // If a_dir is negative then we need to add 1 because we are at the 'top' (larger a) side of block ai
    let a_floor_amt = pos.x - (ipos.x + (ray_dir.x < 0.0) as i32) as f32;
    pos -= ray_dir * a_floor_amt * a_dir as f32;

    // Step ray since we just checked the first block already
    step_ray(&mut ipos, &mut pos);

    // If we didn't hit the starting block, search all other intersecting blocks

    let mut i = 0;
    loop {
        dbg!(ipos);
        // At each iteration of this traversal loop, ipos is
        // the integer position of the block the ray has just come into contact with. This
        // means that if the ray direction is negative, the ipos coordinate will be equal to
        // the ray coord - 1, because the ray has just hit the "top" side of the block while
        // the integer position of the block is the "lower" side. If the ray direction is
        // positive, they will be equal.

        let b_crossed = ipos.y != last_bi;
        let c_crossed = ipos.z != last_ci;
        let b_ib = ipos.y > min_pt.y && ipos.y < max_pt.y; // is b still in bounds
        let c_ib = ipos.z > min_pt.z && ipos.z < max_pt.z; // is c still in bounds

        // Check first possible voxel location, in the case where we cross all three axes, the first voxel
        // intersected is determined by which axis we cross first.
        // NOTE: b_first is used in a non-obvious way, see later use of it for details
        let b_first = if b_crossed {
            if c_crossed {
                let b_dist_to_crossed_pt = ((ipos.y as f32 + ((ray_dir.y < 0.0) as i32) as f32
                    - (pos.y - ray_dir.y))
                    / (ray_dir.y + f32::EPSILON))
                    .abs();
                let c_dist_to_crossed_pt = ((ipos.z as f32 + ((ray_dir.z < 0.0) as i32) as f32
                    - (pos.z - ray_dir.z))
                    / (ray_dir.z + f32::EPSILON))
                    .abs();
                let b_first = b_dist_to_crossed_pt < c_dist_to_crossed_pt; // whether crossed axis b before crossing axis c
                if (b_first && b_ib) || (!b_first && c_ib) {
                    // if b crossed first check intersect voxel [a-a_dir, bi, last_ci] else [a-a_dir, last_bi, ci]
                    let ipos_to_check = ipos
                        - Vector3 {
                            x: a_dir,
                            y: if b_first {
                                0
                            } else {
                                (ray_dir.y > 0.0) as i32 * 2 - 1
                            },
                            z: if b_first {
                                (ray_dir.z > 0.0) as i32 * 2 - 1
                            } else {
                                0
                            },
                        };
                    let idx = vox_idx(&ipos_to_check);
                    if chunk_voxels[idx] != VE::empty() {
                        return hit(tlc, idx, if b_first { ax_b } else { ax_c });
                    }
                }
                b_first
            } else {
                false
            }
        } else {
            true // this value needs to be true when c_crossed, else it's never read
        };

        // If we went out of bounds in either b axis or c axis, break
        if !b_ib && (c_ib || b_first) {
            let mut new_tlc = tlc;
            if ray_dir.y > 0.0 {
                new_tlc.0[ax_b] += 1;
                pos[ax_b] = 0.0;
                ipos[ax_b] = 0;
            } else {
                new_tlc.0[ax_b] -= 1;
                pos[ax_b] = tlc_size as f32;
                ipos[ax_b] = tlc_size - 1;
            }
            return Ok(CastRayInTlcResult::Miss(RayPos {
                tlc,
                pos: pos_xyz(pos),
                ipos: ipos_xyz(ipos),
            }));
        }
        if !c_ib {
            let mut new_tlc = tlc;
            if ray_dir.y > 0.0 {
                new_tlc.0[ax_c] += 1;
                pos[ax_c] = 0.0;
                ipos[ax_c] = 0;
            } else {
                new_tlc.0[ax_c] -= 1;
                pos[ax_c] = tlc_size as f32;
                ipos[ax_c] = tlc_size - 1;
            }
            return Ok(CastRayInTlcResult::Miss(RayPos {
                tlc,
                pos: pos_xyz(pos),
                ipos: ipos_xyz(ipos),
            }));
        }

        // Check second possible voxel location, in the case where we cross two or three axes, the voxel
        // the ray intersects before going into [a-a_dir, bi, ci]
        if b_crossed || c_crossed {
            let ipos_to_check = ipos
                - Vector3 {
                    x: a_dir,
                    y: 0,
                    z: 0,
                };
            let idx = vox_idx(&ipos_to_check);
            if chunk_voxels[idx] != VE::empty() {
                // Reusing b_first here (with augmented meaning) to determine which axis was crossed for this check.
                return hit(tlc, idx, if b_first { ax_c } else { ax_b });
            }
        }

        // Check the final possible voxel location, which we will always intersect
        if ipos.x < max_pt.x && ipos.x > min_pt.y {
            // check intersect voxel [ai, bi, ci]
            let idx = vox_idx(&ipos);
            if chunk_voxels[idx] != VE::empty() {
                return hit(tlc, idx, ax_a);
            }
        }

        last_bi = ipos.y;
        last_ci = ipos.z;
        step_ray(&mut ipos, &mut pos);

        // If we finished loop without hitting a present block or leaving the axis b or c boundary, then we exited
        // due to axis a boundary being reached. In that case, we need to roll back pos by one step.
        if ipos.x > max_pt.x {
            pos -= ray_dir;

            let mut tlc = tlc;
            tlc.0[ax_a] += 1;
            pos[ax_a] = 0.0;
            ipos[ax_a] = 0;
            return Ok(CastRayInTlcResult::Miss(RayPos {
                tlc,
                pos: pos_xyz(pos),
                ipos: ipos_xyz(ipos),
            }));
        }
        if ipos.x < min_pt.x {
            ipos.x += 1;
            pos -= ray_dir;

            let mut tlc = tlc;
            tlc.0[ax_a] -= 1;
            pos[ax_a] = tlc_size as f32;
            ipos[ax_a] = tlc_size - 1;
            return Ok(CastRayInTlcResult::Miss(RayPos {
                tlc,
                pos: pos_xyz(pos),
                ipos: ipos_xyz(ipos),
            }));
        }

        i += 1;
        if i > TRAVERSAL_SAFETY_LIMIT {
            panic!("Ray traversal stuck in infinite loop")
        }
    }
}

pub enum CastRayResult {
    Hit(RayVoxelIntersect),
    Miss,
}

/// Cast a ray and get the first block it intersects. Operates only at LOD 0.
/// Error return means the ray entered an unloaded chunk. Assumes that the
/// starting TLC is the center one.
pub fn cast_ray<VE: VoxelTypeEnum, const N: usize, CE: ChunkEditorVoxels<VE, N>, MD>(
    editor: &mut MemoryGridEditor<CE, MD>,
    // position relative to the bottom corner of the memory grid
    start_pos: VoxelPos<f32>,
    ray_dir: Vector3<f32>,
    chunk_size: ChunkSize,
    largest_chunk_lvl: u8,
) -> Result<CastRayResult, ()> {
    // local_pos should be between 0 and tlc_size in all dims.
    // When we trace the ray, if it goes outisde that, we need to switch chunks

    let tlc_size = chunk_size.size().pow(largest_chunk_lvl as u32) as i32;
    let pos = start_pos.0 - Vector3::from_value((tlc_size as usize * (editor.size / 2 - 1)) as f32);
    let mut ray_pos = RayPos {
        pos,
        ipos: pos.map(|a| a.floor() as i32),
        tlc: editor.center_chunk_pos(),
    };

    for _ in 0..=1 {
        match cast_ray_in_tlc(
            editor,
            ray_pos,
            ray_dir,
            chunk_size,
            largest_chunk_lvl,
            None,
        )? {
            CastRayInTlcResult::Hit(intersect) => return Ok(CastRayResult::Hit(intersect)),
            CastRayInTlcResult::Miss(pos) => {
                ray_pos = pos;
            }
            CastRayInTlcResult::OutOfArea => return Ok(CastRayResult::Miss),
        }
    }

    Ok(CastRayResult::Miss)
}
