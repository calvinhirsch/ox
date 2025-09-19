use cgmath::{Array, InnerSpace, Point3, Vector3};
use std::ops::AddAssign;

use crate::{
    voxel_type::VoxelTypeEnum,
    world::{
        mem_grid::{
            utils::{ChunkSize, VoxelPosInLod},
            voxel::grid::ChunkVoxelEditor,
            MemoryGridEditor,
        },
        TlcPos, VoxelPos, VoxelVector,
    },
};

pub trait ChunkEditorVoxels<VE: VoxelTypeEnum, const N: usize> {
    fn voxels(&self) -> &ChunkVoxelEditor<'_, VE, N>;
}

pub struct VoxelFace {
    pub ax: u8,    // 0, 1, or 2
    pub dir: bool, // true for positive, false for negative
}

impl VoxelFace {
    /// Get delta position to the voxel this face faces
    pub fn delta(&self) -> VoxelVector<i32> {
        let mut v = Vector3::from_value(0);
        v[self.ax as usize] = (self.dir as i32) * 2 - 1;
        VoxelVector(v)
    }
}

pub struct RayVoxelIntersect {
    pub tlc: TlcPos<i64>,
    pub pos: VoxelPos<u32>,
    pub index: usize,
    pub face: VoxelFace,
}

pub struct RayPos {
    // top level chunk
    tlc: TlcPos<i64>,
    // position within tlc
    pos: Point3<f32>,
    // which voxel is being examined during traversal
    ipos: Point3<i32>,
    last_crossed_ax: Option<usize>,
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
    RayPos {
        tlc,
        pos,
        mut ipos,
        last_crossed_ax,
    }: RayPos,
    ray_dir: Vector3<f32>,
    chunk_size: ChunkSize,
    largest_chunk_lvl: u8,
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
    dbg!(ipos);

    // ENHANCEMENT: a bunch of this gets repeated when called from cast_ray

    let ray_dir = ray_dir.normalize();
    let tlc_size = chunk_size.size().pow(largest_chunk_lvl as u32) as i32;

    // Identify axis (x, y, or z) the ray is most parallel to and set to axis A (with others set to B, C)
    let (ax_a, ax_b, ax_c) = if ray_dir.x.abs() > ray_dir.y.abs() {
        if ray_dir.x.abs() > ray_dir.z.abs() {
            (0, 1, 2)
        } else {
            (2, 0, 1)
        }
    } else {
        if ray_dir.y.abs() > ray_dir.z.abs() {
            (1, 2, 0)
        } else {
            (2, 0, 1)
        }
    };

    // Put everything in abc coords
    // Shadowing the vars here so the XYZ versions are not accidentally used

    let xyz_to_abc = {
        let mut v = [0, 0, 0];
        v[ax_a] = 0;
        v[ax_b] = 1;
        v[ax_c] = 2;
        v
    };
    let mut crossed_ax = match last_crossed_ax {
        None => 0,
        Some(ax) => xyz_to_abc[ax],
    };

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

    // Bounds of traversal
    let min_pt = Vector3::from_value(0i32);
    let max_pt = Vector3::from_value(tlc_size - 1);

    let a_dir = (ray_dir.x > 0.0) as i32 * 2 - 1;

    // helpers

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

    let vox_idx = |ipos: Point3<i32>| {
        dbg!("ipos to check", &ipos);
        VoxelPosInLod::in_full_lod(VoxelPos(ipos_xyz(ipos).cast::<u32>().unwrap()))
            .index(chunk_size, largest_chunk_lvl)
    };

    let hit = |tlc, voxel_index, crossed_ax_abc, ipos: Point3<i32>| {
        Ok(CastRayInTlcResult::Hit(RayVoxelIntersect {
            tlc: tlc,
            pos: VoxelPos(ipos_xyz(ipos).cast::<u32>().unwrap()),
            index: voxel_index,
            face: VoxelFace {
                ax: [ax_a, ax_b, ax_c][crossed_ax_abc] as u8,
                dir: ray_dir[crossed_ax_abc] < 0.0,
            },
        }))
    };

    // Traverse chunk

    let mut i = 0;
    loop {
        dbg!(ipos);
        // At each iteration of this traversal loop, ipos is
        // the integer position of the block the ray has just come into contact with. This
        // means that if the ray direction is negative, the ipos coordinate will be equal to
        // the ray coord - 1, because the ray has just hit the "top" side of the block while
        // the integer position of the block is the "lower" side. If the ray direction is
        // positive, they will be equal.

        // FIRST CHECK: block at ipos
        // Here, if we just entered this chunk by going out of bounds of the previous chunk in
        // axis B or C, ray origin will be at the point we crossed, not at a round A value.
        // This means that ipos may also not be at a block where we cross the A axis border.
        let idx = vox_idx(ipos);
        if chunk_voxels[idx] != VE::empty().id() {
            return hit(tlc, idx, crossed_ax, ipos);
        }

        // Step the ray forward to the next integer value in the A axis
        // we will then check all the blocks it crosses through

        let last_bi = ipos.y;
        let last_ci = ipos.z;
        // Step A axis by 1
        ipos.x += a_dir;
        // Snap ray origin to appropriate A axis value along ray direction.
        // As noted above, this is because it may not have started at a round A value.
        pos.add_assign(
            ray_dir
                * (ipos.x as f32 - pos.x + (ray_dir.x < 0.0) as i32 as f32)
                * (1.0 / (ray_dir.x + f32::EPSILON)),
        );
        // set B and C axis values to match ray origin
        ipos.y = pos.y.floor() as i32;
        ipos.z = pos.z.floor() as i32;

        let b_crossed = ipos.y != last_bi;
        let c_crossed = ipos.z != last_ci;
        let b_ib = ipos.y > min_pt.y && ipos.y < max_pt.y; // is b still in bounds
        let c_ib = ipos.z > min_pt.z && ipos.z < max_pt.z; // is c still in bounds

        // SECOND CHECK: in the case where we cross all three axes, the second voxel
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
                    let idx = vox_idx(ipos_to_check);
                    if chunk_voxels[idx] != VE::empty().id() {
                        return hit(tlc, idx, if b_first { 1 } else { 2 }, ipos_to_check);
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
        let mut oob_b_or_c = |ax_xyz: usize, ax_abc: usize| {
            let mut new_tlc = tlc;
            ipos.x -= a_dir;
            if ray_dir[ax_abc] > 0.0 {
                new_tlc.0[ax_xyz] += 1;
                // pos[ax_abc] = 0.0;
                // ipos[ax_abc] = 0;
            } else {
                new_tlc.0[ax_xyz] -= 1;
                // pos[ax_abc] = tlc_size as f32;
                // ipos[ax_abc] = tlc_size - 1;
            }

            // Our position is going to be at the next round value in A axis, but
            // we want it to be at the point we crossed the axis that went OOB
            let target_pos = if ray_dir[ax_abc] > 0.0 { tlc_size } else { 0 };
            let delta = (target_pos as f32 - pos[ax_abc]) / (ray_dir[ax_abc] + f32::EPSILON);
            pos += ray_dir * delta;
            pos[ax_abc] = target_pos as f32;
            ipos = pos.map(|a| a.floor() as i32);
            ipos[ax_abc] = target_pos;

            dbg!("oob", ax_abc);

            Ok(CastRayInTlcResult::Miss(RayPos {
                tlc: new_tlc,
                pos: pos_xyz(pos),
                ipos: ipos_xyz(ipos),
                last_crossed_ax: Some(ax_xyz),
            }))
        };
        if !b_ib && (c_ib || b_first) {
            return oob_b_or_c(ax_b, 1);
        }
        if !c_ib {
            return oob_b_or_c(ax_c, 2);
        }

        // THIRD CHECK: in the case where we cross two or three axes, the voxel
        // the ray intersects before crossing into the next integer A-axis value
        if b_crossed || c_crossed {
            let ipos_to_check = ipos
                - Vector3 {
                    x: a_dir,
                    y: 0,
                    z: 0,
                };
            let idx = vox_idx(ipos_to_check);
            if chunk_voxels[idx] != VE::empty().id() {
                // Reusing b_first here (with augmented meaning) to determine which axis was crossed for this check.
                return hit(tlc, idx, if b_first { 2 } else { 1 }, ipos_to_check);
            }
        }

        // Check if we are out of bounds

        crossed_ax = 0;

        // If we finished loop without hitting a present block or leaving the axis b or c boundary, then we exited
        // due to axis a boundary being reached. In that case, we need to roll back pos by one step.
        if ipos.x > max_pt.x {
            let mut tlc = tlc;
            tlc.0[ax_a] += 1;
            pos.x = 0.0;
            ipos = pos.map(|a| a.floor() as i32);
            ipos.x = 0;
            return Ok(CastRayInTlcResult::Miss(RayPos {
                tlc,
                pos: pos_xyz(pos),
                ipos: ipos_xyz(ipos),
                last_crossed_ax: Some(ax_a),
            }));
        }
        if ipos.x < min_pt.x {
            let mut tlc = tlc;
            tlc.0[ax_a] -= 1;
            pos.x = tlc_size as f32;
            ipos = pos.map(|a| a.floor() as i32);
            ipos.x = tlc_size - 1;
            return Ok(CastRayInTlcResult::Miss(RayPos {
                tlc,
                pos: pos_xyz(pos),
                ipos: ipos_xyz(ipos),
                last_crossed_ax: Some(ax_a),
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
        last_crossed_ax: None,
    };

    for _ in 0..=1 {
        match cast_ray_in_tlc(editor, ray_pos, ray_dir, chunk_size, largest_chunk_lvl)? {
            CastRayInTlcResult::Hit(intersect) => return Ok(CastRayResult::Hit(intersect)),
            CastRayInTlcResult::Miss(pos) => {
                ray_pos = pos;
            }
            CastRayInTlcResult::OutOfArea => return Ok(CastRayResult::Miss),
        }
    }

    Ok(CastRayResult::Miss)
}
