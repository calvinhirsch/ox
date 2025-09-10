use cgmath::{Array, Point3, Vector3};
use getset::CopyGetters;
use std::ops::{DerefMut, Index, IndexMut, Mul};

use crate::world::VoxelPos;

pub fn squared<T: Copy + Mul<Output = T>>(x: T) -> T {
    x * x
}
pub fn cubed<T: Copy + Mul<Output = T>>(x: T) -> T {
    x * x * x
}

/// Standard indexing scheme to store a 3D cubic array in a 1D array. `size` is the length of the cube side.
pub const fn index_for_pos(pos: Point3<u32>, size: usize) -> usize {
    pos.x as usize + pos.y as usize * size as usize * size as usize + pos.z as usize * size as usize
}

pub fn amod(n: Point3<i64>, d: usize) -> Point3<usize> {
    (((n % d as i64) + Vector3::<i64>::from_value(d as i64)) % d as i64)
        .cast::<usize>()
        .unwrap()
}

pub const fn pos_for_index(index: usize, size: usize) -> Point3<usize> {
    Point3 {
        x: index % size,
        y: index / (size * size),
        z: (index % (size * size)) / size,
    }
}

#[derive(Debug, Clone)]
/// Position relative to bottom corner of current TLC in units of this LOD. For example,
/// if LOD  lvl=0 and sublvl=2, pos_in_tlc should be in units 4x larger than highest
/// fidelity voxels (i.e. 4 in world coords).
pub struct VoxelPosInLOD {
    pub pos: Point3<u32>, // position in units of current LOD voxels
    pub lvl: u8,
    pub sublvl: u8,
}
impl VoxelPosInLOD {
    pub fn in_full_lod(pos: VoxelPos<u32>) -> Self {
        Self {
            pos: pos.0,
            lvl: 0,
            sublvl: 0,
        }
    }

    /// Index in top level chunk. This is not trivial to compute.
    pub fn index(&self, chunk_size: ChunkSize, largest_chunk_lvl: u8) -> usize {
        // ENHANCEMENT: make largest_chunk_lvl const to allow loop unrolling here
        let mut idx = 0usize;
        for lvl in (self.lvl + 1..largest_chunk_lvl).rev() {
            idx *= cubed(chunk_size.size()) as usize;
            // block size for this level in units of self.lvl, self.sublvl
            let lvl_block_size = 1u32 << (chunk_size.exp() * (lvl - self.lvl) - self.sublvl);
            let pos_at_lvl = (self.pos / lvl_block_size as u32) % chunk_size.size() as u32;
            idx += index_for_pos(pos_at_lvl, chunk_size.size());
        }

        // e.g., last_chunk_lvl_size = 4 if sublvl == 1 (voxels of size 2x2x2) and chunk size == 8
        let last_chunk_lvl_size = 1usize << (chunk_size.exp() - self.sublvl);
        idx *= cubed(last_chunk_lvl_size);
        idx += index_for_pos(self.pos % last_chunk_lvl_size as u32, last_chunk_lvl_size);

        idx
    }
}

pub struct IteratorWithIndexing<I, T>
where
    I: IndexMut<usize>,
    T: DerefMut<Target = I>,
{
    content: T,
    len: usize,
    i: usize,
}

impl<I, T> IteratorWithIndexing<I, T>
where
    I: IndexMut<usize>,
    T: DerefMut<Target = I>,
{
    pub fn new(content: T, len: usize) -> Self {
        Self { content, len, i: 0 }
    }

    pub fn apply<F: FnMut(usize, &mut I::Output, &Self)>(&mut self, mut f: F) {
        for i in 0..self.len {
            self.i = i;
            unsafe {
                let s = self as *const Self;
                f(i, &mut self.content[i], &*s);
            }
        }
    }
}

impl<I, T> Index<usize> for IteratorWithIndexing<I, T>
where
    I: IndexMut<usize>,
    T: DerefMut<Target = I>,
{
    type Output = I::Output;

    fn index(&self, index: usize) -> &Self::Output {
        if self.i == index {
            panic!("Tried to index same element as currently mutably borrowed.")
        }
        self.content.index(index)
    }
}

#[derive(Debug, CopyGetters, Clone, Copy)]
pub struct ChunkSize {
    #[get_copy = "pub"]
    exp: u8,
}
impl ChunkSize {
    /// Chunk size will be 2^`exp`
    pub const fn new(exp: u8) -> Self {
        Self { exp }
    }

    /// Length in voxels per side. Total voxels in the chunk would be this cubed.
    pub fn size(&self) -> usize {
        1usize << self.exp()
    }

    /// Number of chunk sublevels that would exist given this chunk size
    pub fn n_sublvls(&self) -> u8 {
        self.exp() - 1
    }
}
