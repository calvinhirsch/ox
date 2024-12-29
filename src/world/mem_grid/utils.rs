use cgmath::{Array, EuclideanSpace, Point3, Vector3};
use std::ops::{DerefMut, Index, IndexMut, Mul};

pub fn squared<T: Copy + Mul<Output = T>>(x: T) -> T {
    x * x
}
pub fn cubed<T: Copy + Mul<Output = T>>(x: T) -> T {
    x * x * x
}

/// Standard indexing scheme to store a 3D structure in a 1D array
pub const fn index_for_pos(pos: Vector3<usize>, size: usize) -> usize {
    pos.x + pos.y * size * size + pos.z * size
}

pub fn amod(n: Vector3<i64>, d: usize) -> Vector3<usize> {
    (((n % d as i64) + Vector3::<i64>::from_value(d as i64)) % d as i64)
        .cast::<usize>()
        .unwrap()
}

pub const fn pos_for_index(index: usize, size: usize) -> Vector3<usize> {
    Vector3 {
        x: index % size,
        y: index / (size * size),
        z: (index % (size * size)) / size,
    }
}

/// Get the index of a point at any level/LOD within a top level chunk.
///
/// pos_in_tlc: Position relative to bottom corner of current TLC in units of lvl and lod. For
///             example, if lvl=0 and lod=2, pos_in_tlc should be in units 4x larger than highest
///             fidelity voxels (i.e. 4 in world coords).
pub fn index_for_pos_in_tlc(
    pos_in_tlc: Point3<u32>,
    chunk_size: usize,
    n_chunk_lvls: usize,
    chunk_lvl: usize,
    lod: usize,
) -> usize {
    let mut idx = 0;
    for lvl in (chunk_lvl + 1..n_chunk_lvls).rev() {
        idx *= cubed(chunk_size);
        let lvl_block_size = chunk_size.pow(lvl as u32);
        let pos_at_lvl = (pos_in_tlc / lvl_block_size as u32) % chunk_size as u32;
        idx += index_for_pos(pos_at_lvl.to_vec().cast::<usize>().unwrap(), chunk_size);
    }

    let last_chunk_lvl_size = chunk_size / 2usize.pow(lod as u32);
    idx += index_for_pos(
        (pos_in_tlc % last_chunk_lvl_size as u32)
            .to_vec()
            .cast::<usize>()
            .unwrap(),
        last_chunk_lvl_size,
    );

    idx
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
