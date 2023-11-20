use cgmath::{Array, Vector3};

pub const fn cubed(x: usize) -> usize {
    x * x * x
}

/// Standard indexing scheme to store a 3D structure in a 1D array
pub const fn pos_index(pos: Vector3<usize>, size: usize) -> usize {
    pos.x + pos.y * size * size + pos.z * size
}

pub const fn amod(n: Vector3<i64>, d: usize) -> Vector3<usize> {
    (((n % d as i64) + Vector3::<i64>::from_value(d as i64)) % d as i64).cast::<usize>().unwrap()
}

pub const fn pos_for_index(index: usize, size: usize) -> Vector3<usize> {
    return Vector3 {
        x: index % size,
        y: index / (size*size),
        z: (index % (size*size)) / size,
    }
}