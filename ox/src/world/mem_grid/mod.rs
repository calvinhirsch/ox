use std::ops::Range;

use crate::world::{BufferChunkState, TlcPos};
use crate::{loader::ChunkLoadQueueItem, world::TlcVector};
use cgmath::{Array, EuclideanSpace, MetricSpace, Point3, Vector3};
use derive_new::new;
use getset::CopyGetters;

pub mod layer;
mod layer_set;
pub mod utils;
pub mod voxel;

#[derive(new, CopyGetters, Debug)]
pub struct ShiftGridAxisVal {
    #[get_copy = "pub"]
    chunks: i32, // number of chunks to shift by
    #[get_copy = "pub"]
    preloaded_first: bool, // whether a chunk was preloaded in buffer chunks--if true, you can skip loading the first one
}
impl ShiftGridAxisVal {
    fn n_to_load(&self) -> i32 {
        self.chunks - self.preloaded_first as i32
    }

    /// When shifting this axis, what range of chunks should be loaded
    fn load_range_when_shifting(&self, active_grid_size: usize) -> Range<i32> {
        if self.chunks() < 0 {
            0..self.n_to_load()
        } else {
            active_grid_size as i32 - self.n_to_load()..active_grid_size as i32
        }
    }
}

/// What to do for each axis when shifting a memory grid.
#[derive(Debug)]
pub enum ShiftGridAxis {
    // Shift the grid by this amount
    Shift(ShiftGridAxisVal),
    // Don't shift the grid in this dimension, but if other dimensions are being loaded, load the upper buffer chunks in
    // this dim to maintain the current buffer chunk states.
    MaintainUpperLoadedBufferChunks,
    // Same as above but for lower buffer chunks.
    MaintainLowerLoadedBufferChunks,
    LoadUpperBufferChunks,
    LoadLowerBufferChunks,
    DoNothing,
}
impl ShiftGridAxis {
    pub fn as_shift(&self) -> Option<&ShiftGridAxisVal> {
        match self {
            ShiftGridAxis::Shift(shift) => Some(shift),
            _ => None,
        }
    }

    /// When shifting/loading along this axis, what chunk range needs to be loaded
    fn load_range_for_main_axis(&self, active_grid_size: usize) -> Option<Range<i32>> {
        match self {
            ShiftGridAxis::Shift(shift) => Some(shift.load_range_when_shifting(active_grid_size)),
            ShiftGridAxis::MaintainUpperLoadedBufferChunks
            | ShiftGridAxis::MaintainLowerLoadedBufferChunks => None,
            ShiftGridAxis::LoadUpperBufferChunks => {
                Some(active_grid_size as i32..active_grid_size as i32 + 1)
            }
            ShiftGridAxis::LoadLowerBufferChunks => Some(-1..0),
            ShiftGridAxis::DoNothing => None,
        }
    }

    /// When shifting/loading along another axis, what chunk range needs to be loaded in this axis.
    /// Considerations:
    /// 1. Buffer chunks: if buffer chunks are loaded then we need to load new ones in parallel with the shift.
    /// 2. Multiple shifts: if this axis is also being shifted, the chunk range to load should reflect this. This
    ///     creates a problem where corner/edge chunks might be double counted by each axis's individual shift.
    ///     Thus, `load_overlapping` can be used to set whether to include these in the range.
    fn load_range_for_other_axis(
        &self,
        active_grid_size: usize,
        load_overlapping: bool,
    ) -> Range<i32> {
        match self {
            ShiftGridAxis::DoNothing => 0..active_grid_size as i32,
            ShiftGridAxis::MaintainUpperLoadedBufferChunks => 0..active_grid_size as i32 + 1,
            ShiftGridAxis::MaintainLowerLoadedBufferChunks => -1..active_grid_size as i32,
            ShiftGridAxis::LoadUpperBufferChunks => {
                0..(active_grid_size as i32) + (if load_overlapping { 1 } else { 0 })
            }
            ShiftGridAxis::LoadLowerBufferChunks => {
                0 - (if load_overlapping { 1 } else { 0 })..active_grid_size as i32
            }
            ShiftGridAxis::Shift(shift_val) => {
                if shift_val.chunks > 0 {
                    shift_val.chunks..(if load_overlapping {
                        active_grid_size as i32 + shift_val.chunks
                    } else {
                        active_grid_size as i32
                    })
                } else {
                    (if load_overlapping {
                        shift_val.chunks
                    } else {
                        0
                    })..(active_grid_size as i32 - shift_val.chunks)
                }
            }
        }
    }
}

fn abc_pos<T: Into<i64>>(av: T, bv: T, cv: T, a: usize, b: usize, c: usize) -> TlcPos<i64> {
    let mut chunk = Point3::<i64> { x: 0, y: 0, z: 0 };
    chunk[a] = av.into();
    chunk[b] = bv.into();
    chunk[c] = cv.into();
    TlcPos(chunk)
}

#[derive(Debug)]
pub struct MemGridShift([ShiftGridAxis; 3]);
impl MemGridShift {
    pub fn new(axes: [ShiftGridAxis; 3]) -> Option<Self> {
        if axes.iter().all(|x| {
            matches!(
                *x,
                ShiftGridAxis::DoNothing
                    | ShiftGridAxis::MaintainLowerLoadedBufferChunks
                    | ShiftGridAxis::MaintainUpperLoadedBufferChunks
            )
        }) {
            None
        } else {
            Some(MemGridShift(axes))
        }
    }

    pub fn offset_delta(&self) -> Vector3<i32> {
        Vector3 {
            x: self.0[0].as_shift().map(|s| s.chunks).unwrap_or(0),
            y: self.0[1].as_shift().map(|s| s.chunks).unwrap_or(0),
            z: self.0[2].as_shift().map(|s| s.chunks).unwrap_or(0),
        }
    }

    pub fn collect_chunks_to_load<O, F: Fn(TlcPos<i64>) -> O>(
        &self,
        mem_grid_size: usize,
        start_tlc: TlcPos<i64>,
        f: F,
    ) -> Vec<O> {
        // ENHANCEMENT: Do this without all the collect()s in the middle, causes closure escape problems though

        let active_grid_size = mem_grid_size - 1;

        // Note: when shifting multiple axes at once, this scheme would queue the corner chunks to load twice.
        // The `load_overlapping_*` bools are to make sure this only happens once without needing to dedup after.
        [
            (0, (1, true), (2, true)),
            (1, (2, true), (0, false)),
            (2, (0, false), (1, false)),
        ]
        .into_iter()
        .filter_map(|(a, (b, load_overlapping_b), (c, load_overlapping_c))| {
            self.0[a]
                .load_range_for_main_axis(active_grid_size)
                .map(|range| {
                    range
                        .flat_map(|av| {
                            self.0[b]
                                .load_range_for_other_axis(active_grid_size, load_overlapping_b)
                                .into_iter()
                                .flat_map(|bv| {
                                    self.0[c]
                                        .load_range_for_other_axis(
                                            active_grid_size,
                                            load_overlapping_c,
                                        )
                                        .into_iter()
                                        .map(|cv| {
                                            f(TlcPos(
                                                abc_pos(av, bv as i32, cv as i32, a, b, c).0
                                                    + start_tlc.0.to_vec(),
                                            ))
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
        })
        .flatten()
        .collect()
    }
}

pub trait MemoryGrid: Sized {
    /// Size including any buffer chunks
    fn size(&self) -> usize;

    fn start_tlc(&self) -> TlcPos<i64>;

    fn center_chunk_pos(&self) -> TlcPos<i64> {
        TlcPos(self.start_tlc().0 + Vector3::from_value(self.size() as i64 / 2 - 1))
    }

    fn chunk_vgrid_pos_in(
        global_tlc_pos: TlcPos<i64>,
        grid_start_tlc: TlcPos<i64>,
    ) -> Option<TlcVector<usize>> {
        (global_tlc_pos.0 - grid_start_tlc.0.to_vec())
            .cast::<usize>()
            .map(|v| TlcVector(v.to_vec()))
    }

    fn chunk_vgrid_pos(&self, global_tlc_pos: TlcPos<i64>) -> Option<TlcVector<usize>> {
        Self::chunk_vgrid_pos_in(global_tlc_pos, self.start_tlc())
    }

    fn chunk_loading_priority(&self, chunk_pos: TlcPos<i64>) -> u32 {
        let center_pos = Vector3::from_value((self.size() - 1) as f32 / 2.);
        let chunk_pos = (chunk_pos.0 - self.start_tlc().0).map(|a| a as f32);
        u32::MAX - (center_pos.distance(chunk_pos) * 10.0) as u32
    }
}

pub trait MemoryGridLoadChunks {
    type ChunkLoadQueueItemData;

    /// Queue all chunks in memory grid to be loaded. Does not queue buffer chunks or change their state.
    fn queue_load_all(&mut self) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;

    /// Shift this memory grid. This should modify its offsets and queue new chunks to load if the shift is nonzero.
    /// This doesn't need to invalidate chunks it returns as this should be done by the chunk loader. This also
    /// specified what to do with buffer chunks, whether that is to maintain them when shifting or load new ones.
    fn shift(
        &mut self,
        shift: &MemGridShift,
    ) -> Vec<ChunkLoadQueueItem<Self::ChunkLoadQueueItemData>>;
}

pub trait EditMemoryGridChunk<M = ()>: MemoryGrid {
    type ChunkEditor<'a>
    where
        Self: 'a;

    // fn edit_chunk_at_vgrid_pos(
    //     &mut self,
    //     vgrid_pos: TlcVector<usize>,
    //     vgrid_size: usize,
    // ) -> Option<Self::ChunkEditor<'_>>;

    // fn edit_chunk_at_grid_pos(&mut self, grid_pos: TlcVector<usize>) -> Self::ChunkEditor<'_> {
    //     self.edit_chunk_at_vgrid_pos(grid_pos, self.size()).unwrap()
    // }

    fn edit_chunk(
        &mut self,
        pos: TlcPos<i64>,
        buffer_chunk_states: [BufferChunkState; 3],
    ) -> Option<Self::ChunkEditor<'_>>;
}
