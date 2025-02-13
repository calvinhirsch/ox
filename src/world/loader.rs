use crate::world::mem_grid::utils::index_for_pos;
use crate::world::mem_grid::MemoryGridEditor;
use crate::world::{BufferChunkState, TLCPos};
use cgmath::{Array, EuclideanSpace, MetricSpace, Point3, Vector3};
use getset::{CopyGetters, Getters};
use priority_queue::PriorityQueue;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
use std::thread;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ChunkLoadQueueItem<D> {
    pub pos: TLCPos<i64>,
    pub data: D,
}
impl<D> Hash for ChunkLoadQueueItem<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos.0.hash(state);
    }
}

mod layer_chunk {
    use std::cell::UnsafeCell;

    #[derive(Clone, Copy, Debug)]
    pub enum LayerChunkState {
        Valid,
        Invalid, // Data is not being used by any loading thread but is not valid/correct
        Missing, // Data is being used by a chunk loading thread
    }

    /// Chunk data from a single memory grid layer. It is safe to read the state and it is safe to
    /// read the data if the state is not Missing. If the variant is Missing, the data is considered borrowed by a
    /// chunk loading thread, and thus should not be read. The variant should only ever be changed in the main thread
    /// when no loading thread has borrowed this, which should be orchestrated by a chunk loader.
    #[derive(Debug)]
    pub struct LayerChunk<T> {
        data: T,
        state: UnsafeCell<LayerChunkState>,
    }

    impl<T> LayerChunk<T> {
        pub fn new(data: T) -> Self {
            LayerChunk {
                data,
                state: UnsafeCell::new(LayerChunkState::Invalid),
            }
        }

        pub fn new_valid(data: T) -> Self {
            LayerChunk {
                data,
                state: UnsafeCell::new(LayerChunkState::Valid),
            }
        }

        pub fn state(&self) -> LayerChunkState {
            unsafe { *self.state.get() }
        }

        /// Returns referece data if it's valid
        pub fn get(&self) -> Option<&T> {
            match self.state() {
                LayerChunkState::Valid => Some(&self.data),
                _ => None,
            }
        }

        /// Returns mutable reference to data if it's valid
        pub fn get_mut(&mut self) -> Option<&mut T> {
            match self.state() {
                LayerChunkState::Valid => Some(&mut self.data),
                _ => None,
            }
        }

        /// Get data for loading, should be called from within a chunk loading thread
        pub unsafe fn get_mut_for_loading(&mut self) -> &mut T {
            debug_assert!(
                matches!(self.state(), LayerChunkState::Missing),
                "Expected missing, got state {:?}",
                self.state()
            );
            &mut self.data
        }

        pub unsafe fn get_for_loading(&self) -> &T {
            debug_assert!(
                matches!(self.state(), LayerChunkState::Missing),
                "Expected missing, got state {:?}",
                self.state()
            );
            &self.data
        }

        /// Set the state to "invalid"--if the current state is "missing", will return an Err(())
        pub unsafe fn set_invalid(&mut self) -> Result<(), ()> {
            match self.state() {
                LayerChunkState::Missing => Err(()),
                LayerChunkState::Valid | LayerChunkState::Invalid => {
                    *self.state.get_mut() = LayerChunkState::Invalid;
                    Ok(())
                }
            }
        }

        /// Set the state to "missing". State should be "invalid" to do this according to the chunk loading process.
        /// Setting the state to missing means that a chunk loading thread has borrowed this data. `self` is `Pin`
        /// here because if that ever happens, the data must never move, because the chunk loading thread keeps a
        /// raw pointer to this. If the data ever moves, that pointer will be invalidated.
        pub unsafe fn set_missing(&mut self) {
            debug_assert!(
                matches!(self.state(), LayerChunkState::Invalid),
                "Expected invalid, got state {:?}",
                self.state()
            );
            *self.state.get_mut() = LayerChunkState::Missing;
        }

        /// Set the state to "valid". State should be "missing" to do this according to the chunk loading process
        pub unsafe fn set_valid(&mut self) {
            debug_assert!(
                matches!(self.state(), LayerChunkState::Missing),
                "Expected missing, got state {:?}",
                self.state()
            );
            *self.state.get_mut() = LayerChunkState::Valid;
        }
    }
}
pub use layer_chunk::{LayerChunk, LayerChunkState};

/// Chunk data that can be loaded with a `ChunkLoader`. The `ChunkLoader` will first `mark_invalid` when
/// the chunk is queued, then `take_data_for_loading`, send it to a separate thread, load the data, then
/// when loading is complete, `mark_valid` and release its pointer.
pub unsafe trait BorrowChunkForLoading<BC> {
    // TODO: derive macro

    /// Called when chunk is first queued. Data in chunks is assumed to no longer be valid when they are
    /// queued. This method is called when a chunk is queued to mark it invalid so it is not used elsewhere.
    /// This should call `set_invalid` on all `LayerChunk`s. If any of them return `Err(())`, this should
    /// also return that. However, it should not short circuit.
    fn mark_invalid(&mut self) -> Result<(), ()>;

    /// Mark chunk data as taken for loading. This should call `set_missing` on all `LayerChunk`s. Then,
    /// construct and return self, which should be comprised of raw pointers to all the `LayerChunk
    fn take_data_for_loading(
        &mut self,
        // chunk_qi: &ChunkLoadQueueItem<QI>,
        // metadata: &MD,
    ) -> BC;
}

pub unsafe trait BorrowedChunk: Send {
    // TODO: derive macro

    /// Mark all chunk data valid. Called by chunk loader from the main thread once `load` has been run.
    /// Should call `set_valid` on all `LayerChunk`s.
    unsafe fn mark_valid(&mut self);
}

#[derive(Debug, Getters, CopyGetters)]
pub struct ChunkLoader<QI, MD, BC>
where
    BC: BorrowedChunk,
    QI: Eq,
{
    metadata_type: PhantomData<MD>,
    active_threads: Vec<Option<Receiver<BC>>>,
    #[get = "pub"]
    prequeue: Vec<ChunkLoadQueueItem<QI>>,
    #[get = "pub"]
    queue: PriorityQueue<ChunkLoadQueueItem<QI>, u32>,
    #[get_copy = "pub"]
    prequeued_last: usize,
    #[get_copy = "pub"]
    queued_last: usize,
    #[get_copy = "pub"]
    skipped_queue_last: usize,
    #[get_copy = "pub"]
    started_loading_last: usize,
    #[get_copy = "pub"]
    skipped_loading_last: usize,
    #[get_copy = "pub"]
    finished_loading_last: usize,
}

pub struct ChunkLoaderParams {
    pub n_threads: usize,
}

fn vgrid_index(
    start_tlc: TLCPos<i64>,
    grid_size: usize,
    buffer_chunk_states: &[BufferChunkState; 3],
    pos: TLCPos<i64>,
) -> Option<usize> {
    let mut pt = pos.0 - start_tlc.0.to_vec();

    // If position is outside vgrid, it may be a buffer chunk. Check if it is, and if not, return None
    for c in [0, 1, 2] {
        if pt[c] < 0 {
            if pt[c] == -1 && buffer_chunk_states[c] == BufferChunkState::LoadedLower {
                pt[c] = grid_size as i64 - 1;
            } else {
                return None;
            }
        } else if pt[c] >= grid_size as i64 - 1 {
            if pt[c] == grid_size as i64 - 1
                && buffer_chunk_states[c] == BufferChunkState::LoadedUpper
            {
                pt[c] = grid_size as i64 - 1;
            } else {
                return None;
            }
        }
    }

    Some(index_for_pos(pt.cast::<u32>()?, grid_size))
}

impl<QI: Eq, MD, BC: BorrowedChunk> ChunkLoader<QI, MD, BC> {
    pub fn new(params: ChunkLoaderParams) -> Self {
        ChunkLoader {
            metadata_type: PhantomData,
            active_threads: (0..params.n_threads).map(|_| None).collect(),
            prequeue: vec![],
            queue: PriorityQueue::new(),
            prequeued_last: 0,
            queued_last: 0,
            skipped_queue_last: 0,
            started_loading_last: 0,
            skipped_loading_last: 0,
            finished_loading_last: 0,
        }
    }

    pub fn active_loading_threads(&self) -> usize {
        self.active_threads
            .iter()
            .filter_map(|o| o.as_ref())
            .count()
    }

    pub fn print_status(&self) {
        println!(
            "CHUNK LOADER:  in prequeue: {}, in queue: {},  loading: {}",
            self.prequeue.len(),
            self.queue.len(),
            self.active_loading_threads(),
        );
        println!(
            "  Last frame:  prequeued: {}, queued: {}, skipped_queue: {}, started loading: {}, skipped loading: {}, finished loading: {}",
            self.prequeued_last, self.queued_last, self.skipped_queue_last, self.started_loading_last, self.skipped_loading_last, self.finished_loading_last,
        );
    }
}

impl<
        QI: Clone + Send + Eq + std::fmt::Debug + 'static,
        MD: Clone + Send + 'static,
        BC: BorrowedChunk + 'static, // why 'static? it's borrowed data but the refernces should be raw pointers
    > ChunkLoader<QI, MD, BC>
{
    /// Queues new chunks for loading and puts loaded chunks back in memory grid using editor.
    pub fn sync<
        CE: BorrowChunkForLoading<BC>,
        F: Fn(&mut BC, ChunkLoadQueueItem<QI>, MD) + Sync,
    >(
        &mut self,
        start_tlc: TLCPos<i64>,
        editor: &mut MemoryGridEditor<CE, MD>,
        queue: Vec<ChunkLoadQueueItem<QI>>,
        buffer_chunk_states: &[BufferChunkState; 3],
        load: &'static F,
    ) {
        self.prequeued_last = queue.len();
        self.queued_last = 0;
        self.skipped_queue_last = 0;
        self.started_loading_last = 0;
        self.skipped_loading_last = 0;
        self.finished_loading_last = 0;

        // Receive chunks that have finished loading and put them in "chunks"
        for thread_slot in self.active_threads.iter_mut() {
            if let Some(receiver) = thread_slot {
                match receiver.try_recv() {
                    Ok(mut chunk_data) => {
                        self.finished_loading_last += 1;
                        unsafe {
                            chunk_data.mark_valid();
                        };
                        *thread_slot = None;
                    }
                    Err(TryRecvError::Disconnected) => {
                        panic!("Thread disconnected before completing.")
                    }
                    Err(TryRecvError::Empty) => {}
                }
            }
        }

        let grid_size = editor.size;
        let priority = |chunk_pos: TLCPos<i64>| -> u32 {
            u32::MAX
                - (Vector3::from_value(grid_size as f32 / 2.)
                    .cast::<f32>()
                    .unwrap()
                    .distance((chunk_pos.0 - start_tlc.0).cast::<f32>().unwrap())
                    * grid_size as f32) as u32
        };

        let editor_chunk_i = |pos| vgrid_index(start_tlc, grid_size, buffer_chunk_states, pos);

        // Add newly passed chunks to load into the chunk loader prequeue
        self.prequeue.extend(queue);

        // If items in the prequeue are ready to be queued for loading (meaning all their data is available),
        // then add them to the queue.
        for i in (0..self.prequeue.len()).rev() {
            if let Some(chunk_idx) = editor_chunk_i(dbg!(self.prequeue[i].pos)) {
                dbg!(chunk_idx);
                match editor.chunks[chunk_idx].mark_invalid() {
                    Ok(()) => {
                        // All data is present & was able to mark chunk invalid, so add it to priority queue
                        let prio = priority(self.prequeue[i].pos);
                        self.queue.push(self.prequeue.remove(i), prio);
                        self.queued_last += 1;
                    }
                    Err(()) => {}
                }
            } else {
                self.prequeue.remove(i);
                self.skipped_queue_last += 1;
            }
        }

        // Enqueue new chunks for loading until queue is empty or there are no thread slots left
        if !self.queue.is_empty() {
            for thread_slot in self.active_threads.iter_mut() {
                if thread_slot.is_none() {
                    let (item, _) = self.queue.pop().unwrap();
                    let (sender, receiver) = sync_channel(0);

                    // Get index of current chunk. If this returns None, the chunk no longer is relevant
                    // and so we just skip loading it (it remains "invalid")
                    if item.pos.0 == (Point3 { x: 1, y: -1, z: -1 }) {
                        let idx = editor_chunk_i(item.pos);
                        dbg!(&item, idx);
                    }
                    if let Some(chunk_idx) = editor_chunk_i(item.pos) {
                        let meta = editor.metadata().clone();
                        let chunk = &mut editor.chunks[chunk_idx];
                        self.started_loading_last += 1;
                        let mut chunk_data = chunk.take_data_for_loading();
                        thread::spawn(|| {
                            let sender = sender; // move
                            load(&mut chunk_data, item, meta);
                            sender.send(chunk_data).unwrap_or_else(|e| {
                                panic!("Failed to send loaded chunk back to main thread: {}", e)
                            });
                        });

                        *thread_slot = Some(receiver);
                    } else {
                        self.skipped_loading_last += 1;
                    }

                    if self.queue.is_empty() {
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use cgmath::Point3;

    use crate::world::mem_grid::utils::cubed;

    use super::*;

    #[derive(Clone)]
    struct TestChunkEditor(bool);
    struct BorrowedTestChunkEditor(*mut bool);

    unsafe impl BorrowChunkForLoading<BorrowedTestChunkEditor> for TestChunkEditor {
        fn mark_invalid(&mut self) -> Result<(), ()> {
            Ok(())
        }

        fn take_data_for_loading(
            &mut self,
            // chunk_qi: &ChunkLoadQueueItem<QI>,
            // metadata: &MD,
        ) -> BorrowedTestChunkEditor {
            BorrowedTestChunkEditor(&mut self.0)
        }
    }

    unsafe impl Send for BorrowedTestChunkEditor {}
    unsafe impl BorrowedChunk for BorrowedTestChunkEditor {
        unsafe fn mark_valid(&mut self) {}
    }

    #[test]
    fn test_load_all_with_buffers() {
        let mut loader = ChunkLoader::new(ChunkLoaderParams { n_threads: 1 });

        let start_tlc = TLCPos(Point3::<i64> { x: 0, y: 0, z: 0 } - Vector3::from_value(7));
        let mem_grid_size = 16;
        let mut mem_grid_editor = MemoryGridEditor::new(
            vec![TestChunkEditor(false); cubed(mem_grid_size)],
            mem_grid_size,
            start_tlc,
            (),
        );

        for x in -7..=8 {
            for y in -7..=8 {
                for z in -7..=8 {
                    let pos = TLCPos(Point3 { x, y, z });
                    loader.sync(
                        start_tlc,
                        &mut mem_grid_editor,
                        vec![ChunkLoadQueueItem { data: (), pos: pos }],
                        &[
                            BufferChunkState::LoadedUpper,
                            BufferChunkState::LoadedUpper,
                            BufferChunkState::LoadedUpper,
                        ],
                        &|data, _, _| unsafe {
                            assert!(!*data.0);
                            *data.0 = true;
                        },
                    )
                }
            }
        }

        while loader.active_loading_threads() > 0 {
            loader.sync(
                start_tlc,
                &mut mem_grid_editor,
                vec![],
                &[
                    BufferChunkState::LoadedUpper,
                    BufferChunkState::LoadedUpper,
                    BufferChunkState::LoadedUpper,
                ],
                &|data, _, _| unsafe {
                    assert!(!*data.0);
                    *data.0 = true;
                },
            );
            loader.print_status();
        }

        for chunk in mem_grid_editor.chunks.iter() {
            assert!(chunk.0)
        }
    }

    #[test]
    fn test_load_all_without_buffers() {
        let mut loader = ChunkLoader::new(ChunkLoaderParams { n_threads: 1 });

        let start_tlc = TLCPos(Point3::<i64> { x: 0, y: 0, z: 0 } - Vector3::from_value(7));
        let mem_grid_size = 16;
        let mut mem_grid_editor = MemoryGridEditor::new(
            vec![TestChunkEditor(false); cubed(mem_grid_size)],
            mem_grid_size,
            start_tlc,
            (),
        );

        for x in -7..=7 {
            for y in -7..=7 {
                for z in -7..=7 {
                    let pos = TLCPos(Point3 { x, y, z });
                    loader.sync(
                        start_tlc,
                        &mut mem_grid_editor,
                        vec![ChunkLoadQueueItem { data: (), pos: pos }],
                        &[
                            BufferChunkState::Unloaded,
                            BufferChunkState::Unloaded,
                            BufferChunkState::Unloaded,
                        ],
                        &|data, _, _| unsafe {
                            assert!(!*data.0);
                            *data.0 = true;
                        },
                    )
                }
            }
        }

        while loader.active_loading_threads() > 0 {
            loader.sync(
                start_tlc,
                &mut mem_grid_editor,
                vec![],
                &[
                    BufferChunkState::Unloaded,
                    BufferChunkState::Unloaded,
                    BufferChunkState::Unloaded,
                ],
                &|data, _, _| unsafe {
                    assert!(!*data.0);
                    *data.0 = true;
                },
            );
            loader.print_status();
        }

        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let idx = x + y * 16 * 16 + z * 16;
                    if x == 15 || y == 15 || z == 15 {
                        assert!(!mem_grid_editor.chunks[idx].0);
                    } else {
                        assert!(mem_grid_editor.chunks[idx].0);
                    }
                }
            }
        }
    }

    // #[test]
    // fn test_indexing() {
    //     for x in -7..=7 {
    //         for y in -7..=8 {
    //             for z in -7..=8 {
    //                 let pos = TLCPos(Point3 { x, y, z });
    //                 let index =
    //             }
    //         }
    //     }
    // }
}
