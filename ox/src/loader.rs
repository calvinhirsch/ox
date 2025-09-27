use crate::world::mem_grid::{EditMemoryGridChunk, MemoryGrid, MemoryGridLoadChunks};
use crate::world::{TlcPos, World};
use getset::{CopyGetters, Getters};
use priority_queue::PriorityQueue;
use std::hash::{Hash, Hasher};
use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
use std::thread;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ChunkLoadQueueItem<D> {
    pub pos: TlcPos<i64>,
    pub data: D,
}
impl<D> Hash for ChunkLoadQueueItem<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos.0.hash(state);
    }
}

mod layer_chunk {
    use std::cell::UnsafeCell;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
                LayerChunkState::Valid => {
                    *self.state.get_mut() = LayerChunkState::Invalid;
                    Ok(())
                }
                LayerChunkState::Invalid => Ok(()),
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
pub unsafe trait BorrowChunkForLoading<BC, QI> {
    // TODO: derive macro

    /// Called when chunk is ready to be loaded to see if the load still needs to happen.
    fn should_still_load(&self, queue_item: &QI) -> bool;

    /// Called when chunk is first queued. Data in chunks is assumed to no longer be valid when they are
    /// queued. This method is called when a chunk is queued to mark it invalid so it is not used elsewhere.
    /// This should call `set_invalid` on all `LayerChunk`s. If any of them return `Err(())`, this should
    /// also return that. However, it should not short circuit.
    fn mark_invalid(&mut self) -> Result<(), ()>;

    /// Mark chunk data as taken for loading. This should call `set_missing` on all `LayerChunk`s. Then,
    /// construct and return self, which should be comprised of raw pointers to all the `LayerChunk
    fn take_data_for_loading(
        &mut self,
        queue_item: &QI,
        // metadata: &MD,
    ) -> BC;
}

pub unsafe trait BorrowedChunk: Send {
    type MemoryGrid;

    // TODO: derive macro

    /// Things to do when done loading. Called by chunk loader from the main thread once `load` has been run.
    /// Should always include calling `set_valid` on all `LayerChunk`s. May also include additional steps like,
    /// for voxel data, setting up a transfer region to update the chunk data on the GPU.
    unsafe fn done_loading(&mut self, grid: &mut Self::MemoryGrid);
}

#[derive(Debug, Getters, CopyGetters)]
pub struct ChunkLoader<QI: Eq, BC> {
    active_threads: Vec<Option<Receiver<BC>>>,
    #[get = "pub"]
    queue: PriorityQueue<ChunkLoadQueueItem<QI>, u32>,
    #[get_copy = "pub"]
    queued_last: usize,
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

impl<QI: Eq, BC: BorrowedChunk> ChunkLoader<QI, BC> {
    pub fn new(params: ChunkLoaderParams) -> Self {
        ChunkLoader {
            active_threads: (0..params.n_threads).map(|_| None).collect(),
            queue: PriorityQueue::new(),
            queued_last: 0,
            started_loading_last: 0,
            skipped_loading_last: 0,
            finished_loading_last: 0,
        }
    }

    pub fn enqueue(&mut self, chunk: ChunkLoadQueueItem<QI>, priority: u32) {
        self.queue.push(chunk, priority);
        self.queued_last += 1;
    }

    pub fn active_loading_threads(&self) -> usize {
        self.active_threads
            .iter()
            .filter_map(|o| o.as_ref())
            .count()
    }

    pub fn print_status(&self) {
        println!(
            "CHUNK LOADER:  in queue: {},  loading: {}",
            self.queue.len(),
            self.active_loading_threads(),
        );
        println!(
            "  Last frame: queued: {}, started loading: {}, skipped loading: {}, finished loading: {}",
            self.queued_last, self.started_loading_last, self.skipped_loading_last, self.finished_loading_last,
        );
    }
}

impl<QI, BC> ChunkLoader<QI, BC>
where
    BC: BorrowedChunk + 'static, // why 'static? it's borrowed data but the refernces should be raw pointers,
    BC::MemoryGrid: MemoryGrid + MemoryGridLoadChunks<ChunkLoadQueueItemData = QI>,
    QI: Clone + Send + Eq + std::fmt::Debug + 'static,
{
    /// Queues new chunks for loading and puts loaded chunks back in memory grid using editor.
    pub fn sync<F, LP, M>(
        &mut self,
        world: &mut World<BC::MemoryGrid>,
        load: &'static F,
        load_params: LP,
    ) where
        BC::MemoryGrid: EditMemoryGridChunk<M>,
        for<'a> <BC::MemoryGrid as EditMemoryGridChunk<M>>::ChunkEditor<'a>:
            BorrowChunkForLoading<BC, QI>,
        LP: Clone + Send + 'static,
        F: Fn(&mut BC, ChunkLoadQueueItem<QI>, LP) + Sync,
    {
        self.queued_last = 0;
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
                            chunk_data.done_loading(&mut world.mem_grid);
                        }
                        *thread_slot = None;
                    }
                    Err(TryRecvError::Disconnected) => {
                        panic!("Thread disconnected before completing.")
                    }
                    Err(TryRecvError::Empty) => {}
                }
            }
        }

        // Enqueue new chunks for loading until queue is empty or there are no thread slots left
        if !self.queue.is_empty() {
            let mut requeue = vec![];
            'threads: for thread_slot in self.active_threads.iter_mut() {
                if thread_slot.is_none() {
                    loop {
                        let (item, prio) = self.queue.pop().unwrap();
                        let (sender, receiver) = sync_channel(0);

                        // Get index of current chunk. If this returns None, the chunk no longer is relevant
                        // and so we just skip loading it (it remains "invalid")
                        dbg!(&item);
                        let skipped = if let Some(mut chunk) = world.edit_chunk(item.pos) {
                            if chunk.should_still_load(&item.data) {
                                match chunk.mark_invalid() {
                                    Ok(()) => {
                                        self.started_loading_last += 1;
                                        dbg!("loading!");
                                        let mut chunk_data =
                                            chunk.take_data_for_loading(&item.data);
                                        let lp = load_params.clone();
                                        thread::spawn(|| {
                                            let sender = sender; // move
                                            load(&mut chunk_data, item, lp);
                                            sender.send(chunk_data).unwrap_or_else(|e| {
                                                panic!(
                                                    "Failed to send loaded chunk back to main thread: {}",
                                                    e
                                                )
                                            });
                                        });

                                        *thread_slot = Some(receiver);
                                        break;
                                    }
                                    Err(()) => {
                                        requeue.push((item, prio));
                                        false
                                    }
                                }
                            } else {
                                true
                            }
                        } else {
                            true
                        };
                        if skipped {
                            dbg!("skipped!");
                            self.skipped_loading_last += 1;
                        }

                        if self.queue.is_empty() {
                            break 'threads;
                        }
                    }
                }
            }

            for (item, prio) in requeue {
                self.queue.push(item, prio);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::world::{
        camera::{controller::CameraController, Camera},
        BufferChunkState, TlcPos, World,
    };
    use cgmath::{Array, Point3, Vector3};

    use crate::world::mem_grid::layer::{DefaultLayerChunkEditor, MemoryGridLayer};

    use super::*;

    const MG_SIZE: usize = 32;
    type TestMemoryGrid = MemoryGridLayer<bool, (), ()>;

    struct BorrowedTestChunkEditor(*mut LayerChunk<bool>);

    unsafe impl<'a> BorrowChunkForLoading<BorrowedTestChunkEditor, ()>
        for DefaultLayerChunkEditor<'a, bool, (), ()>
    {
        fn should_still_load(&self, _: &()) -> bool {
            true
        }

        fn mark_invalid(&mut self) -> Result<(), ()> {
            unsafe { self.chunk.set_invalid() }
        }

        fn take_data_for_loading(&mut self, _: &()) -> BorrowedTestChunkEditor {
            unsafe { self.chunk.set_missing() }
            BorrowedTestChunkEditor(self.chunk)
        }
    }

    unsafe impl Send for BorrowedTestChunkEditor {}
    unsafe impl BorrowedChunk for BorrowedTestChunkEditor {
        type MemoryGrid = TestMemoryGrid;
        unsafe fn done_loading(&mut self, _: &mut Self::MemoryGrid) {
            (&mut *self.0).set_valid()
        }
    }

    struct TestCameraController;
    impl CameraController for TestCameraController {
        fn apply(&mut self, camera: &mut Camera, _: std::time::Duration) {
            camera.position.0 += Vector3::from_value(2.0);
        }
    }

    #[test]
    fn test_load_all_with_buffers() {
        let start_tlc = TlcPos(
            Point3::<i64> { x: 0, y: 0, z: 0 } - Vector3::from_value(MG_SIZE as i64 / 2 - 1),
        );
        let mg = TestMemoryGrid::new(
            (0..MG_SIZE * MG_SIZE * MG_SIZE)
                .map(|_| LayerChunk::new(false))
                .collect(),
            start_tlc,
            MG_SIZE,
            (),
            (),
        );
        let mut world = World::new(mg, Camera::new(8, MG_SIZE), 8, 3);
        let mut loader = ChunkLoader::new(ChunkLoaderParams { n_threads: 1 });
        // Load upper buffer chunks
        world.move_camera(
            &mut TestCameraController,
            Duration::from_secs(0),
            &mut loader,
        );
        debug_assert!(
            *world.metadata().buffer_chunk_states()
                == [
                    BufferChunkState::LoadedUpper,
                    BufferChunkState::LoadedUpper,
                    BufferChunkState::LoadedUpper
                ],
            "{:?}",
            world.metadata().buffer_chunk_states(),
        );

        fn load_f(editor: &mut BorrowedTestChunkEditor, _: ChunkLoadQueueItem<()>, _: ()) {
            let val = unsafe { (&mut *editor.0).get_mut_for_loading() };
            assert!(!*val);
            *val = true;
        }

        let min_chunk = -(world.mem_grid.size() as i64) / 2 + 1;
        let max_chunk = (world.mem_grid.size() as i64) / 2;

        for x in min_chunk..=max_chunk {
            for y in min_chunk..=max_chunk {
                for z in min_chunk..=max_chunk {
                    let pos = TlcPos(Point3 { x, y, z });
                    loader.enqueue(
                        ChunkLoadQueueItem { data: (), pos },
                        world.mem_grid.chunk_loading_priority(pos),
                    );
                }
            }
        }
        assert!(loader.skipped_loading_last() == 0);

        loader.sync(&mut world, &load_f, ());
        assert!(loader.skipped_loading_last() == 0);

        while loader.active_loading_threads() > 0 {
            loader.sync(&mut world, &load_f, ());
            assert!(loader.skipped_loading_last() == 0);
        }

        for x in min_chunk..=max_chunk {
            for y in min_chunk..=max_chunk {
                for z in min_chunk..=max_chunk {
                    let err_msg = format!("{}, {}, {}", x, y, z);
                    assert!(
                        world
                            .edit_chunk(TlcPos(Point3 { x, y, z }))
                            .unwrap()
                            .chunk
                            .get()
                            .expect(&err_msg),
                        "{}",
                        err_msg,
                    );
                }
            }
        }
    }

    #[test]
    fn test_load_all_without_buffers() {
        let start_tlc = TlcPos(
            Point3::<i64> { x: 0, y: 0, z: 0 } - Vector3::from_value(MG_SIZE as i64 / 2 - 1),
        );
        let mg = TestMemoryGrid::new(
            (0..MG_SIZE * MG_SIZE * MG_SIZE)
                .map(|_| LayerChunk::new(false))
                .collect(),
            start_tlc,
            MG_SIZE,
            (),
            (),
        );
        let v = 2; // this doesn't matter
        let mut world = World::new(mg, Camera::new(v, MG_SIZE), v, v as u32);
        let mut loader = ChunkLoader::new(ChunkLoaderParams { n_threads: 1 });

        fn load_f(editor: &mut BorrowedTestChunkEditor, _: ChunkLoadQueueItem<()>, _: ()) {
            let val = unsafe { (&mut *editor.0).get_mut_for_loading() };
            assert!(!*val);
            *val = true;
        }

        let min_chunk = -(world.mem_grid.size() as i64) / 2 + 1;
        let max_chunk = (world.mem_grid.size() as i64) / 2 - 1;

        for x in min_chunk..=max_chunk {
            for y in min_chunk..=max_chunk {
                for z in min_chunk..=max_chunk {
                    let pos = TlcPos(Point3 { x, y, z });
                    loader.enqueue(
                        ChunkLoadQueueItem { data: (), pos },
                        world.mem_grid.chunk_loading_priority(pos),
                    );
                }
            }
        }
        assert!(loader.skipped_loading_last() == 0);

        loader.sync(&mut world, &load_f, ());
        assert!(loader.skipped_loading_last() == 0);

        while loader.active_loading_threads() > 0 {
            loader.sync(&mut world, &load_f, ());
            assert!(loader.skipped_loading_last() == 0);
        }

        for x in min_chunk..=max_chunk {
            for y in min_chunk..=max_chunk {
                for z in min_chunk..=max_chunk {
                    let err_msg = format!("{}, {}, {}", x, y, z);
                    assert!(
                        world
                            .edit_chunk(TlcPos(Point3 { x, y, z }))
                            .unwrap()
                            .chunk
                            .get()
                            .expect(&err_msg),
                        "{}",
                        err_msg,
                    );
                }
            }
        }
    }
}
