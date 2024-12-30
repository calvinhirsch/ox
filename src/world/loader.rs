use crate::world::mem_grid::utils::index_for_pos;
use crate::world::mem_grid::MemoryGridEditor;
use crate::world::{BufferChunkState, TLCPos};
use cgmath::{Array, MetricSpace, Vector3};
use priority_queue::PriorityQueue;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
use std::thread;

pub struct ChunkLoadQueueItem<D> {
    pub pos: TLCPos<i64>,
    pub data: D,
}
impl<D> Hash for ChunkLoadQueueItem<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos.0.hash(state);
    }
}
impl<D> PartialEq<Self> for ChunkLoadQueueItem<D> {
    fn eq(&self, other: &Self) -> bool {
        self.pos.0.eq(&other.pos.0)
    }
}
impl<D> Eq for ChunkLoadQueueItem<D> {}

pub trait LoadChunk<QI, MD> {
    fn load(&mut self, chunk: ChunkLoadQueueItem<QI>, metadata: MD);
}

mod layer_chunk {
    use std::cell::UnsafeCell;
    use std::pin::Pin;

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
            debug_assert!(matches!(self.state(), LayerChunkState::Missing));
            &mut self.data
        }

        pub unsafe fn get_for_loading(&self) -> &T {
            debug_assert!(matches!(self.state(), LayerChunkState::Missing));
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
        pub fn set_missing(self: Pin<&mut Self>) {
            debug_assert!(matches!(self.state(), LayerChunkState::Invalid));
            *unsafe { self.get_unchecked_mut() }.state.get_mut() = LayerChunkState::Missing;
        }

        /// Set the state to "valid". State should be "missing" to do this according to the chunk loading process
        pub unsafe fn set_valid(&mut self) {
            debug_assert!(matches!(self.state(), LayerChunkState::Missing));
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

pub struct ChunkLoader<QI, MD, BC>
where
    BC: BorrowedChunk,
{
    metadata_type: PhantomData<MD>,
    active_threads: Vec<Option<Receiver<BC>>>,
    queue: PriorityQueue<ChunkLoadQueueItem<QI>, u32>,
}

pub struct ChunkLoaderParams {
    pub n_threads: usize,
}

impl<QI, MD, BC: BorrowedChunk> ChunkLoader<QI, MD, BC> {
    pub fn new(params: ChunkLoaderParams) -> Self {
        ChunkLoader {
            metadata_type: PhantomData,
            active_threads: (0..params.n_threads).map(|_| None).collect(),
            queue: PriorityQueue::new(),
        }
    }

    fn vgrid_index(
        start_tlc: TLCPos<i64>,
        grid_size: usize,
        buffer_chunk_states: &[BufferChunkState; 3],
        pos: TLCPos<i64>,
    ) -> Option<usize> {
        let mut pt = pos.0 - start_tlc.0;

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

        Some(index_for_pos(pt.cast::<usize>()?, grid_size))
    }
}

impl<
        QI: Clone + Send + 'static,
        MD: Clone + Send + 'static,
        BC: BorrowedChunk + LoadChunk<QI, MD> + 'static, // why 'static? it's borrowed data but the refernces should be raw pointers
    > ChunkLoader<QI, MD, BC>
{
    /// Queues new chunks for loading and puts loaded chunks back in memory grid using editor.
    pub fn sync<CE: BorrowChunkForLoading<BC>>(
        &mut self,
        start_tlc: TLCPos<i64>,
        editor: &mut MemoryGridEditor<CE, MD>,
        queue: Vec<ChunkLoadQueueItem<QI>>,
        buffer_chunk_states: &[BufferChunkState; 3],
    ) {
        // Receive chunks that have finished loading and put them in "chunks"
        for thread_slot in self.active_threads.iter_mut() {
            if let Some(receiver) = thread_slot {
                match receiver.try_recv() {
                    Ok(mut chunk_data) => {
                        unsafe { chunk_data.mark_valid() };
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

        let editor_chunk_i = |pos| {
            ChunkLoader::<QI, MD, BC>::vgrid_index(start_tlc, grid_size, buffer_chunk_states, pos)
        };

        // Add newly passed chunks to load into the chunk loader queue
        for item in queue {
            // ENHANCEMENT: This is doing some duplicate work to get the chunk and invalidate chunk data as below
            if let Some(chunk_idx) = editor_chunk_i(item.pos) {
                let _ = editor.chunks[chunk_idx].mark_invalid(); // this is just to mark invalid, we don't care if it was fully successful here
                let prio = priority(item.pos);
                self.queue.push(item, prio);
            }
        }

        // Enqueue new chunks for loading until queue is empty or there are no thread slots left
        // ENHANCEMENT: When about to queue a chunk, check that it is still relevant to load.
        let mut requeue = vec![];
        if !self.queue.is_empty() {
            for thread_slot in self.active_threads.iter_mut() {
                if thread_slot.is_none() {
                    let (item, priority) = self.queue.pop().unwrap();
                    let (sender, receiver) = sync_channel(0);

                    if let Some(chunk_idx) = editor_chunk_i(item.pos) {
                        let meta = editor.metadata().clone();
                        let chunk = &mut editor.chunks[chunk_idx];
                        match chunk.mark_invalid() {
                            Ok(()) => {
                                let mut chunk_data = chunk.take_data_for_loading();
                                thread::spawn(move || {
                                    chunk_data.load(item, meta);
                                    sender.send(chunk_data).unwrap_or_else(|e| {
                                        panic!(
                                            "Failed to send loaded chunk back to main thread: {}",
                                            e
                                        )
                                    });
                                });

                                *thread_slot = Some(receiver);
                            }
                            Err(()) => {
                                // Chunk was not ready for loader to take data for loading, so requeue it
                                requeue.push((item, priority));
                            }
                        }
                    }

                    if self.queue.is_empty() {
                        break;
                    }
                }
            }
        }

        for (item, priority) in requeue {
            self.queue.push(item, priority);
        }
    }
}
