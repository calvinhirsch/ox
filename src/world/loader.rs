use crate::world::mem_grid::utils::index_for_pos;
use crate::world::mem_grid::{ChunkEditor, MemoryGridEditorTrait};
use crate::world::{BufferChunkState, TLCPos};
use cgmath::{Array, MetricSpace, Vector3};
use priority_queue::PriorityQueue;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
use std::{mem, thread};

pub struct ChunkLoadQueueItem<D: ChunkLoadQueueItemData> {
    pub pos: TLCPos<i64>,
    pub data: D,
}
impl<D: ChunkLoadQueueItemData> Hash for ChunkLoadQueueItem<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos.0.hash(state);
    }
}
impl<D: ChunkLoadQueueItemData> PartialEq<Self> for ChunkLoadQueueItem<D> {
    fn eq(&self, other: &Self) -> bool {
        self.pos.0.eq(&other.pos.0)
    }
}
impl<D: ChunkLoadQueueItemData> Eq for ChunkLoadQueueItem<D> {}

pub trait ChunkLoadQueueItemData: Clone + Send {}

pub trait LoadChunk<QI: ChunkLoadQueueItemData, MD> {
    fn load_new(&mut self, chunk: ChunkLoadQueueItem<QI>, metadata: MD);
}

pub struct ChunkLoader<
    QI: ChunkLoadQueueItemData + 'static,
    C: LoadChunk<QI, MD> + Send + 'static,
    MD: Clone + Send + 'static,
> {
    metadata_type: PhantomData<MD>,

    active_threads: Vec<Option<Receiver<(TLCPos<i64>, C)>>>,
    queue: PriorityQueue<ChunkLoadQueueItem<QI>, u32>,
    requeue: Vec<ChunkLoadQueueItem<QI>>,
}

pub struct ChunkLoaderParams {
    pub n_threads: usize,
}

impl<
        QI: ChunkLoadQueueItemData + 'static,
        C: LoadChunk<QI, MD> + Send + 'static,
        MD: Clone + Send + 'static,
    > ChunkLoader<QI, C, MD>
{
    pub fn new(params: ChunkLoaderParams) -> Self {
        ChunkLoader {
            metadata_type: PhantomData,
            active_threads: (0..params.n_threads).map(|_| None).collect(),
            queue: PriorityQueue::new(),
            requeue: vec![],
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
        'a,
        QI: ChunkLoadQueueItemData + 'static,
        C: LoadChunk<QI, MD> + Clone + Send + 'static,
        MD: Clone + Send + 'static,
    > ChunkLoader<QI, C, MD>
{
    /// Queues new chunks for loading and puts loaded chunks back in memory grid using editor.
    pub fn sync<CE: ChunkEditor<'a, Capsule = C>>(
        &mut self,
        start_tlc: TLCPos<i64>,
        editor: &mut impl MemoryGridEditorTrait<CE, MD>,
        queue: Vec<ChunkLoadQueueItem<QI>>,
        buffer_chunk_states: &[BufferChunkState; 3],
    ) {
        // Receive chunks that have finished loading and put them in "chunks"
        for thread_slot in self.active_threads.iter_mut() {
            if if let Some(receiver) = thread_slot {
                match receiver.try_recv() {
                    Ok((pos, chunk_data)) => {
                        // println!("   received {:?}", pos);
                        if let Some(index) = ChunkLoader::<QI, C, MD>::vgrid_index(
                            start_tlc,
                            editor.this().size,
                            buffer_chunk_states,
                            pos,
                        ) {
                            let editor = editor.this_mut().chunks[index].as_mut().unwrap();
                            debug_assert!(editor.ok_to_replace_with_capsule());
                            editor.replace_with_capsule(chunk_data);
                        }
                        true
                    }
                    Err(TryRecvError::Disconnected) => {
                        panic!("Thread disconnected before completing.")
                    }
                    Err(TryRecvError::Empty) => false,
                }
            } else {
                false
            } {
                // Finished with this thread, set its entry to None
                *thread_slot = None;
            }
        }

        let grid_size = editor.this().size;
        let priority = |chunk_pos: TLCPos<i64>| -> u32 {
            u32::MAX
                - (Vector3::from_value(grid_size as f32 / 2.)
                    .cast::<f32>()
                    .unwrap()
                    .distance((chunk_pos.0 - start_tlc.0).cast::<f32>().unwrap())
                    * grid_size as f32) as u32
        };

        let editor_chunk_i = |pos| {
            ChunkLoader::<QI, C, MD>::vgrid_index(start_tlc, grid_size, buffer_chunk_states, pos)
        };

        // Add new items to queue
        for item in queue {
            let pos = item.pos;
            let editor_chunk_idx = editor_chunk_i(item.pos);

            // Call on_queued_for_loading if the chunk is present (i.e. hasn't been taken for loading already)
            // If it is not present, this item will get re-queued until it is.
            if let Some(Some(editor_chunk)) =
                editor_chunk_idx.map(|i| editor.this_mut().chunks[i].as_mut())
            {
                // println!("  on queued for loading {:?} (i: {})", pos, editor_chunk_idx.unwrap());
                editor_chunk.on_queued_for_loading();
            } else {
                // temp (I think)
                panic!(
                    "outside grid, can't load {:?} (i: {:?})",
                    pos, editor_chunk_idx
                );
            }
            self.queue.push(item, priority(pos));
        }

        // Enqueue new chunks for loading until queue is empty or there are no thread slots left
        // ENHANCEMENT: When about to queue a chunk, check that it is still relevant to load.
        if !self.queue.is_empty() {
            for thread_slot in self.active_threads.iter_mut() {
                if thread_slot.is_none() {
                    let (item, _) = self.queue.pop().unwrap();
                    let (sender, receiver) = sync_channel(0);
                    let editor_chunk_idx = editor_chunk_i(item.pos);

                    if let Some(Some(editor_chunk)) =
                        editor_chunk_idx.map(|i| editor.this_mut().chunks[i].as_mut())
                    {
                        if editor_chunk.ok_to_replace_with_placeholder() {
                            // Create a placeholder for this chunk and swap it into the grid so we can edit
                            // the real one.
                            let mut chunk = editor_chunk.replace_with_placeholder();
                            let meta = editor.this().metadata().clone();
                            let pos = item.pos;
                            // println!("  start {:?} (i: {})", pos, editor_chunk_idx.unwrap());
                            thread::spawn(move || {
                                // println!("   in thread {:?}", pos);
                                chunk.load_new(item, meta);
                                // println!("   done loading {:?}", pos);
                                sender.send((pos, chunk)).unwrap_or_else(|e| {
                                    panic!("Failed to send loaded chunk back to main thread: {}", e)
                                });
                            });

                            *thread_slot = Some(receiver);
                        } else {
                            self.requeue.push(item)
                        }
                    }

                    if self.queue.is_empty() {
                        break;
                    }
                }
            }
        }

        // Requeue stuff from 'requeue' and empty it
        for item in mem::take(&mut self.requeue).into_iter() {
            let pos = item.pos;
            self.queue.push(item, priority(pos));
        }
    }
}
