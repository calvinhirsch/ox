use std::sync::mpsc::{Receiver, sync_channel};
use std::{thread};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use cgmath::{Array, MetricSpace, Vector3};
use priority_queue::PriorityQueue;
use thread_priority::{ThreadBuilderExt, ThreadPriority};
use crate::world::mem_grid::utils::{index_for_pos};
use crate::world::mem_grid::{ChunkCapsule, ChunkEditor, MemoryGridEditor};
use crate::world::TLCPos;



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

pub trait ChunkLoadQueueItemData: Clone + Send {

}


pub trait LoadChunk<'a, QI: ChunkLoadQueueItemData, CE: ChunkEditor<'a>, MD> {
    fn load_new(&mut self, chunk: ChunkLoadQueueItem<QI>, metadata: MD);
}


pub struct ChunkLoader<'a, QI: ChunkLoadQueueItemData+ 'static, CE: ChunkEditor<'a, Capsule = C>, C: ChunkCapsule<'a, CE> + LoadChunk<'a, QI, CE, MD> + Send + 'static, MD: Clone + Send + 'static> {
    chunk_editor_type: PhantomData<CE>,
    mem_grid_metadata_type: PhantomData<MD>,
    lt: PhantomData<&'a CE>,

    active_threads: Vec<Option<Receiver<(TLCPos<i64>, C)>>>,
    highest_thread_idx: usize,
    queue: PriorityQueue<ChunkLoadQueueItem<QI>, u8>,
}

pub struct ChunkLoaderParams {
    pub thread_capacity: usize,
}

impl<'a, QI: ChunkLoadQueueItemData + 'static, CE: ChunkEditor<'a, Capsule = C>, C: ChunkCapsule<'a, CE> + LoadChunk<'a, QI, CE, MD> + Clone + Send + 'static, MD: Clone + Send + 'static> ChunkLoader<'a, QI, CE, C, MD> {
    pub fn new(params: ChunkLoaderParams) -> Self {
        ChunkLoader {
            chunk_editor_type: PhantomData,
            mem_grid_metadata_type: PhantomData,
            lt: PhantomData,
            active_threads: (0..params.thread_capacity).map(|_| None).collect(),
            highest_thread_idx: 0,
            queue: PriorityQueue::new(),
        }
    }

    fn grid_index(
        start_tlc: TLCPos<i64>,
        grid_size: usize,
        pos: TLCPos<i64>,
    ) -> Option<usize> {
        let pt = pos.0 - start_tlc.0;
        let max = Vector3::from_value(grid_size as i64);
        if pt.x >= max.x || pt.y >= max.y || pt.z >= max.z || pt.x < 0 || pt.y < 0 || pt.z < 0  {
            None
        }
        else {
            Some(
                index_for_pos(
                    (pos.0 - start_tlc.0).cast::<usize>()?,
                    grid_size
                )
            )
        }
    }

    pub fn sync(
        &mut self,
        start_tlc: TLCPos<i64>,
        editor: &mut MemoryGridEditor<CE, MD>,
        queue: Vec<ChunkLoadQueueItem<QI>>,
    ) {
        // Receive chunks that have finished loading and put them in "chunks"
        let mut prev_active = 0;
        for (i, thread_slot) in self.active_threads.iter_mut().enumerate() {
            if if let Some(receiver) = thread_slot {
                prev_active = i;
                if let Ok((pos, chunk_data)) = receiver.try_recv() {
                    if let Some(index) = ChunkLoader::<QI, CE, C, MD>::grid_index(start_tlc, editor.size, pos) {
                        chunk_data.move_into_chunk(editor.chunks[index].as_mut().unwrap());
                    }
                    true
                }
                else { false }
            }
            else { false } {
                // Finished with this thread, set its entry to None and update highest_thread_idx
                // if this was previously the highest.
                *thread_slot = None;
                if self.highest_thread_idx <= i {
                    self.highest_thread_idx = prev_active;
                }
            }

            if i >= self.highest_thread_idx { break; }
        }

        // Add new items to queue
        // TODO: if an item's memory overlaps with one already in the queue then figure out what to do
        for item in queue {
            let priority = (99.99 - (Vector3::from_value(editor.size/2).cast::<f32>().unwrap()
                .distance((item.pos.0 - start_tlc.0).cast::<f32>().unwrap())) * editor.size as f32 / 50.) as u8;
            self.queue.push(
                item,
                priority,
            );
        }

        // Enqueue new chunks for loading until queue is empty or there are no thread slots left
        if !self.queue.is_empty() {
            for (i, thread_slot) in self.active_threads.iter_mut().enumerate() {
                if thread_slot.is_none() {
                    let (item, priority) = self.queue.pop().unwrap();
                    let (sender, receiver) = sync_channel(0);

                    let grid_index = ChunkLoader::<QI, CE, C, MD>::grid_index(start_tlc, editor.size, item.pos)
                        .expect("A chunk was queued for loading that is not in bounds of current grid.");

                    // Create a placeholder for this chunk and swap it into the grid so we can edit
                    // the real one.
                    let mut chunk = editor.chunks[grid_index].as_mut().unwrap().replace_with_placeholder();

                    let meta = editor.metadata().clone();
                    let pos = item.pos;
                    thread::Builder::new()
                        .spawn_with_priority(
                            ThreadPriority::Crossplatform(priority.try_into().unwrap()),
                            move |result| {
                                result.expect("Failed to set thread priority.");
                                chunk.load_new(item, meta);
                                sender.send((pos, chunk)).unwrap();
                            }
                        ).unwrap();

                    *thread_slot = Some(receiver);
                    if i > self.highest_thread_idx { self.highest_thread_idx = i; }

                    if self.queue.is_empty() { break; }
                }
            }
        }
    }
}