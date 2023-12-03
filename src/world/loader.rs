use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::mpsc::{Receiver, sync_channel};
use std::{mem, thread};
use std::hash::{Hash, Hasher};
use cgmath::{Array, Vector3};
use crate::world::mem_grid::utils::pos_index;
use crate::world::mem_grid::{MemoryGridMetadata, VirtualMemoryGridStruct};
use crate::world::TLCPos;


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


impl<D> Eq for ChunkLoadQueueItem<D> {

}


pub trait LoadChunk<QI> {
    fn load_new(&mut self, chunk: ChunkLoadQueueItem<QI>) -> Self;
}


pub struct ChunkLoader<QI, C: LoadChunk<QI>> {
    queue_item_data_type: PhantomData<QI>,
    queue: VecDeque<Receiver<(TLCPos<i64>, C)>>
}

impl<QI, C: LoadChunk<QI>> ChunkLoader<QI, C> {
    pub fn new() -> Self {
        ChunkLoader { queue_item_data_type: PhantomData::default(), queue: VecDeque::new() }
    }

    fn grid_index<_MD: MemoryGridMetadata>(
        start_tlc: TLCPos<i64>,
        grid: &mut VirtualMemoryGridStruct<C, _MD>,
        pos: TLCPos<i64>,
    ) -> Option<usize> {
        let pt = pos.0 - start_tlc.0;
        if pt >= Vector3::from_value(grid.size() as i64) {
            None
        }
        else {
            Some(
                pos_index(
                    (pos.0 - start_tlc.0).cast::<usize>()?,
                    grid.size()
                )
            )
        }
    }

    pub fn sync<_MD: MemoryGridMetadata>(
        &mut self,
        start_tlc: TLCPos<i64>,
        grid: &mut VirtualMemoryGridStruct<C, _MD>,
        queue: Vec<ChunkLoadQueueItem<QI>>
    ) {
        // Receive chunks that have finished loading and put them in "chunks"
        for receiver in self.queue {
            receiver.try_recv().and_then(|(pos, chunk_data)| {
                match ChunkLoader::grid_index(start_tlc, grid, pos) {
                    None => {},
                    Some(index) => { grid[index] = chunk_data; }
                }
                Ok(())
            }).unwrap();
        }

        // Enqueue new chunks for loading
        for qi in queue {
            let (sender, receiver) = sync_channel(0);

            let grid_index = ChunkLoader::grid_index(start_tlc, grid, qi.pos)
                .expect("A chunk was queued for loading that is not in bounds of current grid.");

            let chunk = mem::replace(grid[grid_index], vec![]);

            thread::spawn(move || {
                sender.send((qi.pos, chunk.load(qi))).unwrap();
            });

            self.queue.push_back(receiver);
        }
    }
}