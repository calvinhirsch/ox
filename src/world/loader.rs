use crate::world::WorldEditor;

pub trait ChunkLoader<C> {
    type Queue;

    fn sync_loading(&mut self, world_editor: &mut WorldEditor<C>, new_chunks_to_load: Self::Queue);
}