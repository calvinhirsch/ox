use crate::renderer::buffers::{BufferScheme, dual::{ConstantDeviceLocalBuffer, DualBuffer}, DynamicBufferScheme};
use vulkano::buffer::BufferContents;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::WriteDescriptorSet;

pub mod camera;
pub mod ubo;
pub mod voxels;
pub mod materials;

pub trait DataComponentSet {
    fn dynamic_components_mut(&mut self) -> Vec<&mut DataComponent<dyn DynamicBufferScheme>>;
    fn constant_components_mut(&self) -> Vec<&mut DataComponent<ConstantDeviceLocalBuffer<dyn BufferContents>>>;
    fn all_components_mut(&self) -> Vec<&mut DataComponent<dyn BufferScheme>> {
        let mut comps = self.list_dynamic_components();
        comps.extend(self.list_constant_components());

        comps
    }
}

pub struct DataComponent<B: BufferScheme> {
    pub buffer_scheme: B,
    pub binding: u32,
}

impl<B: BufferScheme> DataComponent<B> {
    pub fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        self.buffer_scheme.bind(descriptor_writes, self.binding);
    }
}
impl<D: BufferContents> DataComponent<DualBuffer<D>> {
    fn record_transfer<L, A: CommandBufferAllocator>(
        &self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        self.buffer_scheme.record_transfer(builder);
    }

    fn drop_staging(self) -> DataComponent<ConstantDeviceLocalBuffer<D>> {
        DataComponent {
            buffer_scheme: self.buffer_scheme.drop_staging(),
            binding: self.binding,
        }
    }
}
