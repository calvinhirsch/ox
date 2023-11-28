use derive_new::new;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::descriptor_set::WriteDescriptorSet;
use crate::renderer::buffers::{BufferScheme};


/// Buffer scheme with only a device local buffer (does not need to be updated continuously)
#[derive(new)]
pub struct ConstantDeviceLocalBuffer<T: BufferContents> {
    device_local: Subbuffer<T>,
}


impl<T: BufferContents> BufferScheme for ConstantDeviceLocalBuffer<T> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>, binding: u32) {
        descriptor_writes.push(WriteDescriptorSet::buffer(
            binding,
            self.device_local.clone(),
        ))
    }
}
