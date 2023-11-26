use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, BufferWriteGuard, Subbuffer};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo};
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocatePreference, MemoryAllocator, MemoryTypeFilter};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::sync::HostAccessError;


pub trait BufferScheme {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>, binding: u32);
}

/// Buffer scheme with a staging buffer and a device local buffer
pub struct DualBuffer<T: BufferContents> {
    staging: Subbuffer<T>,
    device_local: Subbuffer<T>
}

/// Buffer scheme with only a device local buffer (does not need to be updated continuously)
pub struct ConstantBuffer<T: BufferContents> {
    device_local: Subbuffer<T>,
}


impl<T: BufferContents> ConstantBuffer<T> {
    pub fn from_dual_buffer<L, A : CommandBufferAllocator>(
        one_time_transfer_builder: &mut AutoCommandBufferBuilder<L, A>,
        db: DualBuffer<T>
    ) -> ConstantBuffer<T> {
        let (staging, device_local) = (db.staging, db.device_local);
        one_time_transfer_builder.copy_buffer(
            CopyBufferInfo::buffers(
                staging,
                device_local.clone()
            )
        ).unwrap();

        ConstantBuffer { device_local }
    }
}
impl<T: BufferContents> BufferScheme for ConstantBuffer<T> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>, binding: u32) {
        descriptor_writes.push(
            WriteDescriptorSet::buffer(binding, self.device_local.clone())
        )
    }
}

impl<T: BufferContents> BufferScheme for DualBuffer<T> {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>, binding: u32) {
        descriptor_writes.push(
            WriteDescriptorSet::buffer(binding, self.device_local.clone())
        )
    }
}
impl<T: BufferContents> DualBuffer<T> {
    pub fn from_iter<I: Iterator<Item = T>>(iter: I, allocator: Arc<dyn MemoryAllocator>, is_uniform: bool) -> DualBuffer<T> {
        let (staging, device_local) = buffer_from_iter(iter, allocator, is_uniform);
        DualBuffer { staging, device_local }
    }

    pub fn from_data(data: T, allocator: Arc<dyn MemoryAllocator>, is_uniform: bool) -> DualBuffer<T> {
        let (staging, device_local) = buffer_from_data(data, allocator, is_uniform);
        DualBuffer { staging, device_local }
    }

    pub fn cloned_staging_buffer(&self) -> Subbuffer<[T]> { self.staging.clone() }

    pub fn write_staging(&self) -> Result<BufferWriteGuard<'_, T>, HostAccessError> { self.staging.write() }

    pub fn record_transfer<L, A : CommandBufferAllocator>(&self, builder: &mut AutoCommandBufferBuilder<L, A>) {
        builder.copy_buffer(
            CopyBufferInfo::buffers(
                self.staging.clone(),
                self.device_local.clone(),
            )
        ).unwrap();
    }

    pub fn drop_staging(self) -> ConstantBuffer<T> {
        ConstantBuffer { device_local: self.device_local }
    }
}

fn buffer_from_iter<T: BufferContents, I: Iterator<Item = T>>(
    iter: I,
    allocator: Arc<dyn MemoryAllocator>,
    is_uniform: bool
) -> (Subbuffer<T>, Subbuffer<T>) {
    let staging = Buffer::from_iter(
        Arc::clone(&allocator),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter {
                required_flags: MemoryPropertyFlags::HOST_VISIBLE,
                ..Default::default()
            },
            allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
            ..Default::default()
        },
        iter.iter().copied(),
    ).unwrap();

    let device_local = Buffer::new_slice(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | (if is_uniform { BufferUsage::UNIFORM_BUFFER } else { BufferUsage::STORAGE_BUFFER }),
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter {
                required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            },
            allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
            ..Default::default()
        },
        staging.len()
    ).unwrap();

    (staging, device_local)
}

fn buffer_from_data<T: BufferContents>(data: T, allocator: Arc<dyn MemoryAllocator>, is_uniform: bool) -> (Subbuffer<T>, Subbuffer<T>) {
    let staging = Buffer::from_data(
        Arc::clone(&allocator),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter {
                required_flags: MemoryPropertyFlags::HOST_VISIBLE,
                ..Default::default()
            },
            allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
            ..Default::default()
        },
        data,
    ).unwrap();

    let device_local = Buffer::new_sized(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | (if is_uniform { BufferUsage::UNIFORM_BUFFER } else { BufferUsage::STORAGE_BUFFER }),
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter {
                required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            },
            allocate_preference: MemoryAllocatePreference::AlwaysAllocate,
            ..Default::default()
        },
    ).unwrap();

    (staging, device_local)
}