use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocatePreference, MemoryAllocator, MemoryTypeFilter};
use vulkano::memory::MemoryPropertyFlags;

mod dynamic_regions;
mod full_copy;
mod const_local;

pub use dynamic_regions::DualBufferWithDynamicCopyRegions;
pub use full_copy::DualBufferWithFullCopy;
pub use const_local::ConstantDeviceLocalBuffer;


/// Buffer scheme with a staging buffer and a device local buffer. This buffer scheme is not
/// directly usable and must be converted to a more specific one.
pub struct DualBuffer<T: ?Sized> {
    staging: Subbuffer<T>,
    device_local: Subbuffer<T>,
}


impl<T: ?Sized> DualBuffer<T> {
    pub fn cloned_staging_buffer(&self) -> Subbuffer<T> {
        self.staging.clone()
    }

    pub fn without_staging_buffer<L, A: CommandBufferAllocator>(
        self,
        one_time_transfer_builder: &mut AutoCommandBufferBuilder<L, A>,
    ) -> ConstantDeviceLocalBuffer<T> {
        let (staging, device_local) = (self.staging, self.device_local);
        one_time_transfer_builder
            .copy_buffer(CopyBufferInfo::buffers(staging, device_local.clone()))
            .unwrap();

        ConstantDeviceLocalBuffer::new(device_local)
    }

    pub fn with_full_copy(self) -> DualBufferWithFullCopy<T> {
        DualBufferWithFullCopy::new(
            self.staging,
            self.device_local,
        )
    }
}

impl<T: BufferContents> DualBuffer<T> {
    pub fn from_data(
        data: T,
        allocator: Arc<dyn MemoryAllocator>,
        is_uniform: bool,
    ) -> DualBuffer<T> {

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
                usage: BufferUsage::TRANSFER_DST
                    | (if is_uniform {
                    BufferUsage::UNIFORM_BUFFER
                } else {
                    BufferUsage::STORAGE_BUFFER
                }),
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

        DualBuffer {
            staging,
            device_local,
        }
    }
}


impl<T: BufferContents> DualBuffer<[T]> {
    pub fn from_iter<I: ExactSizeIterator<Item = T>>(
        iter: I,
        allocator: Arc<dyn MemoryAllocator>,
        is_uniform: bool,
    ) -> DualBuffer<[T]> {
        debug_assert!(iter.len() > 0, "DualBuffer::from_iter expects an iterator with >0 length.");

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
            iter,
        )
            .unwrap();

        let device_local = Buffer::new_slice(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST
                    | (if is_uniform {
                    BufferUsage::UNIFORM_BUFFER
                } else {
                    BufferUsage::STORAGE_BUFFER
                }),
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
            staging.len(),
        )
            .unwrap();

        DualBuffer {
            staging,
            device_local,
        }
    }

    pub fn with_copy_regions(self) -> DualBufferWithDynamicCopyRegions<T> {
        DualBufferWithDynamicCopyRegions::new(
            self.staging,
            self.device_local,
            vec![],
        )
    }
}