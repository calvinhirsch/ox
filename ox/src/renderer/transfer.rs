use crate::renderer::component::DataComponentSet;
use crate::renderer::context::Context;
use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::allocator::{
    CommandBufferAllocator, StandardCommandBufferAllocator,
    StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
    SecondaryCommandBufferAbstract,
};
use vulkano::device::{Device, Queue};
use vulkano::sync;
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::GpuFuture;

pub struct TransferManager<CBA: CommandBufferAllocator> {
    always_transfer_command_buffer: Arc<dyn SecondaryCommandBufferAbstract>,
    dynamic_command_buffer_allocator: CBA,
    transfer_fence: Option<Arc<FenceSignalFuture<Box<dyn GpuFuture>>>>,
}

impl<CBA: CommandBufferAllocator + 'static> TransferManager<CBA> {
    pub fn new(
        context: &Context,
        component_set: &mut impl DataComponentSet,
        dynamic_command_buffer_allocator: CBA,
    ) -> TransferManager<CBA> {
        let always_transfer_command_buffer = {
            let command_buffer_allocator = StandardCommandBufferAllocator::new(
                context.device.clone(),
                StandardCommandBufferAllocatorCreateInfo {
                    secondary_buffer_count: 2,
                    ..Default::default()
                },
            );

            let mut builder = AutoCommandBufferBuilder::secondary(
                &command_buffer_allocator,
                context.transfer_queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
                CommandBufferInheritanceInfo::default(),
            )
            .unwrap();

            component_set.record_repeated_buffer_transfer(&mut builder);

            builder.build().unwrap()
        };

        TransferManager {
            always_transfer_command_buffer,
            transfer_fence: None,
            dynamic_command_buffer_allocator,
        }
    }

    pub fn wait_for_staging_buffers(&self, timeout: Option<Duration>) {
        if let Some(tf) = &self.transfer_fence {
            tf.wait(timeout).unwrap();
        }
    }

    pub fn start_transfer(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        component_set: &mut impl DataComponentSet,
    ) -> &Arc<FenceSignalFuture<Box<dyn GpuFuture>>> {
        let previous_transfer_future = match self.transfer_fence.clone() {
            None => {
                let mut now = sync::now(device);
                now.cleanup_finished();
                now.boxed()
            }
            Some(future) => future.boxed(),
        };

        let transfer_command_buffer = {
            let mut builder = AutoCommandBufferBuilder::primary(
                &self.dynamic_command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .execute_commands(Arc::clone(&self.always_transfer_command_buffer))
                .unwrap();

            component_set.record_buffer_transfer_jit(&mut builder);

            builder.build().unwrap()
        };

        let transfer_future = (Box::new(
            previous_transfer_future
                .then_execute(queue, transfer_command_buffer)
                .unwrap(),
        ) as Box<dyn GpuFuture>)
            .then_signal_fence_and_flush();

        self.transfer_fence = match transfer_future {
            Ok(value) => Some(Arc::new(value)),
            Err(e) => {
                println!("failed to flush future: {e:?}");
                None
            }
        };

        self.transfer_fence.as_ref().unwrap()
    }
}
