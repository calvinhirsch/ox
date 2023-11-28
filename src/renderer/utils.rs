use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use crate::renderer::context::Context;


pub fn standard_one_time_transfer_builder(renderer_context: &Context) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
    AutoCommandBufferBuilder::primary(
        &StandardCommandBufferAllocator::new(
            Arc::clone(&renderer_context.device),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: 1,
                secondary_buffer_count: 0,
                ..Default::default()
            }
        ),
        renderer_context.transfer_queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap()
}