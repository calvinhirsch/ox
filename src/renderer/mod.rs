use std::sync::Arc;
use vulkano::command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage, CopyBufferInfo, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::memory::allocator::MemoryAllocator;
use winit::window::CursorGrabMode;


pub mod context;
pub mod buffers;
mod swapchain;
pub mod component;
mod pipeline;

use swapchain::SwapchainPipeline;
use context::Context;
use crate::renderer::component::DataComponentSet;
use crate::renderer::swapchain::SwapchainPipelineParams;
use crate::world::camera::Camera;


pub struct Renderer<D: DataComponentSet, DSA: DescriptorSetAllocator, CBA: CommandBufferAllocator> {
    pub data: D,
    context: Context,
    swapchain_pipeline: SwapchainPipeline<DSA, CBA>,
    always_transfer_command_buffer: Arc<SecondaryAutoCommandBuffer>,
}

impl<D: DataComponentSet, DSA: DescriptorSetAllocator, CBA: CommandBufferAllocator> Renderer<D, DSA, CBA> {
    pub fn new(
        context: Context,
        swapchain_pipeline_params: SwapchainPipelineParams<DSA, CBA>,
        component_set: D,
    ) -> Self {
        let swapchain_pipeline = SwapchainPipeline::new(
            Arc::clone(&context.device),
            Arc::clone(&context.compute_queue),
            context.window.inner_size(),
            component_set.list_all_components(),
            Arc::clone(&context.physical_device),
            Arc::clone(&context.surface),
            swapchain_pipeline_params,
        );

        let always_transfer_command_buffer = {
            let command_buffer_allocator =
                StandardCommandBufferAllocator::new(
                    context.device.clone(),
                    StandardCommandBufferAllocatorCreateInfo {
                        secondary_buffer_count: 2,
                        ..Default::default()
                    }
                );
            
            let mut builder = AutoCommandBufferBuilder::secondary(
                &command_buffer_allocator,
                context.transfer_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
                CommandBufferInheritanceInfo::default(),
            ).unwrap();
            
            for component in component_set.list_dynamic_components() {
                component.buffer_scheme.record_transfer(&mut builder);
            }

            builder.build().unwrap()
        };

        Renderer {
            data,
            context,
            swapchain_pipeline,
            always_transfer_command_buffer,
        }
    }

    pub fn recreate_swapchain(&mut self, camera: &mut Camera, window_resized: bool) {
        let new_dimensions = self.context.window.inner_size();
        self.context.window.set_cursor_grab(CursorGrabMode::Confined).unwrap_or_default();
        camera.resolution = (new_dimensions.width, new_dimensions.height);

        self.swapchain_pipeline.recreate(&new_dimensions, self.data.list_all_components(), window_resized);
    }

    pub fn draw_frame(&mut self) {

    }
}