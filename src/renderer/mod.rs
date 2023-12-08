use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use winit::window::{CursorGrabMode, Window};

pub mod buffers;
pub mod component;
pub mod context;
mod pipeline;
pub mod swapchain;
mod transfer;
pub mod utils;

use crate::renderer::component::DataComponentSet;
use crate::renderer::swapchain::SwapchainPipelineParams;
use context::Context;
use swapchain::SwapchainPipeline;
use crate::renderer::transfer::TransferManager;

pub struct Renderer<
    D: DataComponentSet,
    DSA: DescriptorSetAllocator + 'static,
    CBA: CommandBufferAllocator + 'static,
    DCBA: CommandBufferAllocator + 'static
> {
    component_set: D,
    context: Context,
    swapchain_pipeline: SwapchainPipeline<DSA, CBA>,
    transfer_manager: TransferManager<DCBA>,
}

pub struct RendererComponentEditor<'a, D> {
    pub component_set: &'a mut D,
}

impl<D: DataComponentSet, DSA: DescriptorSetAllocator, CBA: CommandBufferAllocator + 'static, DCBA: CommandBufferAllocator + 'static>
    Renderer<D, DSA, CBA, DCBA>
{
    pub fn new(
        context: Context,
        swapchain_pipeline_params: SwapchainPipelineParams<DSA, CBA>,
        window: &Window,
        mut component_set: D,
        dynamic_command_buffer_allocator: DCBA,
    ) -> Self {
        let swapchain_pipeline = SwapchainPipeline::new(
            Arc::clone(&context.device),
            Arc::clone(&context.compute_queue),
            Arc::clone(&context.graphics_queue),
            window.inner_size(),
            &component_set,
            Arc::clone(&context.physical_device),
            Arc::clone(&context.surface),
            swapchain_pipeline_params,
        );

        let transfer_manager = TransferManager::new(
            &context,
            &mut component_set,
            dynamic_command_buffer_allocator,
        );

        Renderer {
            component_set,
            context,
            swapchain_pipeline,
            transfer_manager,
        }
    }

    pub fn window_resized(
        &mut self,
        window: &mut Window,
    ) {
        let new_dimensions = window.inner_size();
        window
            .set_cursor_grab(CursorGrabMode::Confined)
            .unwrap_or_default();

        self.swapchain_pipeline.resize(
            &new_dimensions,
            &self.component_set,
        );
    }

    pub fn recreate_swapchain(&mut self) {
        self.swapchain_pipeline.recreate();
    }

    pub fn start_updating_staging_buffers(&mut self) -> RendererComponentEditor<D> {
        self.transfer_manager.wait_for_staging_buffers(Some(Duration::from_secs(3)));
        RendererComponentEditor { component_set: &mut self.component_set }
    }

    pub fn draw_frame(&mut self) {
        self.swapchain_pipeline.wait_for_compute_done(Some(Duration::from_secs(3)));

        let transfer_fence = self.transfer_manager.start_transfer(
            Arc::clone(&self.context.device),
            Arc::clone(&self.context.transfer_queue),
            &mut self.component_set,
        );

        self.swapchain_pipeline.present(Arc::clone(&self.context.device), transfer_fence);
    }
}
