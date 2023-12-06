use crate::renderer::component::{DataComponentSet};
use crate::renderer::pipeline::ComputeRenderPipeline;
use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, Queue};
use vulkano::image::{Image, ImageUsage};
use vulkano::shader::ShaderModule;
use vulkano::{swapchain, sync, Validated, VulkanError};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::{FenceSignalFuture};
use vulkano::sync::GpuFuture;
use winit::dpi::PhysicalSize;

pub struct SwapchainPipelineParams<DSA: DescriptorSetAllocator, CBA: CommandBufferAllocator> {
    pub subgroup_width: u32,
    pub subgroup_height: u32,
    pub image_binding: u32,
    pub shader: Arc<ShaderModule>,
    pub descriptor_set_allocator: DSA,
    pub command_buffer_allocator: CBA,
}

pub type GpuFence = FenceSignalFuture<Box<dyn GpuFuture>>;

pub struct SwapchainPipeline<DSA: DescriptorSetAllocator + 'static, CBA: CommandBufferAllocator + 'static> {
    params: SwapchainPipelineParams<DSA, CBA>,
    images: Vec<Arc<Image>>,
    graphics_queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    pipeline: ComputeRenderPipeline<CBA>,

    recreate: bool,
    compute_fence: Option<Arc<GpuFence>>,
    present_fences: Vec<Option<Arc<GpuFence>>>,
    prev_fence_i: u32,
}

impl<DSA: DescriptorSetAllocator + 'static, CBA: CommandBufferAllocator + 'static> SwapchainPipeline<DSA, CBA> {
    pub fn new(
        device: Arc<Device>,
        compute_queue: Arc<Queue>,
        graphics_queue: Arc<Queue>,
        dimensions: PhysicalSize<u32>,
        component_set: &impl DataComponentSet,
        physical_device: Arc<PhysicalDevice>,
        surface: Arc<Surface>,
        params: SwapchainPipelineParams<DSA, CBA>,
    ) -> Self {
        let (swapchain, images) = {
            let caps = physical_device
                .surface_capabilities(&surface, Default::default())
                .expect("failed to get surface capabilities");

            let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
            let image_format = physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

            Swapchain::new(
                Arc::clone(&device),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format,
                    image_extent: dimensions.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
                    composite_alpha,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let pipeline = ComputeRenderPipeline::new(
            params.subgroup_width,
            params.subgroup_height,
            device,
            Arc::clone(&params.shader),
            compute_queue,
            images.as_slice(),
            params.image_binding,
            &params.descriptor_set_allocator,
            &params.command_buffer_allocator,
            &dimensions,
            component_set,
        );

        let len = images.len();
        SwapchainPipeline {
            params,
            images,
            graphics_queue,
            swapchain,
            pipeline,
            recreate: false,
            compute_fence: None,
            present_fences: vec![None; len],
            prev_fence_i: 0,
        }
    }

    pub fn resize(
        &mut self,
        dimensions: &PhysicalSize<u32>,
        component_set: &impl DataComponentSet,
    ) {
        self.recreate_with_dims(*dimensions);
        self.pipeline.recreate(
            &self.images,
            &self.params.descriptor_set_allocator,
            &self.params.command_buffer_allocator,
            dimensions,
            component_set,
        );
    }

    pub fn recreate(&mut self) {
        self.recreate_with_dims(self.swapchain.image_extent());
    }

    pub fn recreate_with_dims(&mut self, dimensions: impl Into<[u32; 2]>,) {
        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: dimensions.into(),
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(_e) => panic!("failed to recreate swapchain: {_e:?}"),
        };
        self.swapchain = new_swapchain;
        self.images = new_images;
    }

    pub fn wait_for_compute_done(&self, timeout: Option<Duration>) {
        if let Some(fence) = &self.compute_fence {
            fence.wait(timeout).unwrap();
        }
    }

    pub fn present(&mut self, device: Arc<Device>, transfer_fence: &Arc<FenceSignalFuture<Box<dyn GpuFuture>>>) {
        if self.recreate {
            self.recreate();
            self.recreate = false;
        }

        let (image_i, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(Arc::clone(&self.swapchain), Some(Duration::from_secs(3))) {
                Ok(r) => r,
                Err(Validated::Error(VulkanError::OutOfDate)) => {
                    self.recreate = true;
                    return;
                }
                Err(_e) => {
                    panic!("failed to acquire next image: {_e:?}")
                },
            };

        if suboptimal {
            self.recreate = true;
        }

        // wait for the fence related to this image to finish (normally this would be the oldest fence)
        if let Some(image_fence) = &self.present_fences[image_i as usize] {
            image_fence.wait(Some(Duration::from_secs(3))).unwrap();
        }

        let previous_future = match self.present_fences[self.prev_fence_i as usize].clone() {
            // Create a NowFuture
            None => {
                let mut now = sync::now(Arc::clone(&device));
                now.cleanup_finished();

                now.boxed()
            }
            // Use the existing FenceSignalFuture
            Some(fence) => fence.boxed(),
        };

        let curr_future = previous_future
            .join(Arc::clone(transfer_fence))
            .join(acquire_future);

        let compute_future =
            (Box::new(self.pipeline.execute(curr_future, image_i as usize)) as Box<dyn GpuFuture>).then_signal_fence_and_flush();

        self.compute_fence = match compute_future {
            Ok(value) => Some(Arc::new(value)),
            Err(e) => {
                println!("failed to flush future: {e:?}");
                None
            }
        };

        let future = (Box::new(
            Arc::clone(self.compute_fence.as_ref().unwrap())
                .then_swapchain_present(
                    Arc::clone(&self.graphics_queue),
                    SwapchainPresentInfo::swapchain_image_index(Arc::clone(&self.swapchain), image_i),
                )
        ) as Box<dyn GpuFuture>).then_signal_fence_and_flush();

        self.present_fences[image_i as usize] = match future {
            Ok(value) => Some(Arc::new(value)),
            Err(Validated::Error(VulkanError::OutOfDate)) => {
                self.recreate = true;
                None
            }
            Err(e) => {
                println!("failed to flush future: {e:?}");
                None
            }
        };

        self.prev_fence_i = image_i;
    }
}
