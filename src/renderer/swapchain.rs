use std::sync::Arc;
use vulkano::command_buffer::allocator::CommandBufferAllocator;
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::device::{Device, Queue};
use vulkano::device::physical::PhysicalDevice;
use vulkano::image::{Image, ImageUsage};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo};
use winit::dpi::PhysicalSize;
use crate::renderer::buffers::BufferScheme;
use crate::renderer::component::{DataComponent, DataComponentSet};
use crate::renderer::pipeline::ComputeRenderPipeline;


pub struct SwapchainPipelineParams<DSA: DescriptorSetAllocator, CBA: CommandBufferAllocator> {
    pub subgroup_width: usize,
    pub subgroup_height: usize,
    pub shader: Arc<ShaderModule>,
    pub descriptor_set_allocator: DSA,
    pub command_buffer_allocator: CBA,
}

pub struct SwapchainPipeline<DSA: DescriptorSetAllocator, CBA: CommandBufferAllocator> {
    params: SwapchainPipelineParams<DSA, CBA>,
    images: Vec<Arc<Image>>,
    swapchain: Arc<Swapchain>,
    pipeline: ComputeRenderPipeline,
}

impl<DSA: DescriptorSetAllocator, CBA: CommandBufferAllocator> SwapchainPipeline<DSA, CBA> {
    fn new_pipeline<D: DataComponentSet, DSA: DescriptorSetAllocator, CBA: CommandBufferAllocator>(
        images: Vec<Arc<Image>>,
        device: Arc<Device>,
        queue: Arc<Queue>,
        dimensions: PhysicalSize<u32>,
        components: Vec<&DataComponent<dyn BufferScheme>>,
        params: SwapchainPipelineParams<DSA, CBA>,
    ) -> ComputeRenderPipeline {
        ComputeRenderPipeline::new(
            params.subgroup_width,
            params.subgroup_height,
            device,
            Arc::clone(&params.shader),
            queue,
            images.as_slice(),
            params.descriptor_set_allocator,
            params.command_buffer_allocator,
            &dimensions,
            components,
        )
    }

    pub fn new<D: DataComponentSet, DSA: DescriptorSetAllocator, CBA: CommandBufferAllocator>(
        device: Arc<Device>,
        queue: Arc<Queue>,
        dimensions: PhysicalSize<u32>,
        components: Vec<&DataComponent<dyn BufferScheme>>,
        physical_device: Arc<PhysicalDevice>,
        surface: Arc<Surface>,
        params: SwapchainPipelineParams<DSA, CBA>,
    ) -> Self {
        let (mut swapchain, images) = {
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
            ).unwrap()
        };

        let pipeline = ComputeRenderPipeline::new(
            params.subgroup_width,
            params.subgroup_height,
            device,
            Arc::clone(&params.shader),
            queue,
            images.as_slice(),
            &params.descriptor_set_allocator,
            &params.command_buffer_allocator,
            &dimensions,
            components,
        );

        SwapchainPipeline {
            params,
            images,
            swapchain,
            pipeline,
        }
    }

    pub fn recreate<D: DataComponentSet>(
        &mut self,
        dimensions: &PhysicalSize<u32>,
        components: Vec<&DataComponent<dyn BufferScheme>>,
        window_resized: bool,
    ) {
        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: dimensions.into(),
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(_e) => panic!("failed to recreate swapchain: {_e:?}"),
        };
        self.swapchain = new_swapchain;
        self.images = new_images;

        if window_resized {
            self.pipeline.recreate(
                &new_images,
                &self.params.descriptor_set_allocator,
                &self.params.command_buffer_allocator,
                &dimensions,
                components,
            );
        }
    }
}