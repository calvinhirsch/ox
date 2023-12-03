use crate::renderer::component::{DataComponentSet};
use std::sync::Arc;
use vulkano::command_buffer::allocator::{CommandBufferAllocator};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::{DescriptorSetAllocator};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::Image;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderModule;
use vulkano::sync::GpuFuture;
use winit::dpi::PhysicalSize;

pub struct ComputeRenderPipeline<CBA: CommandBufferAllocator> {
    subgroup_width: u32,
    subgroup_height: u32,
    image_binding: u32,
    device: Arc<Device>,
    shader: Arc<ShaderModule>,
    queue: Arc<Queue>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer<CBA>>>,
}

impl<CBA: CommandBufferAllocator> ComputeRenderPipeline<CBA> {
    pub fn create_command_buffers<DSA: DescriptorSetAllocator>(
        subgroup_width: u32,
        subgroup_height: u32,
        device: Arc<Device>,
        shader: Arc<ShaderModule>,
        queue: Arc<Queue>,
        images: &[Arc<Image>],
        image_binding: u32,
        descriptor_set_allocator: &DSA,
        command_buffer_allocator: &CBA,
        dimensions: &PhysicalSize<u32>,
        component_set: &impl DataComponentSet,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer<CBA>>> {
        let stage = PipelineShaderStageCreateInfo::new(shader.single_entry_point().unwrap());
        let pipeline = ComputePipeline::new(
            Arc::clone(&device),
            None,
            ComputePipelineCreateInfo::stage_layout(
                stage.clone(),
                PipelineLayout::new(
                    Arc::clone(&device),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                        .into_pipeline_layout_create_info(device)
                        .unwrap(),
                )
                .unwrap(),
            ),
        )
        .unwrap();

        let compute_descriptor_sets: Vec<Arc<PersistentDescriptorSet<DSA::Alloc>>> = images
            .iter()
            .map(|image| {
                let mut descriptor_writes = vec![WriteDescriptorSet::image_view(
                    image_binding,
                    ImageView::new(image.clone(), ImageViewCreateInfo::from_image(image)).unwrap(),
                )];

                component_set.bind(&mut descriptor_writes);

                PersistentDescriptorSet::new(
                    descriptor_set_allocator,
                    pipeline.layout().set_layouts().get(0).unwrap().clone(),
                    descriptor_writes,
                    [],
                )
                .unwrap()
            })
            .collect();

        compute_descriptor_sets
            .iter()
            .map(|descriptor_set| {
                let mut builder = AutoCommandBufferBuilder::primary(
                    command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::MultipleSubmit,
                )
                .unwrap();

                builder
                    .bind_pipeline_compute(pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        pipeline.layout().clone(),
                        0,
                        Arc::clone(descriptor_set),
                    )
                    .unwrap()
                    .dispatch([
                        (dimensions.width + subgroup_width - 1) / subgroup_width,
                        (dimensions.height + subgroup_height - 1) / subgroup_height,
                        1,
                    ])
                    .unwrap();

                builder.build().unwrap()
            })
            .collect()
    }

    pub fn new<DSA: DescriptorSetAllocator>(
        subgroup_width: u32,
        subgroup_height: u32,
        device: Arc<Device>,
        shader: Arc<ShaderModule>,
        queue: Arc<Queue>,
        images: &[Arc<Image>],
        image_binding: u32,
        descriptor_set_allocator: &DSA,
        command_buffer_allocator: &CBA,
        dimensions: &PhysicalSize<u32>,
        component_set: &impl DataComponentSet,
    ) -> Self {
        ComputeRenderPipeline {
            subgroup_width,
            subgroup_height,
            image_binding,
            device: Arc::clone(&device),
            shader: Arc::clone(&shader),
            queue: Arc::clone(&queue),
            command_buffers: Self::create_command_buffers(
                subgroup_width,
                subgroup_height,
                device,
                shader,
                queue,
                images,
                image_binding,
                descriptor_set_allocator,
                command_buffer_allocator,
                dimensions,
                component_set,
            ),
        }
    }

    pub fn recreate<DSA: DescriptorSetAllocator>(
        &mut self,
        images: &[Arc<Image>],
        descriptor_set_allocator: &DSA,
        command_buffer_allocator: &CBA,
        dimensions: &PhysicalSize<u32>,
        component_set: &impl DataComponentSet,
    ) {
        self.command_buffers = Self::create_command_buffers(
            self.subgroup_width,
            self.subgroup_height,
            Arc::clone(&self.device),
            Arc::clone(&self.shader),
            Arc::clone(&self.queue),
            images,
            self.image_binding,
            descriptor_set_allocator,
            command_buffer_allocator,
            dimensions,
            component_set,
        );
    }

    pub fn execute<F: GpuFuture>(&self, future: F, index: usize) -> CommandBufferExecFuture<F> {
        future.then_execute(Arc::clone(&self.queue), Arc::clone(&self.command_buffers[index])).unwrap()
    }
}
