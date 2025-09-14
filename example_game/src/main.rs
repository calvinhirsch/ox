use cgmath::Point3;
use ox::ray::{cast_ray, CastRayResult, RayVoxelIntersect};
use ox::renderer::component::camera::RendererCamera;
use ox::renderer::component::materials::MaterialList;
use ox::renderer::component::ubo::{RendererUBO, Ubo};
use ox::renderer::component::voxels::VoxelData;
use ox::renderer::component::DataComponentSet;
use ox::renderer::context::Context;
use ox::renderer::swapchain::SwapchainPipelineParams;
use ox::renderer::utils::standard_one_time_transfer_builder;
use ox::renderer::Renderer;
use ox::voxel_type::VoxelTypeEnum;
use ox::world::camera::controller::winit::WinitCameraController;
use ox::world::loader::{ChunkLoader, ChunkLoaderParams};
use ox::world::mem_grid::MemoryGrid;
use ox::world::{
    camera::Camera,
    mem_grid::voxel::{VoxelLODCreateParams, VoxelMemoryGrid},
    TlcPos, World,
};
use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::allocator::{
    CommandBufferAllocator, StandardCommandBufferAllocator,
    StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryCommandBufferAbstract};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::memory::allocator::MemoryAllocator;
use vulkano::sync::GpuFuture;
use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

mod blocks;
use blocks::Block;
mod world;
use crate::world::{
    load_chunk, WorldChunkEditor, WorldChunkLoadQueueItemData, WorldEditorMetadata, WorldMemoryGrid,
};
use world::{BorrowedWorldChunkEditor, CHUNK_SIZE};

pub const CAMERA_SPEED: f32 = 10.;
pub const CAMERA_SENS: f32 = 1.;

const N_LODS: usize = 5;

mod raytrace_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "../shaders/raytrace.comp",
    }
}

struct RendererComponents {
    voxel_data: VoxelData<N_LODS>,
    material_list: MaterialList,
    camera: RendererCamera,
    ubo: RendererUBO,
}
impl DataComponentSet for RendererComponents {
    fn bind(&self, descriptor_writes: &mut Vec<WriteDescriptorSet>) {
        self.voxel_data.bind(descriptor_writes);
        self.material_list.bind(descriptor_writes);
        self.camera.bind(descriptor_writes);
        self.ubo.bind(descriptor_writes);
    }

    fn record_repeated_buffer_transfer<L, A: CommandBufferAllocator>(
        &self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        self.voxel_data.record_repeated_buffer_transfer(builder);
        self.material_list.record_repeated_buffer_transfer(builder);
        self.camera.record_repeated_buffer_transfer(builder);
        self.ubo.record_repeated_buffer_transfer(builder);
    }

    fn record_buffer_transfer_jit<L, A: CommandBufferAllocator>(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<L, A>,
    ) {
        self.voxel_data.record_buffer_transfer_jit(builder);
        self.material_list.record_buffer_transfer_jit(builder);
        self.camera.record_buffer_transfer_jit(builder);
        self.ubo.record_buffer_transfer_jit(builder);
    }
}

fn main() {
    let event_loop = EventLoop::new();

    let (renderer_context, window) = Context::new(&event_loop);

    let start_tlc = TlcPos(Point3::<i64> { x: 0, y: 0, z: 0 });

    let (voxel_mem_grid, renderer_voxel_data_component) = VoxelMemoryGrid::new(
        [
            VoxelLODCreateParams {
                voxel_resolution: 1,
                lvl: 0,
                sublvl: 0,
                render_area_size: 1,
                bitmask_binding: 8,
                voxel_ids_binding: Some(4),
            },
            VoxelLODCreateParams {
                voxel_resolution: 2,
                lvl: 0,
                sublvl: 1,
                render_area_size: 3,
                bitmask_binding: 9,
                voxel_ids_binding: Some(5),
            },
            VoxelLODCreateParams {
                voxel_resolution: 4,
                lvl: 0,
                sublvl: 2,
                render_area_size: 7,
                bitmask_binding: 10,
                voxel_ids_binding: Some(6),
            },
            VoxelLODCreateParams {
                voxel_resolution: 8,
                lvl: 1,
                sublvl: 0,
                render_area_size: 15,
                bitmask_binding: 11,
                voxel_ids_binding: Some(7),
            },
            VoxelLODCreateParams {
                voxel_resolution: 64,
                lvl: 2,
                sublvl: 0,
                render_area_size: 15,
                bitmask_binding: 12,
                voxel_ids_binding: None,
            },
        ],
        Arc::clone(&renderer_context.memory_allocator) as Arc<dyn MemoryAllocator>,
        CHUNK_SIZE,
        start_tlc,
    );

    let mut one_time_transfer_builder = standard_one_time_transfer_builder(&renderer_context);

    let renderer_components = RendererComponents {
        voxel_data: renderer_voxel_data_component,
        material_list: MaterialList::new(
            &Block::materials(),
            Arc::clone(&renderer_context.memory_allocator) as Arc<dyn MemoryAllocator>,
            1,
            &mut one_time_transfer_builder,
        ),
        camera: RendererCamera::new(
            2,
            Arc::clone(&renderer_context.memory_allocator) as Arc<dyn MemoryAllocator>,
        ),
        ubo: RendererUBO::new(
            Ubo {
                sun_dir: [0.39036, 0.78072, 0.48795],
                start_tlc: [
                    start_tlc.0.x as i32,
                    start_tlc.0.y as i32,
                    start_tlc.0.z as i32,
                ],
                time: 0,
            },
            Arc::clone(&renderer_context.memory_allocator) as Arc<dyn MemoryAllocator>,
            3,
        ),
    };

    one_time_transfer_builder
        .build()
        .unwrap()
        .execute(Arc::clone(&renderer_context.transfer_queue))
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let dev = Arc::clone(&renderer_context.device);
    let mut renderer = Renderer::new(
        renderer_context,
        SwapchainPipelineParams {
            subgroup_width: 8,
            subgroup_height: 8,
            image_binding: 0,
            shader: raytrace_shader::load(Arc::clone(&dev)).expect("Failed to load shader"),
            descriptor_set_allocator: StandardDescriptorSetAllocator::new(
                Arc::clone(&dev),
                Default::default(),
            ),
            command_buffer_allocator: StandardCommandBufferAllocator::new(
                Arc::clone(&dev),
                Default::default(),
            ),
        },
        &window,
        renderer_components,
        StandardCommandBufferAllocator::new(
            dev,
            StandardCommandBufferAllocatorCreateInfo::default(),
        ),
    );

    let tlc_size = voxel_mem_grid.metadata().tlc_size();
    let mem_grid = WorldMemoryGrid::new(voxel_mem_grid, start_tlc, 7);
    let mem_grid_size = mem_grid.size();
    let mut world = World::new(
        mem_grid,
        ChunkLoader::<
            WorldChunkLoadQueueItemData<N_LODS>,
            WorldEditorMetadata,
            BorrowedWorldChunkEditor<N_LODS>,
        >::new(ChunkLoaderParams {
            // n_threads: 1,
            n_threads: usize::from(
                std::thread::available_parallelism()
                    .expect("Failed to determine available parallelism"),
            ) * 2,
        }),
        Camera::new(tlc_size, mem_grid_size),
        tlc_size,
        16,
    );

    world.queue_load_all(); // load all chunks in render distance

    let largest_lod = world.mem_grid.voxel.metadata().largest_lod();

    // Event loop

    let mut last_render_time = Instant::now();
    let start_time = Instant::now();
    let mut window_resized = false;
    let mut camera_controller = WinitCameraController::new(CAMERA_SPEED, CAMERA_SENS);
    let mut clicked = false;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => camera_controller.process_mouse(delta.0, delta.1),
            Event::DeviceEvent {
                event: DeviceEvent::Button { button: _, state },
                ..
            } => match state {
                ElementState::Pressed => {
                    clicked = true;
                }
                _ => {}
            },
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(_) => {
                    window_resized = true;
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(key),
                            state,
                            ..
                        },
                    ..
                } => {
                    camera_controller.process_keyboard(key, state);
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                println!("\n========== Frame ==========");

                if window_resized {
                    let dims = window.inner_size();
                    renderer.window_resized(dims);
                    world.set_camera_res(dims.width, dims.height);

                    window_resized = false;
                }

                let now = Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;

                // World update
                dbg!(world.metadata().buffer_chunk_states());
                world.move_camera(&mut camera_controller, dt); // can only be done before or after editing, not during

                let camera_pos = world.camera().clone();

                world.edit::<WorldChunkEditor<N_LODS>, _, _>(
                    |mut editor| {
                        println!(
                            "Valid chunks: {:.2}%",
                            editor
                                .mem_grid
                                .chunks
                                .iter()
                                .map(|c| c.voxel.all_lods_valid() as u32)
                                .sum::<u32>() as f32
                                * 100.0
                                / editor.mem_grid.chunks.len() as f32,
                        );

                        // let pos = GlobalVoxelPos::new(
                        //     VoxelPos(Point3 {
                        //         x: 50,
                        //         y: 50,
                        //         z: 50,
                        //     }),
                        //     CHUNK_SIZE,
                        //     LARGEST_CHUNK_LVL,
                        // );
                        // let meta = editor.mem_grid.metadata().clone();
                        // let chunk = editor.mem_grid.chunk_mut(pos).unwrap();
                        // if chunk.voxel.no_missing_lods() {
                        //     chunk
                        //         .voxel
                        //         .set_voxel(pos.voxel_index, Block::DIRT, &meta.voxel);
                        // }

                        if clicked {
                            let meta = editor.mem_grid.metadata().voxel.clone();
                            match cast_ray(
                                &mut editor.mem_grid,
                                camera_pos.pos().to_owned(),
                                camera_pos.viewport_center() - camera_pos.pos().0,
                                CHUNK_SIZE,
                                largest_lod.lvl(),
                            ) {
                                Ok(CastRayResult::Hit(RayVoxelIntersect {
                                    pos,
                                    index,
                                    tlc,
                                    ..
                                })) => {
                                    dbg!(pos, index, tlc);
                                    let _ = editor
                                        .mem_grid
                                        .chunk_mut(tlc)
                                        .unwrap()
                                        .voxel
                                        .set_voxel(pos, index, Block::Air, &meta);
                                }
                                _ => {}
                            }
                        }
                    },
                    &load_chunk,
                );

                // Apply updates to staging buffers through the renderer
                {
                    let render_editor = renderer.start_updating_staging_buffers();
                    render_editor
                        .component_set
                        .camera
                        .update_staging_buffer(world.camera());
                    render_editor
                        .component_set
                        .voxel_data
                        .update_staging_buffers_and_prep_copy(world.mem_grid.voxel.get_updates());
                    render_editor
                        .component_set
                        .ubo
                        .buffer_scheme
                        .write_staging()
                        .time = (now.duration_since(start_time).as_micros() / 100) as u32;
                    render_editor
                        .component_set
                        .ubo
                        .buffer_scheme
                        .write_staging()
                        .start_tlc
                        .copy_from_slice(&[
                            world.mem_grid.voxel.start_tlc().0.x as i32,
                            world.mem_grid.voxel.start_tlc().0.y as i32,
                            world.mem_grid.voxel.start_tlc().0.z as i32,
                        ]);
                }

                renderer.draw_frame();
                world.chunk_loader().print_status();

                clicked = false;
            }
            _ => (),
        }
    });
}
