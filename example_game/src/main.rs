use cgmath::Point3;
use ox::loader::{ChunkLoader, ChunkLoaderParams};
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
use ox::world::mem_grid::utils::VoxelPosInLod;
use ox::world::mem_grid::voxel::grid::{
    global_voxel_pos_from_pos_in_tlc, voxel_pos_in_tlc_from_global_pos,
};
use ox::world::mem_grid::MemoryGrid;
use ox::world::VoxelPos;
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
use crate::world::{load_chunk, WorldChunkLoadQueueItemData, WorldMemoryGrid};
use world::{TakenWorldChunkEditor, CHUNK_SIZE};

pub const CAMERA_SPEED: f32 = 10.;
pub const CAMERA_SENS: f32 = 0.001;

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

    // The top level chunk (TLC) that defines the bottom corner of our loaded area
    let start_tlc = TlcPos(Point3::<i64> { x: 0, y: 0, z: 0 });

    let (voxel_mem_grid, renderer_voxel_data_component) = VoxelMemoryGrid::new(
        [
            VoxelLODCreateParams {
                voxel_resolution: 1,
                lvl: 0,
                sublvl: 0,
                render_area_size: 3,
                bitmask_binding: 8,
                voxel_ids_binding: Some(4),
            },
            VoxelLODCreateParams {
                voxel_resolution: 2,
                lvl: 0,
                sublvl: 1,
                render_area_size: 5,
                bitmask_binding: 9,
                voxel_ids_binding: Some(5),
            },
            VoxelLODCreateParams {
                voxel_resolution: 4,
                lvl: 0,
                sublvl: 2,
                render_area_size: 9,
                bitmask_binding: 10,
                voxel_ids_binding: Some(6),
            },
            VoxelLODCreateParams {
                voxel_resolution: 8,
                lvl: 1,
                sublvl: 0,
                render_area_size: 23,
                bitmask_binding: 11,
                voxel_ids_binding: Some(7),
            },
            VoxelLODCreateParams {
                voxel_resolution: 64,
                lvl: 2,
                sublvl: 0,
                render_area_size: 23,
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
    let mem_grid = WorldMemoryGrid::new(voxel_mem_grid, start_tlc, 5);
    let mem_grid_size = mem_grid.size();
    let mut world = World::new(mem_grid, Camera::new(tlc_size, mem_grid_size), tlc_size, 16);
    let mut loader: ChunkLoader<
        WorldChunkLoadQueueItemData<N_LODS>,
        TakenWorldChunkEditor<N_LODS>,
    > = ChunkLoader::new(ChunkLoaderParams { n_threads: 48 });

    world.queue_load_all(&mut loader); // load all chunks in render distance

    let voxel_md = world.mem_grid.voxel.metadata().clone();

    // Event loop

    let mut last_render_time = Instant::now();
    let start_time = Instant::now();
    // variables to track input since last frame
    let mut window_resized = false;
    let mut camera_controller = WinitCameraController::new(CAMERA_SPEED, CAMERA_SENS);
    let mut left_clicked = false;
    let mut right_clicked = false;

    event_loop.run(move |event, _, control_flow| {
        match event {
            // Take mouse movement and store in camera controller
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => camera_controller.process_mouse(delta.0, delta.1),
            // Handle left/right click
            Event::DeviceEvent {
                event: DeviceEvent::Button { button, state },
                ..
            } => match state {
                ElementState::Pressed => match button {
                    1 => {
                        left_clicked = true;
                    }
                    3 => {
                        right_clicked = true;
                    }
                    _ => {}
                },
                _ => {}
            },
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(_) => {
                    window_resized = true;
                }
                // Handle keyboard input with camera controller
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
                // Start of frame
                // println!("\n========== Frame ==========");

                // Lock cursor in window
                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                window.set_cursor_visible(false);

                // Handle window resizing
                if window_resized {
                    let dims = window.inner_size();
                    renderer.window_resized(dims);
                    world.set_camera_res(dims.width, dims.height);

                    window_resized = false;
                }

                let frame_start = Instant::now();
                let dt = frame_start - last_render_time;
                // dbg!(dt);
                last_render_time = frame_start;

                // Move camera based on the inputs since last frame as stored in `camera_controller`.
                // This may queue new chunks to load in `loader`.
                world.move_camera(&mut camera_controller, dt, &mut loader);

                // Synchronize chunk loader with `world` and start loading queued chunks when possible.
                loader.sync(&mut world, &load_chunk, voxel_md.clone());

                let camera_pos = world.camera().clone();

                // Check if we clicked last frame--if so, delete block or add new block
                if left_clicked || right_clicked {
                    match cast_ray(
                        &mut world,
                        camera_pos.pos().to_owned(),
                        camera_pos.viewport_center() - camera_pos.pos().0,
                        CHUNK_SIZE,
                        voxel_md.largest_lod().lvl(),
                    ) {
                        Ok(CastRayResult::Hit(RayVoxelIntersect {
                            pos,
                            index,
                            tlc,
                            face,
                        })) => {
                            if left_clicked {
                                let _ = world.edit_chunk(tlc).unwrap().voxel.set_voxel(
                                    pos,
                                    index,
                                    Block::Air,
                                    &voxel_md,
                                );
                            }
                            if right_clicked {
                                let global_pos = global_voxel_pos_from_pos_in_tlc(
                                    tlc,
                                    pos,
                                    world.mem_grid.voxel.metadata().chunk_size(),
                                    voxel_md.largest_lod().lvl(),
                                )
                                .0 + face.delta().0.map(|a| a as i64);
                                let (new_tlc, new_pos) = voxel_pos_in_tlc_from_global_pos(
                                    VoxelPos(global_pos),
                                    CHUNK_SIZE,
                                    voxel_md.largest_lod().lvl(),
                                );

                                // make sure this TLC has LOD 0
                                let v = &mut world.edit_chunk(new_tlc).unwrap().voxel;
                                if v.lods()[0].is_some() {
                                    let _ = v.set_voxel(
                                        new_pos,
                                        VoxelPosInLod {
                                            pos: new_pos.0,
                                            lvl: 0,
                                            sublvl: 0,
                                        }
                                        .index(CHUNK_SIZE, voxel_md.largest_lod().lvl()),
                                        Block::Metal,
                                        &voxel_md,
                                    );
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Apply updates to staging buffers through the renderer
                {
                    let render_editor = renderer.start_updating_staging_buffers();
                    render_editor
                        .component_set
                        .voxel_data
                        .update_staging_buffers_and_prep_copy(world.mem_grid.voxel.get_updates());
                    render_editor
                        .component_set
                        .camera
                        .update_staging_buffer(world.camera());
                    render_editor
                        .component_set
                        .ubo
                        .buffer_scheme
                        .write_staging()
                        .time = (frame_start.duration_since(start_time).as_micros() / 100) as u32;
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
                // loader.print_status();

                left_clicked = false;
                right_clicked = false;
            }
            _ => (),
        }
    });
}
