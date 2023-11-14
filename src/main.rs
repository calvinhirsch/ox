use std::error::Error;
use std::f32::consts::PI;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::{Instant, Duration, UNIX_EPOCH, SystemTime};

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice};
use vulkano::device::{
    Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags, DeviceExtensions
};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{ImageUsage, Image};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocatePreference, MemoryTypeFilter, StandardMemoryAllocator};

use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::{Validated, VulkanError};
use vulkano::instance::debug::ValidationFeatureEnable;

use winit::event::{DeviceEvent, Event, KeyboardInput, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowBuilder};
use winit::dpi::PhysicalSize;

use cgmath::{Point3, Rad};
use itertools::Itertools;
use num_traits::PrimInt;

mod camera;
use crate::camera::{Camera, CameraController, CameraUBO};


const N_CHUNK_LVLS: usize = 2;
const CHUNK_SIZE: usize = 8;
const CHUNK_BLOCK_SIZES: [usize; N_CHUNK_LVLS+1] = [
    1,
    CHUNK_SIZE,
    CHUNK_SIZE*CHUNK_SIZE,
];
const RENDER_N_TLCS: usize = 4;
const SUBGROUP_WIDTH: u32 = 8;
const SUBGROUP_HEIGHT: u32 = 8;

const N_CHUNKS: [usize; N_CHUNK_LVLS+1] = [
    CHUNK_SIZE.pow(6) * RENDER_N_TLCS.pow(2),
    CHUNK_SIZE.pow(3) * RENDER_N_TLCS.pow(2),
    RENDER_N_TLCS.pow(2),
];
const TLC_SIZE: usize = CHUNK_SIZE.pow(N_CHUNK_LVLS as u32);
const VMI_BUFFER_LEN: usize = N_CHUNKS[0] / 16;  // 16 u8's per uvec4


trait ArithmeticModulo {
    fn amod(&self, m: Self) -> Self;
}
impl<T: PrimInt> ArithmeticModulo for T {
    fn amod(&self, m: T) -> T {
        ((*self % m) + m) % m
    }
}


#[derive(Debug)]
struct OutOfBoundsErr {
    index: usize,
    size: usize,
}
impl Error for OutOfBoundsErr {}
impl Display for OutOfBoundsErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tried to index with {} in array of size {}", self.index, self.size)
    }
}


#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
struct Material {
    color: [f32; 3],
    _pad1: f32,
    specular_color: [f32; 3],
    _pad2: f32,
    emission_color: [f32; 3],
    emission_strength: f32,
    specular_prob_perpendicular: f32,
    specular_prob_parallel: f32,
    _pad3: [f32; 2],
}

const MATERIALS: [Material; 5] = [
    // Air
    Material {
        color: [0.0, 0.0, 0.0],
        _pad1: 0.,
        specular_color: [0.0, 0.0, 0.0],
        _pad2: 0.,
        emission_color: [0.0, 0.0, 0.0],
        emission_strength: 0.0,
        specular_prob_perpendicular: 0.0,
        specular_prob_parallel: 0.0,
        _pad3: [0., 0.],
    },
    // Some matte block
    Material {
        color: [0.6, 0.6, 0.6],
        _pad1: 0.,
        specular_color: [1.0, 1.0, 1.0],
        _pad2: 0.,
        emission_color: [0.0, 0.0, 0.0],
        emission_strength: 0.0,
        specular_prob_perpendicular: 0.0,
        specular_prob_parallel: 0.25,
        _pad3: [0., 0.],
    },
    // Some reflective block
    Material {
        color: [0.6, 0.6, 0.6],
        _pad1: 0.,
        specular_color: [1.0, 1.0, 1.0],
        _pad2: 0.,
        emission_color: [0.0, 0.0, 0.0],
        emission_strength: 0.0,
        specular_prob_perpendicular: 0.85,
        specular_prob_parallel: 0.95,
        _pad3: [0., 0.],
    },
    // Some light emitting block
    Material {
        color: [0.6, 0.6, 0.6],
        _pad1: 0.,
        specular_color: [0.0, 0.0, 0.0],
        _pad2: 0.,
        emission_color: [1.0, 0.0, 0.0],
        emission_strength: 1.0,
        specular_prob_perpendicular: 0.0,
        specular_prob_parallel: 0.0,
        _pad3: [0., 0.],
    },
    // Some light emitting block 2
    Material {
        color: [0.6, 0.6, 0.6],
        _pad1: 0.,
        specular_color: [0.0, 0.0, 0.0],
        _pad2: 0.,
        emission_color: [0.0, 1.0, 0.0],
        emission_strength: 1.0,
        specular_prob_perpendicular: 0.0,
        specular_prob_parallel: 0.0,
        _pad3: [0., 0.],
    },
];

#[derive(BufferContents, Debug, Clone)]
#[repr(C)]
struct Ubo {
    sun_dir: [f32; 3],
    tlc_minxi: i32,
    tlc_minzi: i32,
    time: u32,
}


#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
struct TopLevelChunkIndex {
    // #[format(R32G32B32A32_UINT)]
    indices: [u32; 4],
}

#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
struct MaterialIndex {
    // #[format(R32G32B32A32_UINT)]
    indices: [u32; 4],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct VoxelBitmask {
    // #[format(R32G32B32A32_UINT)]
    mask: [u32; 4],
}

impl Display for VoxelBitmask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#032b}; {:#032b}; {:#032b}; {:#032b}", self.mask[3], self.mask[2], self.mask[1], self.mask[0])
    }
}


mod raytrace_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/raytrace.comp",
    }
}


pub fn select_physical_device(
    instance: &Arc<Instance>,
) -> Arc<PhysicalDevice> {
    instance.enumerate_physical_devices().unwrap().next().expect("No devices.")
}

fn get_compute_pipeline(
    device: Arc<Device>,
    shader: Arc<ShaderModule>,
) -> Arc<ComputePipeline> {
    let stage = PipelineShaderStageCreateInfo::new(shader.single_entry_point().unwrap());
    ComputePipeline::new(
        Arc::clone(&device),
        None,
        ComputePipelineCreateInfo::stage_layout(
             stage.clone(),
            PipelineLayout::new(
                Arc::clone(&device),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device)
                    .unwrap(),
            ).unwrap(),
        )
    ).unwrap()
}

fn get_compute_command_buffers(
    images: &[Arc<Image>],
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    material_buffer: &Subbuffer<[Material]>,
    camera_buffer: &Subbuffer<CameraUBO>,
    ubo_buffer: &Subbuffer<Ubo>,
    tlci_buffer: &Subbuffer<[TopLevelChunkIndex]>,
    vmi_buffer: &Subbuffer<[MaterialIndex]>,
    chunk_bitmask_buffers: &Vec<Subbuffer<[VoxelBitmask]>>,
    queue: &Arc<Queue>,
    compute_pipeline: &Arc<ComputePipeline>,
    dimensions: &PhysicalSize<u32>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    let compute_descriptor_sets: Vec<Arc<PersistentDescriptorSet>> = images.iter()
        .map(|image|
            PersistentDescriptorSet::new(
                descriptor_set_allocator,
                compute_pipeline.layout().set_layouts().get(0).unwrap().clone(),
                vec![
                    WriteDescriptorSet::image_view(0, ImageView::new(image.clone(), ImageViewCreateInfo::from_image(image)).unwrap()),
                    WriteDescriptorSet::buffer(1, material_buffer.clone()),
                    WriteDescriptorSet::buffer(2, camera_buffer.clone()),
                    WriteDescriptorSet::buffer(3, ubo_buffer.clone()),
                    WriteDescriptorSet::buffer(4, tlci_buffer.clone()),
                    WriteDescriptorSet::buffer(5, vmi_buffer.clone()),
                ].into_iter().chain(
                    chunk_bitmask_buffers.iter()
                        .zip(6..6+chunk_bitmask_buffers.len())
                        .map(|(cbm_buffer, i)|
                             WriteDescriptorSet::buffer(i as u32, cbm_buffer.clone()),
                        )
                ).collect_vec(),
                [],
            ).unwrap()
        ).collect();

    compute_descriptor_sets.iter().map(|descriptor_set| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .bind_pipeline_compute(compute_pipeline.clone()).unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0,
                    Arc::clone(descriptor_set),
                )
                .unwrap()
                .dispatch([dimensions.width / SUBGROUP_WIDTH, dimensions.height / SUBGROUP_HEIGHT, 1])
                .unwrap();

            builder.build().unwrap()
        })
        .collect()
}

fn tlc_for_xz(x: i32, z: i32) -> (i32, i32) {
    (((x as f32) / (TLC_SIZE as f32)).floor() as i32, (((z as f32) / (TLC_SIZE as f32)).floor() as i32))
}

// Get a voxel's global index at each chunk level (0 = voxel index, ... N_CHUNK_LVLS = top level chunk index)
fn voxel_index(x: i32, y: i32, z: i32, tlci: &[TopLevelChunkIndex; RENDER_N_TLCS*RENDER_N_TLCS/4], min_tlcx: i32, min_tlcz: i32) -> [usize; N_CHUNK_LVLS+1] {
    let mut idx: [usize; N_CHUNK_LVLS+1] = [0; N_CHUNK_LVLS+1];

    // Top level chunk
    let (tlcx, tlcz) = tlc_for_xz(x, z);
    let tlc_idx_to_look_up = (tlcx - min_tlcx) as usize + (tlcz - min_tlcz) as usize * RENDER_N_TLCS;
    idx[N_CHUNK_LVLS] = tlci[tlc_idx_to_look_up/4].indices[tlc_idx_to_look_up % 4] as usize;

    // Position in top level chunk
    for i in (0..N_CHUNK_LVLS).rev() {
        // x, y, z in units of the current level
        let xlvl = ((x as f32 / CHUNK_BLOCK_SIZES[i] as f32).floor() as i64).amod(CHUNK_SIZE as i64) as usize;
        let ylvl = ((y as f32 / CHUNK_BLOCK_SIZES[i] as f32).floor() as i64).amod(CHUNK_SIZE as i64) as usize;
        let zlvl = ((z as f32 / CHUNK_BLOCK_SIZES[i] as f32).floor() as i64).amod(CHUNK_SIZE as i64) as usize;
        idx[i] = idx[i+1] * CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE + xlvl + ylvl * CHUNK_SIZE*CHUNK_SIZE + zlvl * CHUNK_SIZE;
    }

    idx
}

fn update_voxel(voxel_idx: [usize; N_CHUNK_LVLS+1], material_id: u8, vmi: &mut Vec<MaterialIndex>, bitmasks: &mut Vec<Vec<VoxelBitmask>>) -> Result<(), OutOfBoundsErr> {
    let vmi_idx1 = voxel_idx[0]/16;
    let vmi_idx2 = (voxel_idx[0] / 4) % 4;
    let mut bytes = vmi[vmi_idx1].indices[vmi_idx2].to_le_bytes();
    bytes[(voxel_idx[0] / 4) % 4] = material_id.to_le_bytes()[0];
    vmi[vmi_idx1].indices[vmi_idx2] = u32::from_le_bytes(bytes);

    for (bitmask, idx) in bitmasks.iter_mut().zip( voxel_idx.iter()) {
        if let Some(bm) = bitmask.get_mut(idx / 128) {
            let bit: u32 = 1 << (idx % 32);
            if material_id == 0 {
                bm.mask[(idx % 128) / 32] &= !bit;
            }
            else {
                bm.mask[(idx % 128) / 32] |= bit;
            }
        }
        else { return Err(OutOfBoundsErr { index: idx / 128, size: bitmask.len() }); }
    }
    // TODO: Add region to transfer to GPU
    Ok(())
}


fn main() {
    println!("Num chunks: {N_CHUNKS:?}");

    /////////////////
    // Vulkan init //
    /////////////////

    let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let event_loop = EventLoop::new();

    let required_extensions = Surface::required_extensions(&event_loop);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions.union(&InstanceExtensions { ext_validation_features: true, ..Default::default() }),
            enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_string()],
            enabled_validation_features: vec![ValidationFeatureEnable::BestPractices, ValidationFeatureEnable::GpuAssisted, ValidationFeatureEnable::GpuAssistedReserveBindingSlot, ValidationFeatureEnable::SynchronizationValidation],
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    window.set_cursor_grab(CursorGrabMode::Locked).unwrap_or_default();
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let window = surface
        .object()
        .unwrap()
        .clone()
        .downcast::<Window>()
        .unwrap();
    let dimensions = window.inner_size();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ext_scalar_block_layout: true,
        ..DeviceExtensions::empty()
    };

    let physical_device = select_physical_device(&instance);
    let transfer_queue_family_i = physical_device.queue_family_properties().iter().position(|f| f.queue_flags.contains(QueueFlags::TRANSFER)).expect("No transfer queue") as u32;
    let compute_queue_family_i = physical_device.queue_family_properties().iter().position(|f| f.queue_flags.contains(QueueFlags::COMPUTE)).expect("No compute queue") as u32;
    let graphics_queue_family_i = physical_device.queue_family_properties().iter().position(|f| f.queue_flags.contains(QueueFlags::GRAPHICS)).expect("No graphics queue") as u32;

    let (queue_idxs, queue_family_idxs) = {
        if transfer_queue_family_i == compute_queue_family_i && compute_queue_family_i == graphics_queue_family_i {
            (vec![transfer_queue_family_i], vec![0, 0, 0])
        }
        else if transfer_queue_family_i == compute_queue_family_i {
            (vec![transfer_queue_family_i, graphics_queue_family_i], vec![0, 0, 1])
        }
        else if transfer_queue_family_i == graphics_queue_family_i {
            (vec![transfer_queue_family_i, compute_queue_family_i], vec![0, 1, 0])
        }
        else if compute_queue_family_i == graphics_queue_family_i {
            (vec![transfer_queue_family_i, compute_queue_family_i], vec![0, 1, 1])
        }
        else {
            (vec![transfer_queue_family_i, compute_queue_family_i, graphics_queue_family_i], vec![0, 1, 2])
        }
    };

    let (device, queues) = Device::new(
        Arc::clone(&physical_device),
        DeviceCreateInfo {
            queue_create_infos: queue_idxs.iter().map(|_| QueueCreateInfo {
                queue_family_index: transfer_queue_family_i,
                ..Default::default()
            }).collect_vec(),
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .expect("failed to create device");

    let queues = queues.collect_vec();
    let (transfer_queue, compute_queue, graphics_queue) = (
        queues[queue_family_idxs[0] as usize].clone(),
        queues[queue_family_idxs[1] as usize].clone(),
        queues[queue_family_idxs[2] as usize].clone()
    );

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(Arc::clone(&device)));

    ///////////////
    // Data init //
    ///////////////

    // Top level chunk index
    let tlci = [
        TopLevelChunkIndex { indices: [0, 1, 2, 3] },
        TopLevelChunkIndex { indices: [4, 5, 6, 7] },
        TopLevelChunkIndex { indices: [8, 9, 10, 11] },
        TopLevelChunkIndex { indices: [12, 13, 14, 15] },
    ];

    // Voxel material index
    let mut vmi: Vec<MaterialIndex> = (0..VMI_BUFFER_LEN)
        .map(|_| MaterialIndex { indices: [0; 4] })
        .collect();

    // Bitmasks at each chunk level (excluding 0)
    let mut chunk_bitmasks: Vec<Vec<VoxelBitmask>> = (1..N_CHUNKS.len())
        .map(|i| vec![VoxelBitmask { mask: [0; 4] }; N_CHUNKS[i] * CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE / 128])
        .collect_vec();

    let mut camera = Camera {
        position: Point3 { x: 14., y: 15.5, z: 3.0 },
        pitch: Rad(0.0),
        yaw: Rad(-PI),
        viewport_dist: 0.1,
        resolution: (dimensions.width, dimensions.height),
        avg_fov: Rad(1.),
        controller: CameraController::new(10., 1.),
    };

    let ubo = Ubo {
        sun_dir: [0.39036, 0.78072, 0.48795],
        tlc_minxi: -2,
        tlc_minzi: -2,
        time: 0,
    };

    println!("UBO: {:?}", ubo);

    for tlcx in -2..2 {
        for tlcy in -2..2 {
            for x in 0..TLC_SIZE as i32 {
                for y in 0..(tlcx+tlcy + 5) {
                    for z in 0..TLC_SIZE as i32 {
                        match update_voxel(
                            voxel_index(x + tlcx * TLC_SIZE as i32, y + if x % 12 == 0 { 1 } else { 0 }, z + tlcy * TLC_SIZE as i32, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
                            if x % 12 == 0 { 3 } else if z % 7 == 0 { 2 } else { 1 },
                            &mut vmi,
                            &mut chunk_bitmasks,
                        ) {
                            Ok(_) => {},
                            Err(e) => { panic!("Failed to update voxel ({}, {}, {}) in TLC ({}, {}) : {}", x, y, z, tlcx, tlcy, e); }
                        }
                    }
                }
            }
        }
    }

    // update_voxel(
    //     voxel_index(-1, 14, 3, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
    //     1,
    //     &mut vmi,
    //     &mut chunk_bitmasks,
    // ).unwrap();
    update_voxel(
        voxel_index(1, 14, 3, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        3,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    // update_voxel(
    //     voxel_index(-1, 15, 7, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
    //     1,
    //     &mut vmi,
    //     &mut chunk_bitmasks,
    // ).unwrap();
    update_voxel(
        voxel_index(1, 15, 7, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        2,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    // update_voxel(
    //     voxel_index(-3, 13, 10, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
    //     1,
    //     &mut vmi,
    //     &mut chunk_bitmasks,
    // ).unwrap();
    update_voxel(
        voxel_index(3, 13, 10, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(9, 15, 7, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(9, 15, 2, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(9, 15, 13, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();

    update_voxel(
        voxel_index(9, 15, 14, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(8, 15, 14, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(9, 15, 13, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(8, 15, 13, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(9, 14, 14, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(8, 14, 14, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(9, 14, 13, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();
    update_voxel(
        voxel_index(8, 14, 13, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
        1,
        &mut vmi,
        &mut chunk_bitmasks,
    ).unwrap();

    // for x in -25..25 {
    //     for y in 0..25 {
    //         for z in -25..25 {
    //             update_voxel(
    //                 voxel_index(x, y, z, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
    //                 1,
    //                 &mut vmi,
    //                 &mut chunk_bitmasks,
    //             ).unwrap();
    //         }
    //     }
    // }
    // for x in -24..24 {
    //     for y in 0..24 {
    //         for z in -24..24 {
    //             update_voxel(
    //                 voxel_index(x, y, z, &tlci, ubo.tlc_minxi, ubo.tlc_minzi),
    //                 0,
    //                 &mut vmi,
    //                 &mut chunk_bitmasks,
    //             ).unwrap();
    //         }
    //     }
    // }

    let idx = voxel_index(-3, 13, 10, &tlci, ubo.tlc_minxi, ubo.tlc_minzi);
    println!("0: {}, {}", idx[0], idx[0] % 128);
    println!("0: {}", vmi[idx[0] / 16].indices[(idx[0] / 4) % 4]);
    println!("0: {:#b}", chunk_bitmasks[0][idx[0]/128].mask[(idx[0]%128)/32]);
    println!("1: {}, {}", idx[1], idx[1] % 128);
    println!("1: {:#b}", chunk_bitmasks[1][idx[1]/128].mask[(idx[1]%128)/32]);

    let mut sum = 0;
    for vmi_i in vmi.iter() {
        for x in vmi_i.indices.iter() {
            if *x > 0 {
                sum += 1;
            }
        }
    }
    println!("{}", sum);

    // Camera

    let camera_staging_buffer: Subbuffer<CameraUBO> = Buffer::from_data(
        memory_allocator.clone(),
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
        CameraUBO::new(&camera, Point3::new(0.0, 0.0, 0.0)).clone(),
    ).unwrap();

    let camera_buffer: Subbuffer<CameraUBO> = Buffer::new_sized(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::UNIFORM_BUFFER,
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

    // UBO

    let ubo_staging_buffer = Buffer::from_data(
        memory_allocator.clone(),
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
        ubo,
    ).unwrap();

    let ubo_buffer: Subbuffer<Ubo> = Buffer::new_sized(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::UNIFORM_BUFFER,
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

    // Top level chunk index

    let tlci_staging_buffer = Buffer::from_iter(
        memory_allocator.clone(),
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
        tlci,
    ).unwrap();

    let tlci_buffer: Subbuffer<[TopLevelChunkIndex]> = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::UNIFORM_BUFFER,
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
        tlci_staging_buffer.len()
    ).unwrap();

    // Materials

    let temp_material_staging_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC | BufferUsage::UNIFORM_BUFFER,
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
        MATERIALS,
    ).unwrap();

    let material_buffer: Subbuffer<[Material]> = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::UNIFORM_BUFFER,
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
        temp_material_staging_buffer.len(),
    ).unwrap();

    // Voxel material index buffer

    let vmi_staging_buffer: Subbuffer<[MaterialIndex]> = Buffer::from_iter(
        memory_allocator.clone(),
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
        vmi,
    ).unwrap();

    let vmi_buffer: Subbuffer<[MaterialIndex]> = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
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
        vmi_staging_buffer.len(),
    ).unwrap();

    // Bitmasks

    let chunk_bitmask_staging_buffers: Vec<Subbuffer<[VoxelBitmask]>> = chunk_bitmasks.iter()
        .map(|cbm|
            Buffer::from_iter(
                memory_allocator.clone(),
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
                cbm.iter().copied(),
            ).unwrap()
        ).collect();

    let chunk_bitmask_buffers: Vec<Subbuffer<[VoxelBitmask]>> = chunk_bitmask_staging_buffers
        .iter()
        .map(|cbm_sb|
            Buffer::new_slice(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
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
                cbm_sb.len(),
            ).unwrap()
        ).collect();


    ////////////////////
    // Pipeline setup //
    ////////////////////

    let (mut swapchain, images) = {
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

        println!("MIC: {:?}", caps.min_image_count);

        Swapchain::new(
            Arc::clone(&device),
            surface,
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

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 2,
                ..Default::default()
            }
        );

    // Transfer stage

    let continuous_transfer_sub_command_buffer: Arc<SecondaryAutoCommandBuffer> = {
        let mut builder = AutoCommandBufferBuilder::secondary(
            &command_buffer_allocator,
            transfer_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: None,
                occlusion_query: None,
                query_statistics_flags: Default::default(),
                ..Default::default()
            },
        ).unwrap();

        builder.copy_buffer(CopyBufferInfo::buffers(
            temp_material_staging_buffer.clone(),
            material_buffer.clone(),
        )).unwrap();
        builder.copy_buffer(CopyBufferInfo::buffers(
            camera_staging_buffer.clone(),
            camera_buffer.clone(),
        )).unwrap();
        builder.copy_buffer(CopyBufferInfo::buffers(
            ubo_staging_buffer.clone(),
            ubo_buffer.clone(),
        )).unwrap();
        builder.copy_buffer(CopyBufferInfo::buffers(
            tlci_staging_buffer.clone(),
            tlci_buffer.clone(),
        )).unwrap();

        builder.build().unwrap()
    };

    let one_time_transfer_command_buffer: Arc<PrimaryAutoCommandBuffer> = {
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            transfer_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        builder.execute_commands(continuous_transfer_sub_command_buffer.clone()).unwrap();

        builder.copy_buffer(CopyBufferInfo::buffers(
            vmi_staging_buffer.clone(),
            vmi_buffer.clone(),
        )).unwrap();
        for (staging, local) in chunk_bitmask_staging_buffers
            .iter()
            .zip(chunk_bitmask_buffers.iter()) {
            builder.copy_buffer(CopyBufferInfo::buffers(
                staging.clone(),
                local.clone(),
            )).unwrap();
        };

        builder.build().unwrap()
    };

    one_time_transfer_command_buffer.execute(transfer_queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let continuous_transfer_command_buffer: Arc<PrimaryAutoCommandBuffer> = {
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            transfer_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        ).unwrap();
        builder.execute_commands(continuous_transfer_sub_command_buffer).unwrap();

        builder.build().unwrap()
    };


    // Compute stage

    let shader = raytrace_shader::load(Arc::clone(&device)).unwrap();

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
        Arc::clone(&device),
        StandardDescriptorSetAllocatorCreateInfo { ..Default::default() },
    );

    let compute_pipeline = get_compute_pipeline(
        Arc::clone(&device),
        Arc::clone(&shader),
    );

    let mut compute_command_buffers = get_compute_command_buffers(
        &images,
        &descriptor_set_allocator,
        &command_buffer_allocator,
        &material_buffer,
        &camera_buffer,
        &ubo_buffer,
        &tlci_buffer,
        &vmi_buffer,
        &chunk_bitmask_buffers,
        &compute_queue,
        &compute_pipeline,
        &dimensions,
    );


    ////////////////
    // Event loop //
    ////////////////

    println!("Materials staging buffer: {} ({} bytes)", temp_material_staging_buffer.len(), temp_material_staging_buffer.size());
    println!("Materials buffer: {} ({} bytes)", material_buffer.len(), material_buffer.size());
    println!("UBO staging buffer: {} bytes", ubo_staging_buffer.size());
    println!("UBO buffer: {} bytes", ubo_buffer.size());
    println!("TCLI staging buffer: {} ({} bytes)", tlci_staging_buffer.len(), tlci_staging_buffer.size());
    println!("TCLI buffer: {} ({} bytes)", tlci_buffer.len(), tlci_buffer.size());
    println!("VMI staging buffer: {} ({} MB)", vmi_staging_buffer.len(), vmi_staging_buffer.size() as f32 / (1024.0*1024.0));
    println!("VMI buffer: {} ({} MB)", vmi_buffer.len(), vmi_buffer.size() as f32 / (1024.0*1024.0));
    for (i, b) in chunk_bitmask_staging_buffers.iter().enumerate() {
        println!("Chunk bitmask buffer {}: {} ({} MB)", i, b.len(), b.size() as f32 / (1024.0*1024.0));
    }

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut transfer_fence: Option<Arc<FenceSignalFuture<_>>> = None;
    let mut compute_fence: Option<Arc<FenceSignalFuture<_>>> = None;
    let mut present_fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;
    let mut last_render_time = Instant::now();


    event_loop.run(move |event, _, control_flow| match event {
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion{ delta, },
            .. // We're not using device_id currently
        } => {
            camera.controller.process_mouse(delta.0, delta.1)
        }
        Event::WindowEvent { event, .. } => {
            match event {
                WindowEvent::CloseRequested => { *control_flow = ControlFlow::Exit; }
                WindowEvent::Resized(_) => { window_resized = true; }
                WindowEvent::KeyboardInput {
                    input: KeyboardInput {
                        virtual_keycode: Some(key),
                        state,
                        ..
                    },
                    ..
                } => { camera.controller.process_keyboard(key, state); }
                _ => ()
            }
        }
        Event::MainEventsCleared => {
            if window_resized || recreate_swapchain {
                recreate_swapchain = false;

                let new_dimensions = window.inner_size();
                window.set_cursor_grab(CursorGrabMode::Confined).unwrap_or_default();
                camera.resolution = (new_dimensions.width, new_dimensions.height);

                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: new_dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(e) => panic!("failed to recreate swapchain: {e:?}"),
                };
                swapchain = new_swapchain;

                if window_resized {
                    window_resized = false;

                    let new_pipeline = get_compute_pipeline(
                        Arc::clone(&device),
                        shader.clone(),
                    );
                    compute_command_buffers = get_compute_command_buffers(
                        &new_images,
                        &descriptor_set_allocator,
                        &command_buffer_allocator,
                        &material_buffer,
                        &camera_buffer,
                        &ubo_buffer,
                        &tlci_buffer,
                        &vmi_buffer,
                        &chunk_bitmask_buffers,
                        &compute_queue,
                        &new_pipeline,
                        &new_dimensions,
                    );
                }
            }

            let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), Some(Duration::from_secs(3))) {
                    Ok(r) => r,
                    Err(Validated::Error(VulkanError::OutOfDate)) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => {
                        panic!("failed to acquire next image: {e:?}")
                    },
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            // Update

            let now = Instant::now();
            let dt = now - last_render_time;
            last_render_time = now;

            camera.apply_controller_updates(dt);

            // Wait for staging buffers to be available

            if let Some(tf) = &transfer_fence {
                tf.wait(Some(Duration::from_secs(3))).unwrap();
            }

            // Transfer to staging buffers
            // TODO: Parallelize this (at least)

            {
                let mut camera_buffer_content = camera_staging_buffer.write().unwrap();
                camera_buffer_content.update(&camera, Point3::new(0., 0., 0.));
            }
            {
                let mut ubo_buffer_content = ubo_staging_buffer.write().unwrap();
                ubo_buffer_content.time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_millis();
            }

            // Wait for the GPU to be done reading from device local buffer from previous frame (they are shared across frames)
            // (and also make sure this image is available to be drawn to)

            if let Some(fence) = &compute_fence {
                fence.wait(Some(Duration::from_secs(3))).unwrap();
            }
            if let Some(image_fence) = &present_fences[previous_fence_i as usize] {
                image_fence.wait(Some(Duration::from_secs(3))).unwrap();
            }

            // Transfer from staging buffer to device local buffer

            let previous_transfer_future = match transfer_fence.clone() {
                None => {
                    let mut now = sync::now(Arc::clone(&device));
                    now.cleanup_finished();

                    now.boxed()
                }
                Some(future) => { future.boxed() }
            };

            let transfer_future = previous_transfer_future
                .then_execute(transfer_queue.clone(), continuous_transfer_command_buffer.clone())
                .unwrap()
                .then_signal_fence_and_flush();

            transfer_fence = match transfer_future {
                Ok(value) => Some(Arc::new(value)),
                Err(e) => {
                    println!("failed to flush future: {e:?}");
                    None
                }
            };

            // wait for the fence related to this image to finish (normally this would be the oldest fence)
            if let Some(image_fence) = &present_fences[image_i as usize] {
                image_fence.wait(Some(Duration::from_secs(3))).unwrap();
            }

            let previous_future = match present_fences[previous_fence_i as usize].clone() {
                // Create a NowFuture
                None => {
                    let mut now = sync::now(Arc::clone(&device));
                    now.cleanup_finished();

                    now.boxed()
                }
                // Use the existing FenceSignalFuture
                Some(fence) => fence.boxed(),
            };

            // Join futures and set fence for this image

            let compute_future = previous_future
                .join(transfer_fence.clone().unwrap())
                .join(acquire_future)
                .then_execute(compute_queue.clone(), compute_command_buffers[image_i as usize].clone())
                .unwrap()
                .then_signal_fence_and_flush();

            compute_fence = match compute_future {
                Ok(value) => Some(Arc::new(value)),
                Err(e) => {
                    println!("failed to flush future: {e:?}");
                    None
                }
            };

            let future = compute_fence.clone().unwrap()
                .then_swapchain_present(
                    graphics_queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

            present_fences[image_i as usize] = match future {
                Ok(value) => Some(Arc::new(value)),
                Err(Validated::Error(VulkanError::OutOfDate)) => {
                    recreate_swapchain = true;
                    None
                }
                Err(e) => {
                    println!("failed to flush future: {e:?}");
                    None
                }
            };

            previous_fence_i = image_i;
        }
        _ => (),
    });
}