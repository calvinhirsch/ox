use std::sync::Arc;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::debug::ValidationFeatureEnable;
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use winit::event_loop::EventLoop;
use winit::window::{CursorGrabMode, Window, WindowBuilder};

pub struct Context {
    pub instance: Arc<Instance>,
    pub surface: Arc<Surface>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub transfer_queue: Arc<Queue>,
    pub compute_queue: Arc<Queue>,
    pub graphics_queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
}
impl Context {
    pub fn new(event_loop: &EventLoop<()>) -> (Self, Arc<Window>) {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

        let required_extensions = Surface::required_extensions(&event_loop);
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions.union(&InstanceExtensions {
                    ext_validation_features: true,
                    ..Default::default()
                }),
                enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_string()],
                enabled_validation_features: vec![
                    ValidationFeatureEnable::BestPractices,
                    ValidationFeatureEnable::GpuAssisted,
                    ValidationFeatureEnable::GpuAssistedReserveBindingSlot,
                    ValidationFeatureEnable::SynchronizationValidation,
                ],
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
        window
            .set_cursor_grab(CursorGrabMode::Locked)
            .unwrap_or_default();

        let surface = Surface::from_window(Arc::clone(&instance), window).unwrap();

        let window = surface
            .object()
            .unwrap()
            .clone()
            .downcast::<Window>()
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ext_scalar_block_layout: true,
            ..DeviceExtensions::empty()
        };

        let physical_device = instance
            .enumerate_physical_devices()
            .unwrap()
            .next()
            .expect("No devices.");
        let transfer_queue_family_i = physical_device
            .queue_family_properties()
            .iter()
            .position(|f| f.queue_flags.contains(QueueFlags::TRANSFER))
            .expect("No transfer queue") as u32;
        let compute_queue_family_i = physical_device
            .queue_family_properties()
            .iter()
            .position(|f| f.queue_flags.contains(QueueFlags::COMPUTE))
            .expect("No compute queue") as u32;
        let graphics_queue_family_i = physical_device
            .queue_family_properties()
            .iter()
            .position(|f| f.queue_flags.contains(QueueFlags::GRAPHICS))
            .expect("No graphics queue") as u32;

        let (queue_idxs, queue_family_idxs) = {
            if transfer_queue_family_i == compute_queue_family_i
                && compute_queue_family_i == graphics_queue_family_i
            {
                (vec![transfer_queue_family_i], vec![0, 0, 0])
            } else if transfer_queue_family_i == compute_queue_family_i {
                (
                    vec![transfer_queue_family_i, graphics_queue_family_i],
                    vec![0, 0, 1],
                )
            } else if transfer_queue_family_i == graphics_queue_family_i {
                (
                    vec![transfer_queue_family_i, compute_queue_family_i],
                    vec![0, 1, 0],
                )
            } else if compute_queue_family_i == graphics_queue_family_i {
                (
                    vec![transfer_queue_family_i, compute_queue_family_i],
                    vec![0, 1, 1],
                )
            } else {
                (
                    vec![
                        transfer_queue_family_i,
                        compute_queue_family_i,
                        graphics_queue_family_i,
                    ],
                    vec![0, 1, 2],
                )
            }
        };

        let (device, queues) = Device::new(
            Arc::clone(&physical_device),
            DeviceCreateInfo {
                queue_create_infos: queue_idxs
                    .iter()
                    .map(|_| QueueCreateInfo {
                        queue_family_index: transfer_queue_family_i,
                        ..Default::default()
                    })
                    .collect(),
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queues = queues.collect();
        let (transfer_queue, compute_queue, graphics_queue) = (
            queues[queue_family_idxs[0] as usize].clone(),
            queues[queue_family_idxs[1] as usize].clone(),
            queues[queue_family_idxs[2] as usize].clone(),
        );

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(Arc::clone(&device)));

        (
            Context {
                instance,
                surface,
                physical_device,
                device,
                transfer_queue,
                compute_queue,
                graphics_queue,
                memory_allocator,
            },
            window,
        )
    }
}
