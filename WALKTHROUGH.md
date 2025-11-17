# Walkthrough


This will explain most of ox's high level ideas by going through some of the code in `example_game/` and explaining the concepts along the way.

In order to use ox, we need to define a few main things:
1. A `VoxelTypeEnum` to define our voxel types.
2. A `World` to store our world data
3. A function to generate/load chunk data
4. A `ChunkLoader` to populate our world data as we move around in it
5. A `Renderer` to convert our world data into pixels on the screen

# 1. Defining our voxel types with `VoxelTypeEnum`

This can be found in `example_game/src/blocks.rs`.

Here, we define a list of all the possible voxel materials that we want to render.
Then, we will just refer to these definitions by ID later.

```rust
/// The additional attributes (other than render material) to store about each voxel type
pub struct BlockTypeAttrs {
    #[allow(dead_code)]
    dollars: u32,
}

/// Defining our possible voxel types
#[derive(Debug, Sequence, Clone, Copy, FromPrimitive, ToPrimitive, PartialEq, Eq, Hash)]
pub enum Block {
    Air,
    Debug,
    Grass,
    Dirt,
    Rock,
    Snow,
    Mirror,
    RedLight,
    GreenLight,
    BlueLight,
    Metal,
    GrayCarpet,
}

impl VoxelTypeEnum for Block {
    type VoxelAttributes = BlockTypeAttrs;

    fn def(&self) -> VoxelTypeDefinition<Self::VoxelAttributes> {
        use Block::*;
        match *self {
            Air => VoxelTypeDefinition {
                material: Material::default(),
                is_visible: false,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
            Debug => VoxelTypeDefinition {
                material: Material {
                    color: [1., 0., 0.],
                    emission_color: [1., 0., 0.],
                    emission_strength: 1.2,
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 0 },
            },
            Grass => VoxelTypeDefinition {
                material: Material {
                    color: [0.38, 0.45, 0.25],
                    ..Default::default()
                },
                is_visible: true,
                attributes: BlockTypeAttrs { dollars: 3 },
            },
            /* ... */
        }
    }

    fn empty() -> Block {
        Block::Air
    }
}
```

Here, we define `Block` to be our enum that lists all possible voxel types.
Each variant represents one.

Then, we must implement `VoxelTypeEnum` on it.
The `def` method should take a voxel type and return its definition.
The definition includes all the information needed to render it, including the material and whether it is visible.
It also can include additional attributes if we want, which here we define with `BlockTypeAttrs` and we
additionally define a `dollars` value for each one.

This enum should be representable as a `u8`, so it can have up to 256 variants.



# 2. Creating a `World`

The first thing we need to do is create an `ox::World`. This requires a few things:
 - A `MemoryGrid` that will store the world state
 - A `Camera` that will determine the perspective being rendered
 - Some other minor parameters


## Creating a `MemoryGrid`

### Chunking overview

In ox, the world is split up into hierarchical 3D chunks, in a configurable way.
For this example, we will use a "chunk size" of 8 with two chunk levels.
This means that level 0 chunks are individual voxels, level 1 chunks 8x8x8, and level 2 chunks are 64x64x64.

The purpose of this hierarchical system is that it enables faster ray tracing.
Rays are traced through the world and often will pass through lots of open space.
Rather than check every individual voxel it crosses to see if it is solid or air,
we also store whether entire chunks (at all chunk levels) are empty, allowing us to skip them entirely during ray traversal.
Rays can actually traverse the different chunk levels (0, individual voxels, 1, 8x8x8 chunks, or 2, 64x64x64 chunks) in parallel
in different threads of the same warp on the GPU.

### Creating our memory grid

The first thing we need to do is define what data we actually want to store for each chunk.
In most cases, you will want to store voxel data. Otherwise, you won't be able to use ox's rendering capabilities.

Let's also assume we have some "entities" in each chunk that we want to keep track of. These won't really be used
but this is just to demonstrate how you can store whatever data you want in this system.
```rust
#[derive(Debug, Clone)]
pub struct Entity {
    pub _position: Point3<f32>,
    pub _name: String,
}
```

We can then define our memory grid like this:
```rust
/// This is what we will store in each chunk to track entities
#[derive(Debug, Clone)]
pub struct Entities {
    pub entities: Vec<Entity>,
}

/// Our custom memory grid
#[derive(Debug)]
pub struct WorldMemoryGrid<const N: usize> {
    pub voxel: VoxelMemoryGrid<N>,
    pub entity: MemoryGridLayer<Entities>,
}
```

### Memory grid layers

Ox refers to different **layers** of the memory grid.
A layer is essentially a circular/ring buffer of top level chunk (TLC) data.
For example, a single level of detail (LOD) of voxel data is a layer.
For each TLC, it stores a big array of IDs of all the voxels in it
(these IDs represent the `Block` enum variants we defined earlier).
When the camera moves between chunks on an axis, it will unload the furthest chunks and load new ones on the opposite side.
This essentially moves the "start" point of the circular buffer along that axis.
If we are moving in the positive direction, the most negative chunks' memory is overwritten with the memory of the newly loaded, most positive chunks on the other side.
This action is generally called a **shift** of the memory grid layer.

This layer logic is abstracted in a reusable type, `MemoryGridLayer`.
It takes a generic type to determine what data is being stored for each TLC, and handles all the layer shifting logic.
In our example above, we are creating a `MemoryGridLayer` that stores the entity data using the `Entities` struct we just defined.

`MemoryGridLayer` is actually slightly more complicated than described above because it has **buffer chunks**,
which are basically used to pre-load chunks before they are actually needed.
So, if a memory grid layer has size 8x8x8, we only use 7x7x7 of that and the rest are used as buffer chunks.
`MemoryGridLayer` will trigger chunk loading as needed, which we will see later.

Each layer can have its own size, so for example, you could have voxel data (i.e., your render distance)
reach further away from the camera than the entity data.
In fact, the voxel data (`VoxelMemoryGrid`) is made up of several layers of different levels of detail.

### Voxel levels of detail (LODs)

Ox uses multiple voxel levels of detail.
When rendering voxels further away from the camera, it renders them as larger voxels (e.g., 2x2x2 or 4x4x4).
This greatly reduces the total amount of data that needs to be stored both on the CPU and GPU, and speeds up rendering.

Each level of detail is represented as a memory grid layer. These are all stored within `VoxelMemoryGrid`.
Levels of detail are generally referred to by their chunk level and sublevel.
Chunk level 0 sublevel 0 is individual 1x1x1 voxels.

| lvl | sublvl | voxel size |
|---|---|---|
| 0 | 0 | 1 |
| 0 | 1 | 2 |
| 0 | 2 | 4 |
| 1 | 0 | 8 |
| 1 | 1 | 16 |
| 1 | 2 | 32 |
| 2 | 0 | 64 |

The "highest" level of detail (full resolution 1x1x1 voxels) will have the smallest render distance or memory grid layer size.
The lower LODs will have larger layer sizes since they are rendered further away.
Even though the lower LODs are not used when a higher LOD is present, they are still stored for those chunks for simplicity of
memory layout and handling. This causes some extra memory usage.

You can think of memory grid layers (such as voxel LODs but also any others) as overlapping regions of different sizes, all
centered at the camera.
A specific chunk will have data present in each layer (so long as it is within the "render distance" for that layer).
However, if the memory grid shifts, the layers that are different sizes desync, so those corresponding data for that chunk across layers
may now belong to multiple different chunks.
This logic is all handled by the memory grid internals and so is not necessary to understand to effectively use ox.
When a user wants to deal with a specific chunk, they simply request it by position and the relevant pieces of memory from all
layers will be returned. We will see this in action later.

### Back to memory grid definition

```rust
/// Our custom memory grid
#[derive(Debug)]
pub struct WorldMemoryGrid<const N: usize> {
    pub voxel: VoxelMemoryGrid<N>,
    pub entity: MemoryGridLayer<Entities>,
}
```

Looking again at our definition of the world memory grid, we have the `VoxelMemoryGrid` and one additional layer for entity data.

Note that `VoxelMemoryGrid` has a const generic `N`. This defines the number of LODs.

When defining a memory grid, it's recommended to have each field be a `MemoryGrid` implementor as well.
`MemoryGridLayer`s implement `MemoryGrid`.
This makes it very easy to implement `MemoryGridLoadChunks` and `MemoryGrid`, which are required traits.
These will have derive macros in the future that will work if all your fields are `MemoryGrid`s.
We won't go over these, but you can look in `example_game/src/world.rs` to see their implementations.

### Allowing editing

In order to allow editing of the world memory grid data, we need to provide an editor and a way to construct that editor.

```rust
#[derive(Debug)]
pub struct WorldChunkEditor<'a, const N: usize> {
    pub voxel: ChunkVoxelEditor<'a, Block, N>,
    pub entity: Option<DefaultLayerChunkEditor<'a, Entities>>,
}
```

Here, we use the existing `ChunkVoxelEditor` for editing voxel data and the `DefaultLayerChunkEditor` for editing entity data.
We're assuming here that entity data may not be present for a chunk (so it's `Option`) but voxel data always will be.
In other words, the voxel memory grid size is >= the entity grid size.

Then, we implement the `EditMemoryGridChunk` trait for `WorldMemoryGrid` to enable editing:

```rust
impl<const N: usize> EditMemoryGridChunk for WorldMemoryGrid<N> {
    type ChunkEditor<'a> = WorldChunkEditor<'a, N>
        where
            Self: 'a;

    fn edit_chunk(
        &mut self,
        pos: TlcPos<i64>,
        buffer_chunk_states: [ox::world::BufferChunkState; 3],
    ) -> Option<Self::ChunkEditor<'_>> {
        Some(WorldChunkEditor {
            voxel: self.voxel.edit_chunk(pos, buffer_chunk_states)?,
            entity: self.entity.edit_chunk(pos, buffer_chunk_states),
        })
    }
}
```

Implementing this trait enables calling a separate function, `World::edit_chunk` which is what you should generally use to edit world data.

### Creating the memory grid

Now, with all the definitions and implementations done, we can actually create a `WorldMemoryGrid`. Let's define a `new` function for it.

```rust
impl<const N: usize> WorldMemoryGrid<N> {
    pub fn new(
        voxel_mem_grid: VoxelMemoryGrid<N>,
        start_tlc: TlcPos<i64>,
        entity_loaded_area_size: usize,
    ) -> Self {
        let vox_size = voxel_mem_grid.size();
        let entity_grid_size = entity_loaded_area_size + 1;
        WorldMemoryGrid {
            voxel: voxel_mem_grid,
            entity: MemoryGridLayer::<Entities>::new(
                (0..cubed(entity_grid_size))
                    .map(|_| LayerChunk::new(Entities { entities: vec![] }))
                    .collect(),
                TlcPos(start_tlc.0 + Vector3::from_value((vox_size - entity_grid_size) as i64 / 2)),
                entity_grid_size,
                (),
                (),
            ),
        }
    }
}
```

Here, we're just taking in an already instantiated `VoxelMemoryGrid` and creating the entity layer.
We also take as a parameter how big the loaded area for the entity grid should be.
Now we need to create the `VoxelMemoryGrid` to pass in.

First, let's define our chunk size with a constant:

```rust
pub const CHUNK_SIZE: ChunkSize = ChunkSize::new(3);
```

Here we use `ox::world::mem_grid::utils::ChunkSize` and define it based on a power of 2 (2^3 = 8).

Then, we will create the `VoxelMemoryGrid` we will put in our `WorldMemoryGrid`. This code is from `main.rs`.

```rust
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
```

Here, we are defining a list of LODs. For each, we define a level, sublevel, and voxel resolution (as seen in the table before).
We also are defining the `render_area_size` for each one.
Note that they are all odd. This is because the size of the memory grid will be this value plus one, to store the buffer chunks.
This also gives an even center chunk to place the camera in.

We additionally specify bindings for the voxel bitmasks and IDs. This is so that we can find this data from the GPU shader.
Note that the voxel ID binding is optional. This is because not all layers need to have voxel data, but they must all have
a bitmask for the ray tracing to work.

NOTE: this configuration (chunk size and chunk level config) is currently not automatically synced between the Rust code and
the shader code. It is defined once here and again in the shader. Ultimately, this will probably be automatically set in the shader
with a template or something, but for now, they must both be set to the same thing.

Now, we can create the `WorldMemoryGrid`.

```rust
let mem_grid = WorldMemoryGrid::new(voxel_mem_grid, start_tlc, 5);
```


## Finally creating our `World`

```rust
let mut world = World::new(mem_grid, Camera::new(tlc_size, mem_grid_size), tlc_size, 16);
```



# 3. Generating or loading chunk data

Now that we have defined how we will be storing our world state and instantiated it, we need a way to populate it.
Ox allows us to define and pass in a function that will be called whenever a chunk needs to be loaded.
It's then up to us whether we want to generate it on the fly or load it from somewhere.
For now, let's just generate it.

`world.rs` has a more complex implementation of terrain generation, but we will just do something very simple here.

```rust
fn generate_chunk(
    chunk_pos: TlcPos<i64>,
    lvl: u8,
    sublvl: u8,
    voxel_ids_out: &mut ChunkVoxels,
    tlc_size: usize,
    largest_chunk_lvl: u8,
) {
    let voxel_size = CHUNK_SIZE.size().pow(lvl as u32) * 2usize.pow(sublvl as u32);
    let chunk_start_pt: VoxelPos<i64> = VoxelPos(chunk_pos.0 * tlc_size as i64);
    let grid_size = tlc_size / voxel_size;

    for x_grid in 0..grid_size as u32 {
        for z_grid in 0..grid_size as u32 {
            for y_grid in 0..grid_size as u32 {
                // world coords
                let y = y_grid as i64 * voxel_size as i64 + chunk_start_pt.0.y;
                let x = x_grid as i64 * voxel_size as i64 + chunk_start_pt.0.x;
                let z = z_grid as i64 * voxel_size as i64 + chunk_start_pt.0.z;

                // index in voxel_ids_out
                let idx = VoxelPosInLod {
                    pos: Point3 {
                        x: x_grid,
                        y: y_grid,
                        z: z_grid,
                    },
                    lvl,
                    sublvl,
                }
                .index(CHUNK_SIZE, largest_chunk_lvl);

                // draw strips of dirt at one specific height, everything else air
                voxel_ids_out[idx] = if x % 8 == 0 && (y == 64 * 7) {
                    Block::Dirt
                } else {
                    Block::Air
                } as u8;
            }
        }
    }
}
```

This function takes the chunk position, which chunk level and sublevel we are loading, and some other information,
and expects that we will write the voxels to `voxel_ids_out`.

That's it for now! We will use this next with the chunk loader.



# 4. Chunk loader

Chunk loading works by transferring ownership of chunk data from the memory grid (in `World`)
to the chunk loader. Then, chunk loading can happen in separate threads. When loading is done,
the data is moved back.


## "Taking" chunks

Memory grids are defined such that the data can be moved in and out for chunk loading.
However, we need a way to store a chunk's data outside the grid.
We already defined `WorldChunkEditor` earlier, which does something very similar,
but that was only borrowing the data from the memory grid.
We need to be able to "take" the data and own it.

We'll define another struct, similar to the editor, but owned:

```rust
#[derive(Debug)]
pub struct TakenWorldChunkEditor<const N: usize> {
    voxel: TakenChunkVoxelEditor<Block, N>,
    entity: Option<DefaultTakenLayerChunk<Entities>>,
}
```

Similarly to the regular editor, the "taken" editor also has a default implementation we can use for `MemoryGridLayer`: `DefaultTakenLayerChunk`.

Then, we need to implement `TakeChunkForLoading` on `WorldChunkEditor`.
This allows us to use a regular chunk editor to take the data from the memory grid.
We also need to implement `TakenChunk` on `TakenWorldChunkEditor`, which allows us to return the data to the memory grid.
These implementations are relatively straightforward and will likely have derive macros in the future, so we won't go through them.


## Using the chunk loader

With those implemented, we can now create the chunk loader:

```rust
let mut loader: ChunkLoader<
    WorldChunkLoadQueueItemData<N_LODS>,  // we will look at this soon
    TakenWorldChunkEditor<N_LODS>,
> = ChunkLoader::new(ChunkLoaderParams { n_threads: 48 });
```

The first thing we need to do with the chunk loader is track when we need to load new chunks.
The primary source of this is when we call `World::move_camera`, which might shift the memory grids.
We pass the loader directly into `move_camera`, and it will add the chunks that need to be loaded to the loader's queue.
This looks like:

```rust
world.move_camera(&mut camera_controller, dt, &mut loader);
```

We haven't discussed the camera controller yet, but this will be how we move the camera.

After we have chunks queued, we have to call `ChunkLoader::sync` to actually do the chunk loading.

```rust
loader.sync(&mut world, &load_chunk, voxel_md.clone());
```

This function can be called each frame of the game loop.
It will take chunks from `world` that it wants to start loading and return loaded chunks.

Here, we are passing in a mutable reference to the `world` we created earlier.
We are also passing in some metadata (third param).
We haven't yet defined `load_chunk` here.
This is a function that we need to define.
It will be called by the chunk loader when a chunk needs to be loaded.

Let's implement it like this:

```rust
pub fn load_chunk<const N: usize>(
    editor: &mut TakenWorldChunkEditor<N>,
    chunk: ChunkLoadQueueItem<WorldChunkLoadQueueItemData<N>>,
    params: VoxelMemoryGridMetadata,
) {
    editor.voxel.load_new(chunk.pos, generate_chunk, &params);
    if let Some(e) = editor.entity.as_mut() {
        e.chunk.entities.clear();
    }
}
```

Note that we get a `TakenWorldChunkEditor` to work with, which we defined above.
We get this type because we specified it as one of the generic types in `ChunkLoader` above.

We can then call `editor.voxel.load_new` to load the voxel data.
We pass in the `generate_chunk` function we defined in the last section to be used to generate the voxels at the appropriate LOD.

Then, we'll clear the entity list. You could do whatever you want with your custom data here, including something based on the resulting voxels from the prior call.



# Renderer

Finally, we can get to drawing stuff to the screen.

Earlier, we already defined the renderer context, event loop, and `renderer_voxel_data_component` when we were making `world`.
Let's re-examine some of that:

```rust
let event_loop = EventLoop::new();
let (renderer_context, window) = Context::new(&event_loop);
let (voxel_mem_grid, renderer_voxel_data_component) = VoxelMemoryGrid::new(/* ... */);
```

What is `renderer_voxel_data_component`? It's a renderer data component for the voxel data.

## Renderer data components

Data components in the renderer define pieces of data that need to get moved to the GPU.
If we look at our shader that does the ray tracing, we need a few things:
- Voxel data
- A static list of materials, one for each voxel type ID
- The camera position
- A UBO (uniform buffer object) containing some other misc. information (sun direction, time, start TLC)

In order to capture this, need to define a set of data components that contains these.
To do this, `Renderer` requires a struct that implements `ox::renderer::component::DataComponentSet`.
This is very easy to implement if all the fields of the struct are also `DataComponentSet`s.
For this case, there will be a derive macro in the future.

We can define our components like this:
```rust
struct RendererComponents {
    voxel_data: VoxelData<N_LODS>,
    material_list: MaterialList,
    camera: RendererCamera,
    ubo: RendererUBO,
}
```

Let's dig into the data component types a little.

`MaterialList`, `RendererCamera`, and `RendererUBO` are all just type aliases using generic tools from the `renderer` module.
Let's look at them:

#### MaterialList

```rust
pub type MaterialList = DataComponent<ConstantDeviceLocalBuffer<[Material]>>;
```

This is a `ConstantDeviceLocalBuffer` of a sequence of `Material`s. We use `ConstantDeviceLocalBuffer` because it never has to change.

#### RendererCamera

```rust
pub type RendererCamera = DataComponent<DualBufferWithFullCopy<CameraUBO>>;
```

Here, we use `DualBufferWithFullCopy`. Dual means that we have a staging buffer and a device local (GPU local) buffer.
First, we write to the staging buffer, and then transfer from that to the device local buffer. This avoids waiting for
buffers to be available. Full copy means that it always copies the full buffer every frame.

#### RendererUBO

```rust
pub type RendererUBO = DataComponent<DualBufferWithFullCopy<Ubo>>;
```

This one has the same setup as `RendererCamera`.

#### VoxelData

The voxel data is a little more complicated. The definition looks like this:
```rust
pub struct VoxelData<const N: usize> {
    lods: [RendererVoxelLOD; N],
}
```

Remember, this is what we already created when calling `VoxelMemoryGrid::new`.
Each LOD is a `RendererVoxelLOD`, which looks like:

```rust
pub struct RendererVoxelLOD {
    pub bitmask_buffers: DataComponent<DualBufferWithDynamicCopyRegions<VoxelBitmask>>,
    pub id_buffers: Option<DataComponent<DualBufferWithDynamicCopyRegions<VoxelTypeIDs>>>,
}
```

Here, we have separate buffers for the voxel IDs and bitmasks.
Note that these have a different type than before, `DualBufferWithDynamicCopyRegions`.
Before, we saw `DualBufferWithFullCopy`, which copied the full buffer every frame.
We definitely don't want to do that with all the voxel data, so instead we dynamically
copy only small sections of the voxel data to the GPU each frame.
We will see how these copy regions are tracked and passed to the renderer later.

### Instantiating `RendererComponents`

Now, let's create the components.

Remember that `material_list` only needs to be set once and never updated.
We need to do that around now.
In order to copy the data, we set up a one-time transfer:

```rust
let mut one_time_transfer_builder = standard_one_time_transfer_builder(&renderer_context);
```

Now, we can create all the components, and we'll pass in a reference to the transfer builder to `MaterialList::new` to queue up the transfer.

```rust
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
```

Finally, let's execute the transfer:

```rust
one_time_transfer_builder
    .build()
    .unwrap()
    .execute(Arc::clone(&renderer_context.transfer_queue))
    .unwrap()
    .then_signal_fence_and_flush()
    .unwrap()
    .wait(None)
    .unwrap();
```


## Creating the `Renderer`

With the data component set created, let's create the renderer:

```rust
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
```

## Using `renderer`

At the end of each frame, we need to update the staging buffers for the components that are not static.
That looks something like this:

```rust
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
```

Here, we first update the voxel data by calling `world.mem_grid.voxel.get_updates()`.
This is easy because `VoxelMemoryGrid` tracks the necessary updates for us.
This basically passes a bunch of copy regions that the renderer will directly use in a transfer pass to copy those regions of voxel data.

Then, we update the camera, time, and start TLC from the current values.

Finally, we call...

```rust
renderer.draw_frame();
```



# The game loop

We have covered most of the game loop by looking at how each component is used.
However, it's clearer to look at it all together.
I would now recommend reading through `main.rs` in `example_game` starting from `event_loop.run(...)`.
It is commented such that you can follow the code based on the background in this guide.

The game loop in `example_game` also includes code allowing the player to left click to remove the block
they're looking at or right click to place a block.
This uses `ox::ray::cast_ray`.



# Ray tracing

### What is path tracing?

Path tracing aims to directly simulate the real physics of light.

Light, in the real world, eminates from light sources and bounces off of surfaces.
Surfaces absorb some wavelengths of the light and reflect others, essentially changing the color
of the ray after the bounce.
When light enters our eyes, we sense its color and intensity.
This process allows us to see the color of objects as well as other visual characteristics based on
how the light bounces off of them, such as reflectivity.

We can simulate this by casting light rays around a scene.
Each ray will carry a color and intensity that is modified as it bounces.
Bounces are stochastic, so many rays must be cast to create a realistic simulation.

Unfortunately, simulating this whole process sufficiently is far too slow for real time rendering.
One thing we can do to make it more feasible is only simulate rays that actually end up
entering the eye/camera, since we never would see the result of the others anyway.
We do this by tracing the ray paths backwards, shooting them out from the eye and calculating their
bounces in reverse.
During the rendering process, rays are cast out from the camera in a grid.
Each one returns a color and becomes one pixel in the final rendered image.

### Ox's "raw" path tracing

Ox does "raw" or "naive" path tracing.
It does not currently do any postprocessing of the results or reuse the results across frames,
and it also does not intelligently sample bounce directions, they are completely random.
Basically all real time rendering techniques that use path tracing have significant postprocessing,
because the compute required to make raw path tracing give smooth output is exorbitant.
Similarly, many implementations intelligently sample ray bounce directions, such as those that will lead to
light sources.
Ox doesn't do this either.
Both of these are future directions to improve visual fidelity.
For now, the rendered output is somewhat noisy.

### Voxel meshing

The key difference between Ox and many other voxel rendering engines is that the voxels are **not** meshed.

Voxel data is naturally and most easily stored as a big array of IDs that represents, for each voxel slot in
3D space, which type of voxel (if any) is present there.
This is how Ox represents the voxel data both on the CPU and GPU.

This is in contrast to many rendering approaches for voxels, which take the raw voxel data and convert it to meshes.
Since 3D models are usually represnted as meshes, this can then be rendered like a "normal" 3D scene.
For example, this could be rasterized or ray traced using specialized hardware like NVIDIA RT cores.

Ox instead uses a custom algorithm to perform path tracing on the raw voxel data.
This approach, compared to meshing, has a number of tradeoffs:

Pros
 + Much less work to do on the CPU overall (no meshing)
 + Sparse updates are much faster
 + Simplicity

Cons
 - Data representation is much less compact in most cases
    - Much more VRAM usage
    - More time to transfer data CPU -> GPU
 - Can strictly only render voxels, nothing else. When meshing, you can also render other arbitrary meshes.

The path tracing is implemented in a compute shader.
This means that it does not use specialized hardware for ray tracing, such as NVIDIA's RT cores.
This the case because hardware ray tracing generally only works with a standard bounding volume hierarchy,
which is ... TODO
