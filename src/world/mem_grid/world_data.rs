use paste::paste;


#[macro_export]
macro_rules! split_layers_into_chunks {
    ($grid: ident, $meta_builder: ident, $grid_meta: ident, $layer: ident, $value_name: ident) => {
        for (chunk_i, chunk) in self.$layer.chunks.into_iter().enumerate() {
            // If this layer is smaller than full grid, add padding to virtual position so it
            // is centered
            let virtual_pos = self.$layer.meta.virtual_grid_pos_for_index(chunk_i, $grid_meta.size).0
            $grid[pos_index(virtual_pos, $grid_meta.size-1)].data.$value_name = chunk;
        }
        $meta_builder.$layer = self.$layer.meta;
    }
}


#[macro_export]
macro_rules! world_data {
    (
        struct_name: $struct_name:ident,
        layers_struct_name: $layers_struct_name:ident,
        layers: {
            ($layer_name:ident: { data_struct: $layer_data_type:ty, size: $layer_size:expr },),*
        }
    ) => {
        paste! {
            struct $structName {
                $($layer_name: Option<$layer_data_type>,)*
            }

            impl MemoryGridChunkData for $struct_name {
                fn new_empty() -> Self {
                    $struct_name {
                        $($layer_name: None)*
                    }
                }

                fn new_blank(chunk_size: usize) -> Self {
                    $struct_name {
                        $($layer_name: $layer_data_type::new_for_chunk_size(chunk_size))*
                    }
                }
            }

            struct $layers_struct_name {
                $([<$layer_name _layer>]: MemoryGridLayer<$layer_data_type>,)*
            }

            struct [<$layers_struct_name Metadata>] {
                $([<$layer_name _layer>]: MemoryGridLayerMetadata,)*
            }

            struct [<$layers_struct_name MetadataBuilder>] {
                $([<$layer_name _layer>]: Option<MemoryGridLayerMetadata>,)*
            }

            impl [<$layers_struct_name MetadataBuilder>] {
                fn new() -> Self {
                    WorldDataLayerMetadataBuilder { velocity_layer: None, power_layer: None }
                }

                fn build(self) -> [<$layers_struct_name Metadata>] {
                    Self {
                        $([<$layer_name _layer>]: self.[<$layer_name _layer>].unwrap(),)*
                    }
                }
            }

            impl MemoryGridDataLayers for $layers_struct_name {
                type ChunkData = $struct_name;
                type LayersMetadata = [<$layers_struct_name Metadata>];

                fn new(start_tlc: TLCPos<i64>) -> $layers_struct_name {
                    $layers_struct_name {
                        $([<$layer_name _layer>]: MemoryGridLayer::new_raw(
                            $layer_data_type::new($),  // TODO: fix, should be chunk size
                            MemoryGridLayerMetadata::new(start_tlc, $layer_size, true),
                        ),)*
                    }
                }

                fn to_virtual_grid_format(self, &grid_meta: MemoryGridMetadata) -> (Vec<Self::ChunkData>, Self::LayersMetadata) {
                    let mut grid = vec![
                        Self::ChunkData::new_empty(),
                        (grid_meta.grid_size).pow(3)
                    ];
                    let mut meta = [<$layers_struct_name MetadataBuilder>]::new();

                    $(split_layers_into_chunks!(grid, meta, grid_meta, [<$layer_name _layer>], $layer_name);)*

                    (grid, meta.build())
                }

                fn from_virtual(chunks: Vec<Self::ChunkData>, data_layer_meta: Self::LayersMetadata, grid_meta: &MemoryGridMetadata) -> Self {
                    $(let mut [<$layer_name _layer_chunks>] = vec![None; data_layer_meta.[<$layer_name _layer>].meta.size];)*

                    for (i, chunk) in chunks.enumerate() {
                        $([<$layer_name _layer_chunks>][
                            data_layer_meta.[<$layer_name _layer>].index_for_virtual_grid_pos(
                                TLCVector(pos_for_index(i, grid_meta.size).0),
                                grid_meta.size,
                            )
                        ] = chunk.$layer_name;)*
                    }

                    $layers_struct_name {
                        $([<$layer_name _layer>]: MemoryGridLayer::new_raw([<$layer_name _layer_chunks>], data_layer_meta.[<$layer_name _layer>]),)*
                    }
                }
            }
        }
    }
}