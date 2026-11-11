//! The tilemap pipelines: cutout and blended colour variants for each pass a
//! layer draws in, and the pick pipelines. Each is built the first time a draw
//! needs it.

use viewport_lib::plugin_api::builders::DualPipelineDesc;
use viewport_lib::resources::DeviceResources;

/// Cutout layer in the LDR scene pass.
pub(crate) const CUTOUT_LDR: usize = 0;
/// Cutout layer in the HDR scene pass, with the indirect fraction.
pub(crate) const CUTOUT_SCENE: usize = 1;
/// Blended layer testing depth, LDR scene pass.
pub(crate) const BLEND_TEST_LDR: usize = 2;
/// Blended layer drawn over everything, LDR scene pass.
pub(crate) const BLEND_OFF_LDR: usize = 3;
/// Blended layer testing depth, HDR read-only-depth pass.
pub(crate) const BLEND_TEST_DEPTH_READ: usize = 4;
/// Blended layer drawn over everything, HDR read-only-depth pass.
pub(crate) const BLEND_OFF_DEPTH_READ: usize = 5;
/// Pick for cutout layers.
pub(crate) const PICK: usize = 6;
/// Pick for blended layers, drawn in the painter sequence with `LessEqual` and
/// no depth write, so the later of two overlapping draws wins, as on screen.
pub(crate) const PICK_PAINTER: usize = 7;
/// Pick for blended layers that ignore scene depth.
pub(crate) const PICK_PAINTER_NO_TEST: usize = 8;

const COUNT: usize = 9;

/// The instance stream: cell, tile, flags and tint per occupied cell.
const TILE_ATTRS: [viewport_lib::gpu::VertexAttribute; 4] = [
    viewport_lib::gpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: viewport_lib::gpu::VertexFormat::Uint32,
    },
    viewport_lib::gpu::VertexAttribute {
        offset: 4,
        shader_location: 1,
        format: viewport_lib::gpu::VertexFormat::Uint32,
    },
    viewport_lib::gpu::VertexAttribute {
        offset: 8,
        shader_location: 2,
        format: viewport_lib::gpu::VertexFormat::Uint32,
    },
    viewport_lib::gpu::VertexAttribute {
        offset: 12,
        shader_location: 3,
        format: viewport_lib::gpu::VertexFormat::Unorm8x4,
    },
];

fn tile_layout() -> viewport_lib::gpu::VertexBufferLayout<'static> {
    viewport_lib::gpu::VertexBufferLayout {
        array_stride: super::store::INSTANCE_STRIDE,
        step_mode: viewport_lib::gpu::VertexStepMode::Instance,
        attributes: &TILE_ATTRS,
    }
}

/// What a tilemap pipeline build reads.
pub(crate) struct TilemapRecipe {
    device: viewport_lib::gpu::Device,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    pick_shader: viewport_lib::plugin_api::LazyModule,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
    depth_format: viewport_lib::gpu::TextureFormat,
}

pub(crate) type TilemapPipelines = viewport_lib::plugin_api::LazyPipelines<TilemapRecipe, COUNT>;

fn build(r: &TilemapRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    use viewport_lib::gpu;
    let vertex_buffers = [tile_layout()];
    if matches!(i, PICK | PICK_PAINTER | PICK_PAINTER_NO_TEST) {
        let id_target = Some(gpu::ColorTargetState {
            format: gpu::TextureFormat::R32Uint,
            blend: None,
            write_mask: gpu::ColorWrites::ALL,
        });
        let shader = r.pick_shader.get();
        return viewport_lib::plugin_api::builders::render_pipeline(
            &r.device,
            viewport_lib::plugin_api::builders::RenderPipelineDesc {
                label: "tilemap_pick_pipeline",
                layout: &r.layout,
                vertex_module: shader,
                vertex_entry: "vs_main",
                vertex_buffers: &vertex_buffers,
                fragment: Some(gpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[id_target.clone(), id_target.clone(), id_target],
                    compilation_options: gpu::PipelineCompilationOptions::default(),
                }),
                primitive: gpu::PrimitiveState {
                    topology: gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                // Painter picks test depth without writing it, so the
                // sequence's later draw wins wherever two overlap.
                depth_stencil: Some(viewport_lib::plugin_api::builders::scene_depth_stencil(
                    r.depth_format,
                    i == PICK,
                    match i {
                        PICK => viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT,
                        PICK_PAINTER => viewport_lib::plugin_api::depth::DEPTH_COMPARE,
                        _ => gpu::CompareFunction::Always,
                    },
                )),
                multisample: gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );
    }
    let cutout = matches!(i, CUTOUT_LDR | CUTOUT_SCENE);
    let depth_test = matches!(
        i,
        CUTOUT_LDR | CUTOUT_SCENE | BLEND_TEST_LDR | BLEND_TEST_DEPTH_READ
    );
    let depth_read = matches!(i, BLEND_TEST_DEPTH_READ | BLEND_OFF_DEPTH_READ);
    viewport_lib::plugin_api::builders::build_dual_pipeline_variant(
        &r.device,
        &DualPipelineDesc {
            label: if cutout {
                "tilemap_cutout_pipeline"
            } else {
                "tilemap_blended_pipeline"
            },
            layout: &r.layout,
            shader: r.shader.get(),
            vertex_entry: "vs_main",
            fragment_entry: if cutout { "fs_cutout" } else { "fs_blended" },
            vertex_buffers: &vertex_buffers,
            blend: (!cutout).then_some(gpu::BlendState::ALPHA_BLENDING),
            topology: gpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            depth_write: cutout,
            depth_compare: if depth_test {
                viewport_lib::plugin_api::depth::DEPTH_COMPARE_STRICT
            } else {
                gpu::CompareFunction::Always
            },
            sample_count: r.sample_count,
            // The read-only-depth pass has the HDR colour as its one target.
            ldr_format: if depth_read {
                viewport_lib::resources::HDR_COLOR_FORMAT
            } else {
                r.ldr_format
            },
            depth_format: r.depth_format,
            indirect_fraction_entry: Some("fs_cutout_scene"),
        },
        i == CUTOUT_SCENE,
    )
}

/// The tilemap's pipelines and layouts, made on the first frame that draws a
/// layer.
pub(crate) struct TilemapGpu {
    pub(crate) pipelines: TilemapPipelines,
}

impl TilemapGpu {
    pub(crate) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::TilemapLayouts,
    ) -> Self {
        let common = super::TILEMAP_COMMON;
        let shader = resources.lazy_module(
            device,
            "tilemap_shader",
            crate::item_types::shader::scene_shader(
                &[common],
                crate::item_types::shader::wgsl_source!("tilemap"),
            ),
        );
        let pick_shader = resources.lazy_module(
            device,
            "tilemap_pick_shader",
            crate::item_types::shader::scene_shader(
                &[common],
                crate::item_types::shader::wgsl_source!("tilemap_pick"),
            ),
        );
        let layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "tilemap_pipeline_layout",
            &[resources.group0_layout(), &layouts.map, &layouts.layer],
        );
        let pipelines = resources.lazy_pipelines(
            TilemapRecipe {
                device: device.clone(),
                layout,
                shader,
                pick_shader,
                sample_count: resources.sample_count(),
                ldr_format: resources.target_format(),
                depth_format: resources.scene_depth_format(),
            },
            build,
        );
        Self { pipelines }
    }
}
