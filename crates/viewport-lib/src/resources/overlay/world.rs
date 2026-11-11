//! The overlay shape, text and backdrop pipelines for world layers.
//!
//! A world layer's items are prepared exactly as screen overlays are, with a
//! "pixel" that is one world unit on the layer. The shaders differ in two
//! places only: the vertex stage maps a layer position through the camera's
//! view-projection (folded with the layer's basis) instead of the viewport's
//! pixels-to-NDC map, and the clip tests read the fragment's position on the
//! layer, in device pixels of the layer, as it is interpolated rather than
//! snapped to the screen's pixel centres. The world sources are made from the
//! screen ones by those substitutions, so the two can never drift.

/// The uniform a world layer binds where the screen overlays bind the
/// viewport size: the same first 16 bytes (`size`, `ppp`), then the map from
/// layer units to clip space.
const WORLD_VIEWPORT_WGSL: &str = "\
// A world layer's space: its anchor extent and the device pixels one layer
// unit covers (the screen viewport's fields), and the map from layer units to
// clip space through the scene camera, and the size in device pixels of the
// target the layer draws into.
struct WorldViewport {
    size:          vec2<f32>,
    ppp:           f32,
    // The layer's painter mask test (`MaskTest::word`).
    mask:          u32,
    layer_to_clip: mat4x4<f32>,
    target_px:     vec2<f32>,
    // The layer's painter order, the layer the mask test compares and the
    // light range is found by.
    order:         u32,
    // 1 when the layer samples its range's 2D light layer.
    lit:           u32,
};
";

/// The fragment wrapper a masked world variant ends with: the screen
/// shader's `shade`, cut by the layer's painter mask test and lit by its
/// range's 2D light layer, at the target's place under the fragment.
const WORLD_MASK_WGSL: &str = "
@group(2) @binding(0) var painter_mask_tex: texture_2d<f32>;
@group(2) @binding(1) var<uniform> painter_masks: PainterMasks;
@group(2) @binding(2) var painter_light_tex: texture_2d_array<f32>;
@group(2) @binding(3) var<uniform> painter_lights: PainterLights;

// The texel of a `dims`-sized texture over the target under the fragment.
fn world_target_texel(frag: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    let p = vec2<i32>(floor(frag / viewport.target_px * vec2<f32>(dims)));
    return clamp(p, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
}

fn shade(in: VertexOutput) -> vec4<f32> {
    let c = shade_unmasked(in);
    let texel = textureLoad(
        painter_mask_tex,
        world_target_texel(in.clip_position.xy, textureDimensions(painter_mask_tex)),
        0,
    );
    let keep = painter_mask_keep(texel, painter_masks, viewport.mask, viewport.order);
    var rgb = c.rgb;
    let slice = painter_light_slice(painter_lights, viewport.order);
    if viewport.lit != 0u && slice >= 0 {
        let light = textureLoad(
            painter_light_tex,
            world_target_texel(in.clip_position.xy, textureDimensions(painter_light_tex)),
            slice,
            0,
        );
        rgb = painter_light_apply(rgb, light, painter_lights);
    }
    return vec4<f32>(rgb, c.a * keep);
}
";

/// [`world_variant`] cut by the layer's painter mask: the screen `shade`
/// renamed and wrapped, with the mask bindings at group 2.
pub(crate) fn world_masked_variant(src: &str) -> Result<String, MissingLine> {
    let mut s = world_variant(src)?;
    let shade = "fn shade(in: VertexOutput) -> vec4<f32> {";
    if !s.contains(shade) {
        return Err(MissingLine(shade));
    }
    s = s.replace(shade, "fn shade_unmasked(in: VertexOutput) -> vec4<f32> {");
    s.push('\n');
    s.push_str(crate::plugin_api::shared_wgsl::SHARED_PAINTER_MASK_WGSL);
    s.push_str(WORLD_MASK_WGSL);
    Ok(s)
}

/// Bytes of the world viewport uniform.
pub(crate) const WORLD_VIEWPORT_BYTES: u64 = 96;

/// Why a world variant could not be made: the screen source no longer has a
/// line the substitution looks for.
#[derive(Debug)]
pub(crate) struct MissingLine(pub &'static str);

/// The world variant of an overlay shader source.
pub(crate) fn world_variant(src: &str) -> Result<String, MissingLine> {
    let mut s = src.to_string();

    // The viewport uniform: the world struct in place of the screen one.
    let decl = "var<uniform> viewport: Viewport;";
    let at = s.find(decl).ok_or(MissingLine(decl))?;
    let line_start = s[..at].rfind('\n').map_or(0, |i| i + 1);
    s.replace_range(at..at + decl.len(), "var<uniform> viewport: WorldViewport;");
    s.insert_str(line_start, WORLD_VIEWPORT_WGSL);

    // The vertex position through the camera. The screen shader's `frag_px`,
    // its position times `viewport.ppp`, is the position on the layer in
    // device pixels unchanged.
    let head = "out.clip_position = vec4<f32>(px_to_ndc(";
    let tail = "), 0.0, 1.0);";
    let a = s.find(head).ok_or(MissingLine(head))?;
    let b = a + s[a..].find(tail).ok_or(MissingLine(tail))?;
    let expr = s[a + head.len()..b].to_string();
    s.replace_range(
        a..b + tail.len(),
        &format!(
            "let world_px = {expr};\n    \
             out.clip_position = viewport.layer_to_clip * vec4<f32>(world_px, 0.0, 1.0);"
        ),
    );

    let fp = "let fp = overlay_pixel_centre(in.frag_px);";
    if !s.contains(fp) {
        return Err(MissingLine(fp));
    }
    s = s.replace(fp, "let fp = in.frag_px;");

    // The shape shader's edge band: one device pixel of the layer, where the
    // screen shader takes one logical pixel.
    s = s.replace(
        "1.0 / max(inst.aa_scale, 1.0e-6)",
        "1.0 / max(inst.aa_scale * viewport.ppp, 1.0e-6)",
    );
    Ok(s)
}

/// The world variant of the backdrop shader: [`world_variant`], with the
/// edge band one device pixel of the layer and the blurred destination read
/// at the fragment's place on the target rather than its layer position.
pub(crate) fn world_backdrop_variant(src: &str) -> Result<String, MissingLine> {
    let mut s = world_variant(src)?;
    let aa = "let aa = 1.0;";
    if !s.contains(aa) {
        return Err(MissingLine(aa));
    }
    s = s.replace(aa, "let aa = 1.0 / max(viewport.ppp, 1.0e-6);");
    let sample = "textureSample(t_backdrop, s_backdrop, in.uv)";
    if !s.contains(sample) {
        return Err(MissingLine(sample));
    }
    s = s.replace(
        sample,
        "textureSample(t_backdrop, s_backdrop, in.clip_position.xy / viewport.target_px)",
    );
    Ok(s)
}

/// Which overlay pipeline a world draw goes through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WorldPipe {
    Shape,
    Text,
    /// A backdrop-blur shape over the blurred destination.
    Backdrop,
}

/// The pass a world layer draws into: its colour format and sample count, and
/// whether the layer tests against scene depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct WorldPassKey {
    pub format: crate::gpu::TextureFormat,
    pub samples: u32,
    pub depth_test: bool,
}

/// The world variants of the shape and text shaders and the pipelines built
/// from them, one per pass, family and blend mode a frame has drawn with.
#[derive(Default)]
pub(crate) struct OverlayWorldResources {
    pub(crate) shape_shader: Option<crate::gpu::ShaderModule>,
    pub(crate) text_shader: Option<crate::gpu::ShaderModule>,
    backdrop_shader: Option<crate::gpu::ShaderModule>,
    backdrop_layout: Option<crate::gpu::PipelineLayout>,
    /// The screen shape and text layouts with the painter masks at group 2.
    shape_layout: Option<crate::gpu::PipelineLayout>,
    text_layout: Option<crate::gpu::PipelineLayout>,
    pipelines: Vec<(
        (WorldPassKey, WorldPipe, crate::renderer::OverlayBlend),
        crate::gpu::RenderPipeline,
    )>,
}

impl OverlayWorldResources {
    /// The pipeline for `pipe` with `blend` in the pass `key`, if built.
    pub(crate) fn pipeline(
        &self,
        key: WorldPassKey,
        pipe: WorldPipe,
        blend: crate::renderer::OverlayBlend,
    ) -> Option<&crate::gpu::RenderPipeline> {
        self.pipelines
            .iter()
            .find(|(k, _)| *k == (key, pipe, blend))
            .map(|(_, p)| p)
    }

    /// Drop the shaders and pipelines, so the next frame rebuilds them from
    /// the current fill registry.
    pub(crate) fn clear(&mut self) {
        self.shape_shader = None;
        self.text_shader = None;
        self.pipelines
            .retain(|((_, pipe, _), _)| *pipe == WorldPipe::Backdrop);
    }
}

impl crate::resources::DeviceResources {
    /// Build the world pipeline for `pipe` with `blend` in the pass `key` if
    /// it is not built yet. The screen pipeline of the same family is built
    /// first: the world one shares its layout, so the bind groups the overlay
    /// prepare builds serve both.
    pub(crate) fn ensure_overlay_world_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        key: WorldPassKey,
        pipe: WorldPipe,
        blend: crate::renderer::OverlayBlend,
    ) {
        if self.overlay_world.pipeline(key, pipe, blend).is_some() {
            return;
        }
        if pipe == WorldPipe::Backdrop {
            self.ensure_overlay_world_backdrop_pipeline(device, key);
            return;
        }
        let (layout, label, buffers): (_, _, Vec<crate::gpu::VertexBufferLayout<'static>>) =
            match pipe {
                WorldPipe::Shape => {
                    self.ensure_overlay_shape_pipeline(device);
                    let fill = self.overlay_fill_bgl(device).clone();
                    let layout = match (
                        &self.overlay_world.shape_layout,
                        &self.overlay_shape.shadow_bgl,
                    ) {
                        (Some(l), _) => Some(l.clone()),
                        (None, Some(shadow)) => Some(crate::resources::builders::pipeline_layout(
                            device,
                            "overlay_world_shape_layout",
                            &[shadow, &fill, &self.painter_mask.bgl],
                        )),
                        _ => None,
                    };
                    self.overlay_world.shape_layout = layout.clone();
                    (
                        layout,
                        "overlay_world_shape_pipeline",
                        vec![crate::resources::OverlayShapeInstance::buffer_layout()],
                    )
                }
                WorldPipe::Backdrop => unreachable!("built above"),
                WorldPipe::Text => {
                    self.ensure_overlay_text_pipeline(device);
                    let fill = self.overlay_fill_bgl(device).clone();
                    let layout = match (&self.overlay_world.text_layout, &self.overlay_text.bgl) {
                        (Some(l), _) => Some(l.clone()),
                        (None, Some(bgl)) => Some(crate::resources::builders::pipeline_layout(
                            device,
                            "overlay_world_text_layout",
                            &[bgl, &fill, &self.painter_mask.bgl],
                        )),
                        _ => None,
                    };
                    self.overlay_world.text_layout = layout.clone();
                    (
                        layout,
                        "overlay_world_text_pipeline",
                        vec![
                            crate::resources::OverlayTextVertex::buffer_layout(),
                            crate::resources::OverlayInstance::buffer_layout(
                                &crate::resources::overlay::geometry::TEXT_INSTANCE_ATTRIBUTES,
                            ),
                        ],
                    )
                }
            };
        let Some(layout) = layout else {
            return;
        };
        let shader = match pipe {
            WorldPipe::Shape => self.overlay_world.shape_shader.clone(),
            _ => self.overlay_world.text_shader.clone(),
        };
        let shader = match shader {
            Some(s) => s,
            None => {
                let name = match pipe {
                    WorldPipe::Shape => "overlay_shape",
                    _ => "overlay_text",
                };
                let src = self.overlay_fill_shader_source(name);
                let world = match world_masked_variant(&src) {
                    Ok(world) => world,
                    Err(MissingLine(line)) => {
                        tracing::warn!("{name}: no world variant, missing `{line}`");
                        return;
                    }
                };
                let module = crate::resources::builders::wgsl_module(
                    device,
                    match pipe {
                        WorldPipe::Shape => "overlay_world_shape.wgsl",
                        _ => "overlay_world_text.wgsl",
                    },
                    world,
                );
                match pipe {
                    WorldPipe::Shape => self.overlay_world.shape_shader = Some(module.clone()),
                    _ => self.overlay_world.text_shader = Some(module.clone()),
                }
                module
            }
        };
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let (blend_state, fs_entry) =
            crate::resources::overlay::overlay_shape::overlay_blend_target(blend);
        let compare = if key.depth_test {
            crate::plugin_api::depth::DEPTH_COMPARE
        } else {
            crate::gpu::CompareFunction::Always
        };
        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label,
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &buffers,
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    self.scene_depth_format(),
                    false,
                    compare,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: key.samples,
                    ..Default::default()
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs_entry),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: key.format,
                        blend: Some(blend_state),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                cache: None,
            },
        );
        self.overlay_world
            .pipelines
            .push(((key, pipe, blend), pipeline));
    }
}

impl crate::resources::DeviceResources {
    /// Build the world backdrop pipeline for the pass `key`: the screen
    /// backdrop pipeline's layout and blend over the world variant of its
    /// shader.
    fn ensure_overlay_world_backdrop_pipeline(
        &mut self,
        device: &crate::gpu::Device,
        key: WorldPassKey,
    ) {
        self.ensure_overlay_shape_tex_pipeline(device);
        let (Some(bgl), Some(clip_bgl)) = (
            self.overlay_shape.tex_bgl.as_ref(),
            self.overlay_shape.tex_clip_bgl.as_ref(),
        ) else {
            return;
        };
        let layout = self
            .overlay_world
            .backdrop_layout
            .get_or_insert_with(|| {
                crate::resources::builders::pipeline_layout(
                    device,
                    "overlay_world_backdrop_layout",
                    &[bgl, clip_bgl],
                )
            })
            .clone();
        let shader = match self.overlay_world.backdrop_shader.clone() {
            Some(s) => s,
            None => {
                let src = crate::resources::builders::wgsl_source!("overlay_shape_tex");
                let world = match world_backdrop_variant(src) {
                    Ok(world) => world,
                    Err(MissingLine(line)) => {
                        tracing::warn!("overlay_shape_tex: no world variant, missing `{line}`");
                        return;
                    }
                };
                let module = crate::resources::builders::wgsl_module(
                    device,
                    "overlay_world_backdrop.wgsl",
                    world,
                );
                self.overlay_world.backdrop_shader = Some(module.clone());
                module
            }
        };
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "overlay_world_backdrop_pipeline",
                layout: &layout,
                vertex_module: &shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[crate::resources::OverlayBackdropVertex::buffer_layout()],
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    self.scene_depth_format(),
                    false,
                    if key.depth_test {
                        crate::plugin_api::depth::DEPTH_COMPARE
                    } else {
                        crate::gpu::CompareFunction::Always
                    },
                )),
                multisample: crate::gpu::MultisampleState {
                    count: key.samples,
                    ..Default::default()
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: key.format,
                        blend: Some(crate::resources::overlay::overlay_shape::BACKDROP_BLEND),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                cache: None,
            },
        );
        self.overlay_world.pipelines.push((
            (
                key,
                WorldPipe::Backdrop,
                crate::renderer::OverlayBlend::SourceOver,
            ),
            pipeline,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources() -> [(&'static str, &'static str); 2] {
        use crate::resources::overlay::custom_fill::fill_shader;
        [
            ("overlay_shape", fill_shader("overlay_shape")),
            ("overlay_text", fill_shader("overlay_text")),
        ]
    }

    #[test]
    fn the_world_variants_are_made_and_validate() {
        for (name, src) in sources() {
            let world = world_variant(src).unwrap_or_else(|e| panic!("{name}: no line {:?}", e.0));
            assert!(world.contains("layer_to_clip *"), "{name}");
            assert!(world.contains("let fp = in.frag_px;"), "{name}");
            assert!(!world.contains("px_to_ndc(screen_px)"), "{name}");
            let module = naga::front::wgsl::parse_str(&world)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&world)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} world variant: {e:?}"));
        }
    }

    #[test]
    fn the_masked_world_variants_are_made_and_validate() {
        for (name, src) in sources() {
            let world =
                world_masked_variant(src).unwrap_or_else(|e| panic!("{name}: no line {:?}", e.0));
            assert!(world.contains("fn shade_unmasked("), "{name}");
            let module = naga::front::wgsl::parse_str(&world)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&world)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} masked world variant: {e:?}"));
        }
    }

    #[test]
    fn the_world_backdrop_variant_is_made_and_validates() {
        let src = include_str!(concat!(env!("OUT_DIR"), "/overlay_shape_tex.wgsl"));
        let world = world_backdrop_variant(src).unwrap_or_else(|e| panic!("no line {:?}", e.0));
        assert!(world.contains("in.clip_position.xy / viewport.target_px"));
        assert!(world.contains("let fp = in.frag_px;"));
        let module = naga::front::wgsl::parse_str(&world)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&world)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("backdrop world variant: {e:?}"));
    }

    #[test]
    fn the_world_viewport_matches_its_byte_size() {
        let module = naga::front::wgsl::parse_str(WORLD_VIEWPORT_WGSL).unwrap();
        let ty = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("WorldViewport"))
            .unwrap()
            .1;
        assert_eq!(ty.inner.size(module.to_ctx()) as u64, WORLD_VIEWPORT_BYTES);
    }
}
