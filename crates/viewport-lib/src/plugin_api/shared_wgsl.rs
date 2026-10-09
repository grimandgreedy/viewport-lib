//! WGSL helper catalog.
//!
//! Each helper is a `&'static str` of WGSL that a plugin shader prefixes (or
//! concatenates into its own shader source) to gain access to the lib's
//! shared bindings, shading helpers, and target conventions.
//!
//! Versioning: each helper carries a `// @viewport-wgsl-version: N`
//! comment. The version is bumped whenever a function signature, struct
//! field, or binding number changes. Plugins compare against
//! [`WGSL_VERSION`] at build time to detect breakage early. Function bodies
//! and private fields are not part of the contract and may change between
//! patch releases.
//!
//! Composition: plugin shaders typically build their source as:
//!
//! ```ignore
//! let src = format!(
//!     "{bindings}\n{pbr}\n{this_pipeline_specific_wgsl}",
//!     bindings = viewport_lib::plugin_api::shared_wgsl::SHARED_BINDINGS_WGSL,
//!     pbr      = viewport_lib::plugin_api::shared_wgsl::SHARED_PBR_WGSL,
//!     this_pipeline_specific_wgsl = include_str!("my_shader.wgsl"),
//! );
//! ```
//!
//! On wasm32, or with the `minify-shaders` feature, the helpers are embedded
//! with their comments and indentation removed, so match on statements rather than on
//! comments or leading whitespace if you rewrite one.

/// Catalog version. Bumped on any breaking change to a helper signature,
/// struct field, or binding number. Plugins should assert against this at
/// build time:
///
/// ```ignore
/// const _: () = assert!(viewport_lib::plugin_api::shared_wgsl::WGSL_VERSION == 1);
/// ```
pub const WGSL_VERSION: u32 = 6;

/// Group-0 bind declarations and shared scene-data structs.
///
/// Declares every binding in the lib's camera/lights/shadows/clip/IBL group,
/// matching the layout exposed via
/// [`SharedBindings`](super::SharedBindings). Plugin shaders include this
/// once and must not re-declare these bindings.
///
/// Bindings exposed:
///
/// | Binding | Resource | WGSL identifier |
/// |---------|----------|----------------|
/// | 0  | `Camera` uniform | `camera` |
/// | 1  | shadow atlas texture | `shadow_atlas_tex` |
/// | 2  | shadow comparison sampler | `shadow_atlas_sampler` |
/// | 3  | `Lights` header uniform | `lights` |
/// | 4  | `ClipPlanes` uniform | `clip_planes` |
/// | 5  | *(internal: CSM uniform; route through `viewport_sample_csm`)* | *(opaque)* |
/// | 6  | `ClipVolumes` uniform | `clip_volumes` |
/// | 7  | IBL irradiance equirect | `ibl_irradiance_tex` |
/// | 8  | IBL specular equirect | `ibl_specular_tex` |
/// | 9  | BRDF integration LUT | `ibl_brdf_lut` |
/// | 10 | IBL sampler | `ibl_sampler` |
/// | 11 | Skybox equirect | `skybox_tex` |
/// | 13 | per-light array | `lights_storage` |
/// | 17 | point-light shadow cubemap array | `point_shadow_cube` |
pub const SHARED_BINDINGS_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shared_bindings.wgsl"));

/// The clustered scene-light loop the built-in lit item types draw with.
///
/// Provides `apply_scene_lighting(normal, base_colour, two_sided, world_pos,
/// lights)`, the per-light `eval_light`, the cluster lookup, and the
/// environment-zone and probe-volume declarations they read. Unlike
/// [`SHARED_PBR_WGSL`]'s `viewport_apply_scene_lighting`, which is a
/// self-contained Lambert loop over `lights_storage`, this is the same code
/// path the renderer's own mesh and item shaders take: clustered iteration,
/// physical inverse-square falloff with a source-radius clamp, and light
/// channel masks. Compose it when an item has to match a built-in type's
/// shading exactly.
///
/// It brings its own group-0 declarations for the lighting bindings (13 to 18
/// and 20) along with the `SingleLight` and `Lights` structs, which overlap
/// [`SHARED_BINDINGS_WGSL`] on bindings 13 and 17. Compose one or the other,
/// not both: a body using this declares its own camera at binding 0 and the
/// `Lights` uniform at binding 3.
///
/// It applies no shadow term. Multiply by
/// [`viewport_sample_csm`](SHARED_PBR_WGSL) yourself if the item casts into
/// the cascade atlas.
pub const SHARED_SCENE_LIGHTING_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/scene_lighting.wgsl"));

/// The cascaded shadow-map sampler the renderer's own lit shaders use.
///
/// Provides:
///
/// ```ignore
/// fn sample_shadow_csm(world_pos: vec3<f32>, world_normal: vec3<f32>) -> ShadowSample;
/// ```
///
/// Compose it after a body that declares `camera`, `shadow_map`,
/// `shadow_sampler`, `shadow_atlas`, `lights_uniform` and `lights_storage`,
/// which is the binding layout [`SHARED_SCENE_LIGHTING_WGSL`] expects. It
/// declares no bindings of its own.
///
/// [`SHARED_PBR_WGSL`]'s `viewport_sample_csm` wraps the same cascade scheme
/// for a body composing [`SHARED_BINDINGS_WGSL`] instead, and returns only the
/// factor. Take this one when the item needs the full sample, or when it is
/// already composing the clustered lighting path.
pub const SHARED_CSM_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/helpers/csm.wgsl"));

/// The section-view clip-volume test, for a body that declares its own group-0
/// bindings.
///
/// Provides:
///
/// ```ignore
/// fn clip_volume_test(world_pos: vec3<f32>) -> bool;
/// ```
///
/// [`SHARED_BINDINGS_WGSL`] already carries the same test as
/// `viewport_pass_clip_volumes`, so compose this only when the body cannot
/// take the shared bindings: it reads the `clip_volume: ClipVolumeUB` uniform
/// the body declares at binding 6, and declares nothing itself.
pub const SHARED_CLIP_VOLUME_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/clip_volume_test.wgsl"));

/// The direct Cook-Torrance BRDF the renderer's lit mesh shaders evaluate.
///
/// Provides `D_GGX`, `G1_Smith` and the functions built on them, for a body
/// that shades with the scene's lights itself and wants its highlights to
/// match the surface it sits on. It declares no bindings and reads none, so
/// it composes after either set of group-0 declarations.
pub const SHARED_BRDF_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/helpers/brdf.wgsl"));

/// The fullscreen edge trace the selection outline uses: a complete shader,
/// vertex and fragment stage, that reads a single-channel coverage mask and
/// draws a ring around it.
///
/// For an item type that draws its own selection ring from a mask of its own,
/// so the ring matches the one the renderer draws for everything else. Group 0
/// is the mask texture at binding 0, a filtering sampler at binding 1 and an
/// [`OutlineEdgeUniform`](crate::resources::OutlineEdgeUniform) at binding 2;
/// the entry points are `vs_main` and `fs_main`.
pub const SHARED_OUTLINE_EDGE_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/outline_edge.wgsl"));

/// Shared PBR shading helper.
///
/// Provides:
///
/// ```ignore
/// fn viewport_pbr_shade(inp: PbrInputs) -> vec3<f32>;
/// fn viewport_sample_csm(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32;
/// fn viewport_apply_scene_lighting(N, base_colour, two_sided, world_pos) -> vec3<f32>;
/// ```
///
/// `viewport_pbr_shade` returns the final lit colour for a fragment given a
/// `PbrInputs` populated with albedo / normal / metallic / roughness / AO /
/// emissive. It applies the lib's standard hemisphere ambient + light loop
/// and attenuates the primary light's contribution by the CSM shadow factor
/// when `lights.shadows_enabled != 0`. Plugins that compose this helper get
/// shadows automatically; do not multiply by `viewport_sample_csm` again.
/// Future revisions may add IBL and SSAO sampling inside this function;
/// consumers should rebuild their shaders when the catalog version bumps to
/// pick up the upgrade.
///
/// `viewport_sample_csm` returns a 0..1 shadow factor for `world_pos`.
/// Returns 1.0 (fully lit) when shadows are disabled or the position is
/// outside every cascade. The cascade scheme, filter kernel, and bias
/// strategy are internal details and may change between catalog versions;
/// the function signature and return-value semantics are the contract.
///
/// `viewport_apply_scene_lighting` is the simpler Lambert helper used by
/// non-PBR pipelines (glyphs, tubes, ribbons). Use it when a plugin wants
/// scene-light parity with those built-in items.
pub const SHARED_PBR_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shared_pbr.wgsl"));

/// Fragment-output struct and packing helper for the OIT pass.
///
/// A transparent plugin fragment shader returns [`OitOutput`] from its
/// `fs_main`. Use `viewport_oit_pack(color_premul, alpha, view_z)` to
/// build the struct; the weight function matches the lib's built-in OIT
/// pipelines so plugin transparents composite consistently with native
/// transparents in the same pass.
pub const SHARED_OIT_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shared_oit.wgsl"));

/// Scene-depth reconstruction helpers for the read-only-depth pass.
///
/// A plugin drawing in [`paint_depth_read`](crate::plugin_api::ItemTypePlugin::paint_depth_read)
/// includes this after [`SHARED_BINDINGS_WGSL`] (it reads `camera` for the
/// inverse view-projection). The helpers are **binding-agnostic**: they take a
/// depth value the plugin already sampled, so the plugin declares the scene
/// depth texture + sampler in whatever bind group it owns and at whatever
/// index is free. This matters because a plugin already using all four bind
/// groups (the default `max_bind_groups`) has no spare group for a dedicated
/// depth binding: it appends the two depth bindings to an existing group
/// instead. The plugin builds that group's bind group from
/// [`DepthReadContext::scene_depth`](crate::plugin_api::DepthReadContext::scene_depth)
/// and [`scene_depth_sampler`](crate::plugin_api::DepthReadContext::scene_depth_sampler).
///
/// Declare the depth binding and sample it yourself, then pass the value in:
///
/// ```wgsl
/// // In a group the plugin owns, at any free binding index:
/// @group(2) @binding(4) var scene_depth_tex:  texture_depth_2d;
/// @group(2) @binding(5) var scene_depth_samp: sampler;
/// // ...
/// let screen_uv    = frag_pos.xy / viewport_size;
/// let scene_ndc_z  = textureSample(scene_depth_tex, scene_depth_samp, screen_uv);
/// let fade         = viewport_soft_fade_from_ndc(world_pos, screen_uv, scene_ndc_z, soft_dist);
/// ```
///
/// The reconstruction matches the built-in soft-particle sprite path exactly,
/// so plugin results agree with the `Soft` sprite sub-mode:
///
/// - `viewport_view_z(world_pos)` gives positive linear view-space depth
///   (distance in front of the camera) of a world point.
/// - `viewport_scene_view_z_from_ndc(screen_uv, scene_ndc_z)` reconstructs the
///   same for the sampled scene surface via `camera.inv_view_proj` then
///   `camera.view`.
/// - `viewport_soft_fade_from_ndc(world_pos, screen_uv, scene_ndc_z, soft_dist)`
///   returns a `0..1` fade that ramps to zero as `world_pos` approaches the
///   scene surface over `soft_dist` world units; `soft_dist <= 0` returns 1.
///
/// `screen_uv` is the fragment's `@builtin(position).xy` divided by the
/// viewport size in pixels (`clip_planes.viewport_width/height` from group 0,
/// or the value passed to the draw).
pub const SHARED_DEPTH_READ_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shared_depth_read.wgsl"));

/// Fragment helper for the outline mask pass.
///
/// A plugin's mask pipeline reuses its scene-pass vertex stage and uses
/// `fs_mask` (or any function returning `@location(0) f32 = 1.0`). The
/// composite reads any non-zero value as "this pixel belongs to a selected
/// item." Depth state must match the mask pass: depth test on, depth write
/// off.
pub const SHARED_MASK_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shared_mask.wgsl"));

/// Group-0 declarations for the shadow-cast pass.
///
/// The shadow pass binds its own camera at group 0 : a single dynamic-offset
/// uniform holding the cascade's light view-projection, not the scene bind
/// group every other pass uses. A shader for
/// [`cast_shadow_pass`](crate::plugin_api::ItemTypePlugin::cast_shadow_pass)
/// therefore prepends this instead of [`SHARED_BINDINGS_WGSL`], and a pipeline
/// built with
/// [`build_shadow_pipeline`](crate::resources::DeviceResources::build_shadow_pipeline)
/// matches it.
///
/// The pass is depth-only, so the shader needs a vertex stage and no fragment
/// stage: pass `""` as the fragment entry point.
pub const SHARED_SHADOW_BINDINGS_WGSL: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/helpers/shared_shadow_bindings.wgsl"
));

/// Fragment helper for the pick-id pass.
///
/// A plugin's pick pipeline reuses its scene-pass vertex stage (extended to
/// pass a flat-interpolated `pick_id: u32`) and uses `fs_pick`. The
/// renderer reads back the R32U pixel under the cursor to resolve which
/// item was clicked.
pub const SHARED_PICK_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shared_pick.wgsl"));

/// Fragment helper for the pick-id pass that reports the hit instance.
///
/// Same contract as [`SHARED_PICK_WGSL`]'s `viewport_pick_fs`, except the
/// primitive channel carries an instance index the vertex stage passes
/// through: declare `@builtin(instance_index)` in the vertex input and
/// forward it flat-interpolated at `@location(1)` of the fragment input.
/// Unlike [`SHARED_PICK_PRIM_WGSL`], this needs no device feature, matching
/// the built-in instanced pick path, so instanced plugin items stay
/// instance-pickable on every device. The renderer hands the read-back
/// index to the plugin's
/// [`resolve_sub_object`](crate::plugin_api::ItemTypePlugin::resolve_sub_object)
/// when the query mask holds `INSTANCE`.
pub const SHARED_PICK_INSTANCE_WGSL: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/helpers/shared_pick_instance.wgsl"
));

/// Fullscreen-triangle vertex stage for post-effect passes.
///
/// Prepend to a fragment-only post shader and build with
/// [`build_post_effect_pipeline`](crate::resources::DeviceResources::build_post_effect_pipeline):
/// the pipeline draws three vertices with no vertex buffer, and the
/// fragment stage receives `ViewportPostVsOut` with `uv` in [0, 1] (origin
/// top-left, matching texture sampling). Declare the fragment entry as
/// `fn fs_main(in: ViewportPostVsOut) -> @location(0) vec4<f32>`.
pub const POST_EFFECT_VS_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/post_effect_vs.wgsl"));

/// Module directive required by [`SHARED_PICK_PRIM_WGSL`], per wgpu leg.
///
/// naga 29 only accepts `@builtin(primitive_index)` in a module that starts
/// with `enable primitive_index;`; naga 27 rejects the directive. Prepend this
/// constant at the very top of the shader source (before any declaration, so
/// ahead of the vertex stage too) and it resolves to the right text for the
/// active leg.
#[cfg(wgpu27)]
pub const PICK_PRIM_ENABLE_WGSL: &str = "";
/// Module directive required by [`SHARED_PICK_PRIM_WGSL`], per wgpu leg.
///
/// naga 29 only accepts `@builtin(primitive_index)` in a module that starts
/// with `enable primitive_index;`; naga 27 rejects the directive. Prepend this
/// constant at the very top of the shader source (before any declaration, so
/// ahead of the vertex stage too) and it resolves to the right text for the
/// active leg.
#[cfg(any(wgpu29, wgpu30))]
pub const PICK_PRIM_ENABLE_WGSL: &str = "enable primitive_index;\n";

/// Module directive a shader declaring a `binding_array` needs, per wgpu leg.
///
/// naga 30 only accepts `binding_array<...>` in a module that starts with
/// `enable wgpu_binding_array;`; naga 27 and 29 have no such extension and
/// reject the directive. Prepend this constant at the very top of the shader
/// source, before any declaration, and it resolves to the right text for the
/// active leg.
#[cfg(any(wgpu27, wgpu29))]
pub const BINDING_ARRAY_ENABLE_WGSL: &str = "";
/// Module directive a shader declaring a `binding_array` needs, per wgpu leg.
///
/// naga 30 only accepts `binding_array<...>` in a module that starts with
/// `enable wgpu_binding_array;`; naga 27 and 29 have no such extension and
/// reject the directive. Prepend this constant at the very top of the shader
/// source, before any declaration, and it resolves to the right text for the
/// active leg.
#[cfg(wgpu30)]
pub const BINDING_ARRAY_ENABLE_WGSL: &str = "enable wgpu_binding_array;\n";

/// Fragment helper for the pick-id pass that reports the hit triangle.
///
/// Same contract as [`SHARED_PICK_WGSL`]'s `viewport_pick_fs`, except the
/// primitive channel carries `@builtin(primitive_index)` : the index of the
/// rasterised triangle within the draw call : instead of `0u`. With this, a
/// hit on the item can be refined to a sub-object: the renderer hands the
/// read-back index to the plugin's
/// [`resolve_sub_object`](crate::plugin_api::ItemTypePlugin::resolve_sub_object).
///
/// Requirements:
/// - The device must have
///   [`PRIMITIVE_INDEX_FEATURE`](crate::gpu::PRIMITIVE_INDEX_FEATURE); a
///   module using the builtin fails validation without it. Check
///   `device.features()` at `init_gpu` and fall back to `viewport_pick_fs`
///   (picking then stays object-level, matching the built-in surfaces).
/// - The module source must start with [`PICK_PRIM_ENABLE_WGSL`], before any
///   other code:
///   `format!("{PICK_PRIM_ENABLE_WGSL}{MY_VS}{SHARED_PICK_PRIM_WGSL}")`.
pub const SHARED_PICK_PRIM_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shared_pick_prim.wgsl"));
