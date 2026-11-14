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
pub const WGSL_VERSION: u32 = 12;

/// Group-0 bind declarations and shared scene-data structs.
///
/// Declares every binding in the lib's camera/lights/shadows/clip/IBL group,
/// matching
/// [`PluginResources::group0_layout`](super::PluginResources::group0_layout).
/// Plugin shaders include this once and must not re-declare these bindings.
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
/// | 6  | `ClipVolumeUB` uniform | `clip_volume` |
/// | 7  | IBL irradiance equirect | `ibl_irradiance_tex` |
/// | 8  | IBL specular equirect | `ibl_specular_tex` |
/// | 9  | BRDF integration LUT | `ibl_brdf_lut` |
/// | 10 | IBL sampler | `ibl_sampler` |
/// | 11 | Skybox equirect | `skybox_tex` |
/// | 13 | per-light array | `lights_storage` |
/// | 17 | point and spot light shadow array (`texture_depth_2d_array`) | `punctual_shadow_tex` |
///
/// Positions in group 0 are relative to the renderer's world origin (see
/// [`WorldOrigin`](crate::WorldOrigin)): `camera.view_proj`, `camera.eye_pos`,
/// the light positions and the clip planes all are, and a vertex shader
/// multiplies a model matrix whose translation the renderer has already made
/// relative. `camera.world_origin` holds the origin rounded to f32 and
/// `camera.origin_epoch` counts its moves. `viewport_absolute_pos(rel)` adds
/// the origin back for procedural effects that key on world position; it is
/// lossy far from zero, so never use it to place geometry.
///
/// `camera.exposure` is the camera's exposure, the linear multiplier every
/// radiance source takes where a shader reads it: a light's colour times its
/// intensity, the environment, the hemisphere fill, an emissive term. The
/// helpers in [`SHARED_PBR_WGSL`] apply it; a shader that reads `lights` or
/// the IBL textures itself multiplies by it. A colour the item draws without
/// lighting does not take it.
pub const SHARED_BINDINGS_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shared_bindings.wgsl"));

/// The clustered scene-light loop the built-in lit item types draw with.
///
/// Provides `apply_scene_lighting(normal, base_colour, two_sided, world_pos,
/// lights)`, the per-light `eval_light`, the cluster lookup, and the
/// environment-zone and probe-volume declarations they read. A body with its
/// own light loop walks the lights reaching a fragment with the cluster
/// cursor:
///
/// ```wgsl
/// var lights = cluster_lights(world_pos, lights_uniform.count);
/// loop {
///     let i = cluster_next_light(&lights);
///     if i == CLUSTER_LIGHTS_END { break; }
///     let ev = eval_light(lights_storage[i], world_pos);
///     // ...
/// }
/// ``` Unlike
/// [`SHARED_PBR_WGSL`]'s `viewport_apply_scene_lighting`, which is a
/// self-contained Lambert loop over every entry of `lights_storage`, this is
/// the same code path the renderer's own mesh and item shaders take:
/// clustered iteration and light channel masks. Both use the same
/// inverse-square falloff with a source-radius clamp. Compose it when an item
/// has to match a built-in type's shading exactly.
///
/// It brings its own group-0 declarations for the lighting bindings (13 to 18
/// and 20) along with the `SingleLight` and `Lights` structs, which overlap
/// [`SHARED_BINDINGS_WGSL`] on bindings 13 and 17. Compose one or the other,
/// not both: a body using this declares its own camera at binding 0 and the
/// `Lights` uniform at binding 3.
///
/// The camera it declares must carry the uniform's layout as far as
/// `exposure` (`view_proj`, `eye_pos`, a `f32`, `forward`, `exposure`), named
/// `camera`: `eval_light` and `apply_scene_lighting` return exposed radiance,
/// multiplied by `camera.exposure`. A body that reads another radiance source
/// itself (the hemisphere colours, an emissive term) multiplies it the same
/// way.
///
/// It applies no shadow term. Multiply by
/// [`viewport_sample_csm`](SHARED_PBR_WGSL) yourself if the item casts into
/// the cascade atlas, or compose [`SHARED_CSM_WGSL`] and
/// [`SHARED_PUNCTUAL_SHADOW_WGSL`] for the per-light shadow the mesh shaders
/// take.
pub const SHARED_SCENE_LIGHTING_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/scene_lighting.wgsl"));

/// The HDR scene pass's fragment output and the indirect fraction.
///
/// Provides:
///
/// ```ignore
/// struct SceneFragment {
///     @location(0) colour: vec4<f32>,
///     @location(1) indirect_fraction: vec4<f32>,
/// };
/// fn scene_light_split(indirect: vec3<f32>, direct: vec3<f32>, emitted: vec3<f32>);
/// fn scene_fragment(colour: vec4<f32>) -> SceneFragment;
/// var<private> scene_indirect_fraction: f32; // 0 unless the body sets it
/// var<private> scene_direct_fraction: f32;   // 0 unless the body sets it
/// ```
///
/// The HDR scene pass has a second colour target holding how each pixel's
/// light splits: the indirect share, which the composite scales by ambient
/// occlusion, and the direct share, which it scales by contact shadows. A lit
/// body calls `scene_light_split(ambient, direct, emitted)` with exposed
/// radiance in each term, and the fragment entry its HDR pipeline uses
/// returns `scene_fragment(colour)`. An unlit body sets nothing, leaving both
/// shares at 0, so neither term darkens it. The LDR pass has one
/// target, so the LDR pipeline keeps a separate entry returning the colour.
/// [`scene_lighting_terms`](SHARED_SCENE_LIGHTING_WGSL) gives the two terms
/// of the shared light loop. Declares no bindings.
pub const SHARED_SCENE_FRAGMENT_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/scene_fragment.wgsl"));

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

/// Point and spot light shadows, and the per-light shadow the renderer's own
/// lit shaders take.
///
/// Provides:
///
/// ```ignore
/// fn sample_punctual_shadow(light: SingleLight, world_pos: vec3<f32>, geo_normal: vec3<f32>) -> f32;
/// fn shadow_for_light(
///     light: SingleLight,
///     world_pos: vec3<f32>,
///     eye_pos: vec3<f32>,
///     shadow_normal: vec3<f32>,
///     geo_normal: vec3<f32>,
///     light_dir: vec3<f32>,
///     receiver_two_sided: u32,
/// ) -> ShadowSample;
/// ```
///
/// `shadow_for_light` picks by the light's `shadow_kind`: the cascades for the
/// directional light that casts them, the punctual shadow array at binding 17
/// for a point or spot light that was given layers this frame, and 1.0 for
/// any other light. Call it once per light in the shading loop.
///
/// Compose it after [`SHARED_CSM_WGSL`], in a body that declares the same
/// bindings that one reads plus `punctual_shadow_tex` at group 0 binding 17,
/// which [`SHARED_SCENE_LIGHTING_WGSL`] declares. A body that never declares
/// binding 17 leaves this out and keeps directional shadows only.
pub const SHARED_PUNCTUAL_SHADOW_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/punctual_shadow.wgsl"));

/// The `ShadowAtlas` struct: the cascade shadow uniform at group 0 binding 5.
///
/// For a body that declares its own group-0 bindings and binds
/// `shadow_atlas: ShadowAtlas` for [`SHARED_CSM_WGSL`]. It declares the struct
/// only, no binding, so compose it once next to the body that binds it.
/// [`SHARED_BINDINGS_WGSL`] users do not need it.
pub const SHARED_SHADOW_ATLAS_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shadow_atlas.wgsl"));

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

/// The section-view clip uniforms and tests, for a body that declares its own
/// group-0 bindings.
///
/// Declares `clip_planes: ClipPlanes` at group 0 binding 4 and
/// `clip_volume: ClipVolumeUB` at binding 6, and provides:
///
/// ```ignore
/// fn viewport_pass_clip_planes(world_pos: vec3<f32>) -> bool;
/// fn viewport_pass_clip_volumes(world_pos: vec3<f32>) -> bool;
/// fn viewport_clip_test(world_pos: vec3<f32>) -> bool;
/// ```
///
/// [`SHARED_BINDINGS_WGSL`] already includes all of this. Compose it instead
/// when the body cannot take the shared bindings, which is a body lit through
/// [`SHARED_SCENE_LIGHTING_WGSL`]: the two declare the light list on the same
/// binding. Never compose both.
pub const SHARED_CLIP_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/clip_bindings.wgsl"));

/// The scene depth convention for shaders, matching [`crate::plugin_api::depth`].
///
/// Scene depth is reversed: 1 at the near plane and 0 at the far plane, which
/// for a perspective camera is at infinity. The depth buffer clears to 0.
///
/// Provides:
///
/// ```ignore
/// const VIEWPORT_DEPTH_NEAR: f32; // 1.0
/// const VIEWPORT_DEPTH_FAR: f32;  // 0.0
/// fn viewport_depth_is_background(d: f32) -> bool;
/// fn viewport_depth_nearer(a: f32, b: f32) -> bool;
/// fn viewport_depth_nearest(a: f32, b: f32) -> f32;
/// fn viewport_depth_farthest(a: f32, b: f32) -> f32;
/// fn viewport_depth_toward_near(d: f32, amount: f32) -> f32;
/// fn viewport_depth_from_near(d: f32) -> f32;
/// fn viewport_depth_to_view_z(d: f32, clip: vec4<f32>) -> f32;
/// fn viewport_ndc_ray(inv_view_proj: mat4x4<f32>, ndc_xy: vec2<f32>) -> ViewportRay;
/// ```
///
/// Use these rather than literal `1.0` / `0.0` depth values or `min` / `max`
/// over depths, so a shader follows the convention the renderer draws with.
/// `viewport_depth_is_background` tests a value read from the scene depth
/// buffer for the clear value. A shader writing `@builtin(frag_depth)` writes
/// a value in the same convention; nudge it toward the camera with
/// `viewport_depth_toward_near`. `viewport_depth_from_near` gives a 0..1
/// value that grows away from the camera, for weights written against
/// standard depth.
///
/// `viewport_depth_to_view_z` turns a stored depth into positive view-space
/// distance; pass `camera.clip` from [`SHARED_BINDINGS_WGSL`], whose lanes are
/// (near, far, is_ortho, pixels per point) with far 0 for a projection with
/// no far plane.
/// `viewport_ndc_ray` builds the ray through a screen point from the near
/// plane; do not unproject depth 1 and 0 and divide by `w`, since the far
/// point is at infinity. `viewport_view_z` and
/// `viewport_scene_view_z_from_ndc` in [`SHARED_DEPTH_READ_WGSL`] go through
/// the camera's matrices and work under any convention. It declares no
/// bindings, so it composes anywhere.
pub const SHARED_DEPTH_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/helpers/depth.wgsl"));

/// Read one texel of a depth or stencil texture on every backend, GL included.
///
/// Provides:
///
/// ```ignore
/// fn depth_texel(tex: texture_2d<f32>, samp: sampler, coord: vec2<i32>) -> f32;
/// fn stencil_texel(tex: texture_2d<u32>, samp: sampler, coord: vec2<i32>) -> u32;
/// ```
///
/// Bind the depth view as `texture_2d<f32>` with an unfilterable-float layout
/// entry ([`depth_texel_entry`](crate::plugin_api::builders::depth_texel_entry))
/// and a non-filtering sampler
/// ([`nearest_sampler_entry`](crate::plugin_api::builders::nearest_sampler_entry),
/// [`clamp_nearest_sampler`](crate::plugin_api::builders::clamp_nearest_sampler)).
/// Do not `textureLoad` a `texture_depth_2d` (wgpu's GL backend cannot
/// translate it) or sample one without a comparison (the GLSL it becomes does
/// not compile). A `texture_2d<f32>` read with `textureLoad` reads 0 on GLES.
/// It declares no bindings, so it composes anywhere.
pub const SHARED_DEPTH_TEXEL_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/depth_texel.wgsl"));

/// The direct Cook-Torrance BRDF the renderer's lit mesh shaders evaluate.
///
/// Provides `viewport_brdf_d_ggx`, `viewport_brdf_g1_smith` and the functions
/// built on them (`viewport_brdf_light_contrib` evaluates one light), for a
/// body that shades with the scene's lights itself and wants its highlights to
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
/// fn viewport_ibl_ambient(n, v, albedo, metallic, roughness, f0, ao) -> vec3<f32>;
/// ```
///
/// `viewport_pbr_shade` returns the final lit colour for a fragment given a
/// `PbrInputs` populated with albedo / normal / metallic / roughness / AO /
/// emissive. Each light goes through the same attenuation and direct BRDF as
/// the lit mesh shaders; the ambient is `viewport_ibl_ambient` when an environment is
/// active and the hemisphere ambient otherwise. It attenuates the primary
/// light's contribution by the CSM shadow factor when
/// `lights.shadows_enabled != 0`. Plugins that compose this helper get
/// shadows automatically; do not multiply by `viewport_sample_csm` again.
///
/// The section carries [`SHARED_BRDF_WGSL`] and [`SHARED_SHADOW_ATLAS_WGSL`]
/// itself, so do not compose either of those next to it.
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
///
/// Every helper here returns exposed radiance: lights, the environment, the
/// hemisphere fill and `PbrInputs::emissive` are multiplied by
/// `camera.exposure`.
pub const SHARED_PBR_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/helpers/shared_pbr.wgsl"));

/// Fragment-output struct and packing helper for the OIT pass.
///
/// A transparent plugin fragment shader returns `OitOutput` from its
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
/// // In a group the plugin owns, at any free binding index. The depth view is
/// // bound as unfilterable float, with a non-filtering sampler.
/// @group(2) @binding(4) var scene_depth_tex:  texture_2d<f32>;
/// @group(2) @binding(5) var scene_depth_samp: sampler;
/// // ...
/// let screen_uv    = frag_pos.xy / viewport_size;
/// let scene_ndc_z  = textureSampleLevel(scene_depth_tex, scene_depth_samp, screen_uv, 0.0).r;
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

/// The painter mask test, for painter content that honours a
/// [`MaskTest`](crate::MaskTest).
///
/// Declares the `PainterMasks` struct and two functions, and no bindings. The
/// drawing shader declares the mask texture (`texture_2d<f32>`) and a
/// `var<uniform>` of `PainterMasks` in a group it owns, laid out as
/// [`painter_mask_bind_group_layout`](crate::resources::DeviceResources::painter_mask_bind_group_layout)
/// (or folded into the depth-read group at bindings 2 and 3, see
/// [`depth_read_bind_group_layout`](crate::resources::DeviceResources::depth_read_bind_group_layout)),
/// and binds [`PainterContext::mask_bind_group`](crate::plugin_api::PainterContext::mask_bind_group)
/// there.
///
/// - `painter_mask_texel(ndc, dims)` is the texel under a fragment at `ndc`
///   (normalised device coordinates) of a mask texture `dims` texels across.
/// - `painter_mask_keep(texel, masks, test, layer)` is the fraction of the
///   fragment a test keeps: `1.0` for no test, a slot no mask drew into or a
///   layer outside the slot's range, else the slot's coverage (visible
///   inside) or one minus it (visible outside). Multiply alpha by it.
///
/// `test` is [`MaskTest::word`](crate::MaskTest::word) and `layer` the
/// content's painter layer.
///
/// It also declares the `PainterLights` struct for 2D lighting, whose uniform
/// rides beside the masks with the light layers (a `texture_2d_array<f32>`):
///
/// - `painter_light_slice(lights, layer)` is the array layer whose light
///   lights painter `layer`, or `-1` when none does.
/// - `painter_light_apply(rgb, light, lights)` is `rgb` lit by the texel
///   `light` loaded from that array layer under the fragment (at
///   `painter_mask_texel` of the light layers' size).
///
/// - `painter_light_dir_slice(lights, layer)` is the array layer holding the
///   light directions for painter `layer`, or `-1` when its range is lit
///   flat.
/// - `painter_light_apply_normal(rgb, light, dir, n, lights)` is `rgb` with
///   the world-space normal `n` lit by `light` and the direction texel `dir`
///   loaded from that array layer. Use it for content with a normal map when
///   the direction slice is not `-1`, `painter_light_apply` otherwise.
///
/// Lit content applies it to its colour; content that is not lit, or whose
/// layer no range holds, keeps its colour.
pub const SHARED_PAINTER_MASK_WGSL: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/helpers/shared_painter_mask.wgsl"
));

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

#[cfg(test)]
mod tests {
    #[cfg(wgpu29)]
    use naga29 as naga;
    #[cfg(wgpu30)]
    use naga30 as naga;

    use super::*;

    fn validate(label: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{label} failed to parse: {}", e.emit_to_string(src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label} failed to validate: {e:?}"));
    }

    // A fragment calling every lighting helper the PBR catalogue publishes,
    // composed the way a plugin composes it.
    const PBR_CALLER: &str = "
@fragment
fn fs_main(@location(0) world_pos: vec3<f32>, @location(1) n: vec3<f32>) -> @location(0) vec4<f32> {
    let v = normalize(camera.eye_pos - world_pos);
    var inputs: PbrInputs;
    inputs.world_pos = world_pos;
    inputs.world_n = n;
    inputs.view_dir = v;
    inputs.albedo = vec3<f32>(0.8);
    inputs.metallic = 0.5;
    inputs.roughness = 0.4;
    inputs.ao = 1.0;
    inputs.emissive = vec3<f32>(0.0);
    let lit = viewport_pbr_shade(inputs);
    let lambert = viewport_apply_scene_lighting(n, vec3<f32>(0.8), false, world_pos);
    let ibl = viewport_ibl_ambient(n, v, vec3<f32>(0.8), 0.5, 0.4, vec3<f32>(0.04), 1.0);
    let shadow = viewport_sample_csm(world_pos, n);
    return vec4<f32>((lit + lambert + ibl) * shadow, 1.0);
}
";

    #[test]
    fn pbr_catalogue_composes_after_the_shared_bindings() {
        let src = format!("{SHARED_BINDINGS_WGSL}\n{SHARED_PBR_WGSL}\n{PBR_CALLER}");
        validate("SHARED_BINDINGS_WGSL + SHARED_PBR_WGSL", &src);
    }

    // The bindings SHARED_CSM_WGSL and SHARED_PUNCTUAL_SHADOW_WGSL read that
    // SHARED_SCENE_LIGHTING_WGSL does not declare, and a light loop calling
    // the per-light shadow.
    const SHADOW_CALLER: &str = "
struct ShadowCamera {
    view_proj: mat4x4<f32>,
    eye_pos:   vec3<f32>,
    _pad0:     f32,
    forward:   vec3<f32>,
    exposure:  f32,
};
@group(0) @binding(0) var<uniform> camera: ShadowCamera;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var<uniform> lights_uniform: Lights;
@group(0) @binding(5) var<uniform> shadow_atlas: ShadowAtlas;
";

    const SHADOW_LOOP: &str = "
@fragment
fn fs_main(@location(0) world_pos: vec3<f32>, @location(1) n: vec3<f32>) -> @location(0) vec4<f32> {
    var lit = 0.0;
    for (var i = 0u; i < lights_uniform.count; i++) {
        let l = lights_storage[i];
        let s = shadow_for_light(l, world_pos, camera.eye_pos, n, n, n, 0u);
        lit += s.factor;
    }
    return vec4<f32>(lit, 0.0, 0.0, 1.0);
}
";

    #[test]
    fn punctual_shadows_compose_after_the_cascades() {
        let src = format!(
            "{SHARED_SCENE_LIGHTING_WGSL}\n{SHARED_SHADOW_ATLAS_WGSL}\n{SHADOW_CALLER}\n\
             {SHARED_CSM_WGSL}\n{SHARED_PUNCTUAL_SHADOW_WGSL}\n{SHADOW_LOOP}"
        );
        validate("SHARED_CSM_WGSL + SHARED_PUNCTUAL_SHADOW_WGSL", &src);
    }

    #[test]
    fn brdf_section_composes_alone() {
        let src = format!(
            "{SHARED_BRDF_WGSL}
@fragment
fn fs_main(@location(0) n: vec3<f32>) -> @location(0) vec4<f32> {{
    let l = normalize(vec3<f32>(0.0, 0.0, 1.0));
    let c = viewport_brdf_light_contrib(n, l, l, vec3<f32>(1.0), vec3<f32>(0.5), 0.0, 0.5, vec3<f32>(0.04));
    return vec4<f32>(c, 1.0);
}}
"
        );
        validate("SHARED_BRDF_WGSL", &src);
    }
}
