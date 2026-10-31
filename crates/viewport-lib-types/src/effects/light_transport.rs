//! How light values travel from a scene to the display: units, where exposure
//! applies, which colours it leaves alone, the default fill and the view
//! transform.
//!
//! This module holds no code. It is the contract the renderer, the built-in
//! item types and plugins keep, so a scene authored against it renders the
//! same way on every path. The clauses are numbered so other documentation can
//! name them.
//!
//! "The camera's exposure" below is the single linear multiplier an
//! [`ExposureMode`] and its compensation resolve to for a frame. A camera
//! carries its settings as [`PhysicalExposure`]; in this release the renderer
//! reads the mode from [`DisplaySettings::exposure`], so set it there (for a
//! physical camera, with [`PhysicalExposure::to_exposure_mode`]).
//!
//! # 1. Units
//!
//! Light intensities are photometric:
//!
//! - directional lights in lux ([`Lux`]);
//! - point and spot lights in candela ([`Candela`]);
//! - emissive surfaces in nits (cd/m^2): [`Material::emissive`] times
//!   [`Material::emissive_strength`], and the emissive slots `0..3` of
//!   [`ItemSettings::custom_data`];
//! - image-based lighting, environment zones and the hemisphere fill in nits,
//!   after [`EnvironmentIntensity`] or
//!   [`LightingSettings::hemisphere_intensity`] scales them.
//!
//! A [`Colour`] carries its own encoding and is linear once it reaches the
//! shader. Textures and material factors (base colour, metallic, roughness,
//! occlusion, the emissive colour itself) are unitless multipliers on those
//! quantities.
//!
//! Lightmap texels are radiance in nits under [`LightmapMode::Replace`],
//! [`LightmapMode::Add`] and [`LightmapMode::Subtractive`], and in the
//! radiance texture of a shadowmask lightmap. Under
//! [`LightmapMode::AmbientOcclusion`] they are a unitless occlusion factor.
//!
//! # 2. Exposure is applied before shading
//!
//! Every radiance source a shader reads is multiplied by the camera's exposure
//! where it is read: punctual lights, image-based lighting and environment
//! zones, the hemisphere fill, light probes, lightmaps, emissive and the
//! custom-data emissive slots. Lighting is then computed on exposed radiance,
//! so the values a pixel carries are already in the range the view transform
//! expects.
//!
//! Material plugin hooks (`shade_surface`, `shade_light`, `shade_ambient`,
//! `recolor`), item-type plugins that read the lights uniform, and post-effect
//! producers see exposed radiance and return exposed radiance. A plugin that
//! adds its own light multiplies it by the exposure the same way.
//!
//! The shader clamps exposed radiance to `lit_clamp`: 65504 (the largest
//! half-float) on the HDR path, 1.0 on [`PipelineMode::Direct`], where the
//! target cannot hold more.
//!
//! # 3. Emissive
//!
//! Emissive is radiance in nits with an exposure weight `w` in `[0, 1]`,
//! default 1 ([`Material::emissive_exposure_weight`]). The shader sums the emissive terms (the material's colour,
//! strength and map, a surface hook's emissive, the custom-data slots) and
//! multiplies the sum by `mix(1, exposure, w)`.
//!
//! At `w = 1` an emitter is a light source: a 2 000 nit panel reads as bright
//! under a daylight exposure as a sunlit white card does. At `w = 0` the
//! emissive is display-relative: an emissive of 1.0 shows as 1.0 whatever the
//! exposure, which suits indicator lights and highlights that must stay
//! visible.
//!
//! # 4. Resolved colours are display-relative
//!
//! Some colours are data rather than light, and exposure does not touch them:
//! the colour of an [`ItemSettings::unlit`] item, matcaps, UV visualisation,
//! face colours, wireframe, the NaN colour of a scalar map, and the colours of
//! item types that draw without lighting, such as a point cloud's solid,
//! per-point or colour-mapped colours. A colour-mapped value that maps to 0.5
//! shows as 0.5 (before the view transform) under any exposure, so a colour
//! bar stays true.
//!
//! On a lit mesh, vertex colours and a scalar colour map are the base colour:
//! the light multiplies them like any albedo, and they reach the display
//! through clause 2. Set [`ItemSettings::unlit`] on the item to show the mapped
//! colours themselves.
//!
//! A surface that has to be an unlit emitter in nits, one that should dim and
//! brighten with the camera, is a black base colour with an emissive term at
//! weight 1, not an unlit item.
//!
//! A `recolor` hook replaces lit radiance, so its output is exposed radiance,
//! clamped to `lit_clamp` like the rest of the lit result.
//!
//! # 5. Exposure modes
//!
//! The default is [`ExposureMode::Passthrough`], whose multiplier is exactly
//! 1.0 ([`PASSTHROUGH_EV100`]). Clauses 2 to 4 are then identities: every
//! scene that does not set an exposure renders as it would with no exposure
//! at all, and lights authored at unit scale read as authored.
//!
//! - [`ExposureMode::Manual`] and [`ExposureMode::PhysicalCamera`] resolve to
//!   a fixed multiplier, `1 / (1.2 * 2^EV100)`, known before the frame is
//!   drawn. Clause 2 scales every radiance source by it, clause 3 scales
//!   emissive at weight 1, and clause 4 leaves resolved colours alone. These
//!   modes work on both pipelines.
//! - [`ExposureMode::Automatic`] meters the HDR target and adapts towards it,
//!   so the multiplier the shader reads is the one adapted for that frame.
//!   Metering needs the HDR target: on [`PipelineMode::Direct`] an automatic
//!   exposure falls back to `Passthrough`.
//!
//! # 6. The fill with no environment
//!
//! With image-based lighting off, the ambient term of a PBR material is the
//! hemisphere fill alone: the sky or ground colour, by surface orientation,
//! times [`LightingSettings::hemisphere_intensity`].
//! [`Material::ambient`] is a Blinn-Phong coefficient and does not reach PBR
//! shading.
//!
//! The default fill is 0.55, so an upward-facing surface under a white sky
//! receives 0.55 times its albedo. [`LightingSettings::daylight`] sets 8 000
//! nits to sit beside its 100 000 lux sun. Set the fill to 0 for a scene that
//! should be dark where no light reaches.
//!
//! # 7. The view transform
//!
//! Exposed radiance becomes display pixels in this order:
//!
//! 1. exposed radiance (clause 2);
//! 2. scene-referred grading ([`GradingSpace::Scene`], the default placement
//!    of [`PostProcessSettings::grade_lut`]);
//! 3. the tone-map operator ([`DisplaySettings::operator`]);
//! 4. the background colour, composited in display space;
//! 5. display-referred grading ([`GradingSpace::Display`]);
//! 6. the vignette.
//!
//! The default operator is [`ToneMapping::KhronosNeutral`], Khronos' PBR
//! Neutral curve. The working primaries are Rec.709 (sRGB); radiance outside
//! that gamut is carried as negative or above-one channels and is not
//! gamut-mapped. A wider working space would need the operator and the
//! scene-referred grade's log shaper evaluated in it, and the result converted
//! to the display's primaries at the end of the composite.
//!
//! A named curve does not change: a corrected or retuned curve is a new
//! [`ToneMapping`] variant, so a scene that names an operator keeps its look.
//!
//! The viewport's background colour is a display-space value. It is not
//! exposed or tone mapped, and it is graded only by a display-referred grade.
//!
//! # 8. Pinning and stability
//!
//! A scene that must keep its look sets each of these explicitly rather than
//! relying on a default:
//!
//! - the exposure mode and compensation ([`DisplaySettings::exposure`]);
//! - the tone-map operator ([`DisplaySettings::operator`]);
//! - where a colour grade is applied ([`PostProcessSettings::grading_space`]);
//! - the hemisphere fill ([`LightingSettings::hemisphere_intensity`]) and its
//!   sky and ground colours.
//!
//! The defaults of these settings, and the behaviour of each named operator,
//! do not change within a major version.
//!
//! # Current behaviour
//!
//! Under [`ExposureMode::Automatic`] a frame shades with the exposure the
//! previous frame metered, so the first frame after switching to it draws at
//! the exposure the viewport held before.
//!
//! [`ExposureMode`]: crate::effects::postprocess::ExposureMode
//! [`ExposureMode::Passthrough`]: crate::effects::postprocess::ExposureMode::Passthrough
//! [`ExposureMode::Manual`]: crate::effects::postprocess::ExposureMode::Manual
//! [`ExposureMode::PhysicalCamera`]: crate::effects::postprocess::ExposureMode::PhysicalCamera
//! [`ExposureMode::Automatic`]: crate::effects::postprocess::ExposureMode::Automatic
//! [`PASSTHROUGH_EV100`]: crate::effects::postprocess::PASSTHROUGH_EV100
//! [`DisplaySettings::exposure`]: crate::effects::postprocess::DisplaySettings::exposure
//! [`DisplaySettings::operator`]: crate::effects::postprocess::DisplaySettings::operator
//! [`PipelineMode::Direct`]: crate::effects::postprocess::PipelineMode::Direct
//! [`ToneMapping`]: crate::effects::postprocess::ToneMapping
//! [`ToneMapping::KhronosNeutral`]: crate::effects::postprocess::ToneMapping::KhronosNeutral
//! [`GradingSpace::Scene`]: crate::effects::postprocess::GradingSpace::Scene
//! [`GradingSpace::Display`]: crate::effects::postprocess::GradingSpace::Display
//! [`PostProcessSettings::grade_lut`]: crate::effects::postprocess::PostProcessSettings::grade_lut
//! [`PostProcessSettings::grading_space`]: crate::effects::postprocess::PostProcessSettings::grading_space
//! [`PhysicalExposure`]: crate::camera::PhysicalExposure
//! [`PhysicalExposure::to_exposure_mode`]: crate::camera::PhysicalExposure::to_exposure_mode
//! [`Lux`]: crate::effects::lighting::Lux
//! [`Candela`]: crate::effects::lighting::Candela
//! [`LightingSettings::hemisphere_intensity`]: crate::effects::lighting::LightingSettings::hemisphere_intensity
//! [`LightingSettings::daylight`]: crate::effects::lighting::LightingSettings::daylight
//! [`EnvironmentIntensity`]: crate::effects::environment::EnvironmentIntensity
//! [`LightmapMode::Replace`]: crate::effects::lightmap::LightmapMode::Replace
//! [`LightmapMode::Add`]: crate::effects::lightmap::LightmapMode::Add
//! [`LightmapMode::Subtractive`]: crate::effects::lightmap::LightmapMode::Subtractive
//! [`LightmapMode::AmbientOcclusion`]: crate::effects::lightmap::LightmapMode::AmbientOcclusion
//! [`Material::emissive`]: crate::material::Material::emissive
//! [`Material::emissive_strength`]: crate::material::Material::emissive_strength
//! [`Material::emissive_exposure_weight`]: crate::material::Material::emissive_exposure_weight
//! [`Material::ambient`]: crate::material::Material::ambient
//! [`ItemSettings::custom_data`]: crate::material::ItemSettings::custom_data
//! [`ItemSettings::unlit`]: crate::material::ItemSettings::unlit
//! [`Colour`]: crate::colour::Colour
