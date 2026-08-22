# Migration guide

## Unreleased: photometric lighting + effects config regroup

This release is intentionally breaking. It pins photometric light units, regroups
the per-frame effects configuration by concern, and returns the default lighting
to a faithful "colour is data" baseline. There is no compatibility shim.

### 1. Effects config field moves (mechanical)

Rename the field paths; the values carry over unchanged.

| Before | After |
| --- | --- |
| `effects.exposure` | `effects.display.exposure` |
| `effects.post_process.enabled = true` | `effects.display.mode = PipelineMode::Hdr` |
| `effects.post_process.enabled = false` | `effects.display.mode = PipelineMode::Direct` |
| `effects.post_process.tone_mapping` | `effects.display.operator` |
| `effects.post_process.bloom` | `effects.post_process.bloom.enabled` |
| `effects.post_process.bloom_threshold` | `effects.post_process.bloom.threshold` |
| `effects.post_process.bloom_intensity` | `effects.post_process.bloom.intensity` |
| `effects.post_process.bloom_max_brightness` | `effects.post_process.bloom.max_brightness` |
| `effects.post_process.dof_enabled` | `effects.post_process.dof.enabled` |
| `effects.post_process.dof_focal_distance` | `effects.post_process.dof.focal_distance` |
| `effects.post_process.dof_focal_range` | `effects.post_process.dof.focal_range` |
| `effects.post_process.dof_max_blur_radius` | `effects.post_process.dof.max_blur_radius` |
| `effects.post_process.contact_shadows` | `effects.post_process.contact_shadows.enabled` |
| `effects.post_process.contact_shadow_max_distance` | `effects.post_process.contact_shadows.max_distance` |
| `effects.post_process.contact_shadow_steps` | `effects.post_process.contact_shadows.steps` |
| `effects.post_process.contact_shadow_thickness` | `effects.post_process.contact_shadows.thickness` |
| `effects.post_process.edl_enabled` | `effects.post_process.edl.enabled` |
| `effects.post_process.edl_radius` | `effects.post_process.edl.radius` |
| `effects.post_process.edl_strength` | `effects.post_process.edl.strength` |
| `effects.lighting.shadow_bias` | `effects.lighting.shadows.bias` |
| `effects.lighting.shadows_enabled` | `effects.lighting.shadows.enabled` |
| `effects.lighting.shadow_cascade_count` | `effects.lighting.shadows.cascade_count` |
| `effects.lighting.shadow_atlas_resolution` | `effects.lighting.shadows.atlas_resolution` |
| `effects.lighting.shadow_filter` | `effects.lighting.shadows.filter` |
| `effects.lighting.pcss_light_radius` | `effects.lighting.shadows.pcss_light_radius` |
| `effects.lighting.shadow_extent_override` | `effects.lighting.shadows.extent_override` |
| `effects.lighting.point_shadow_mode` | `effects.lighting.shadows.point_shadow_mode` |
| `effects.clip_objects` | `effects.clip.objects` |
| `effects.cap_fill_enabled` | `effects.clip.cap_fill_enabled` |
| `effects.show_shadow_atlas` | `effects.debug.show_shadow_atlas` |
| `effects.atlas_viewer_corner` | `effects.debug.atlas_viewer_corner` |
| `effects.atlas_viewer_scale` | `effects.debug.atlas_viewer_scale` |
| `EnvironmentMap` (type) | `EnvironmentSettings` |
| `ViewportEffects.scatter` | `SceneEffects.scatter` (scene-global) |

`FrameData::with_exposure` still works (it now writes `display.exposure`);
`FrameData::with_display` sets the whole display group. `EffectsFrame::split()`
and the `session.effects_mut()` accessor are unchanged in spirit.

### 2. LDR pipeline

`PipelineMode::Direct` (formerly `post_process.enabled = false`) is a constrained
passthrough for host-owned render passes (`paint`/`paint_viewport`, always LDR)
and cheap inline rendering. It has no post chain: no exposure, tone mapping, bloom,
SSAO, DOF, FXAA, SSAA, contact shadows, EDL, scatter, skybox, OIT (transparency is
order-dependent), decals, or item-type plugins. Dropped item-type plugins and
transparent volume meshes now log a one-time warning. Use `PipelineMode::Hdr` (the
default) for the full renderer.

### 3. Faithful default lighting

Defaults returned to a neutral baseline; **units are unchanged**, only the default
magnitudes and models:

- Default `ShadingModel` is `Phong` again (was `Pbr`). Set
  `material.shading_model = ShadingModel::Pbr` to keep PBR.
- Default `ExposureSettings` is neutral `Manual { ev: 0 }` (was `Automatic`).
- Default light `intensity` and `hemisphere_intensity` are modest values that read
  at EV 0. For the physical daylight look, use
  `LightingSettings::daylight()` + `ExposureSettings::automatic()`.

### 4. Re-authoring light values (photometric physics)

These are **not** a mechanical rename and are covered by the photometric plan's
follow-up phase, not this refactor:

- **Directional lights**: to preserve a pre-photometric look under the energy-
  normalised (`albedo/pi`) diffuse, multiply directional `intensity` by ~pi.
  Directional `intensity` is illuminance in **lux**.
- **Point / spot lights**: the falloff changed to physical inverse-square with a
  reach window, and `range` no longer scales brightness (reach only). Point/spot
  `intensity` is luminous intensity in **candela** (lumen constructors exist).
  These lights must be re-authored per light; a single multiplier will not match.
- **Default shading model flip** (`Phong` -> historically `Pbr` on the photometric
  branch, now back to `Phong`): scenes relying on the model default should pin the
  model they want explicitly.

See the photometric lighting units plan for the full physics migration, real-lux
default calibration, and companion-repo updates.
