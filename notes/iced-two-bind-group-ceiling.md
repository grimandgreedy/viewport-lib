# iced's two-bind-group ceiling

iced_wgpu requests its shared device with `max_bind_groups = 2` (for WebGL2
portability) and offers no way to raise it: the limit is hardcoded in
iced_wgpu's device descriptor, so any consumer rendering into iced's textures
inherits it. This is not a hardware limit; every desktop GPU supports at least
four bind groups, and the other supported hosts (eframe, winit, Slint, Bevy)
request the wgpu default of 4.

## What this means for viewport-lib features

viewport-lib's mesh family normally uses three bind groups: camera (group 0),
object (group 1), and the deform sidecar (group 2). Features whose pipelines
need group 2 or higher are disabled on a 2-group device:

- GPU skinning and vertex deformers
- decals
- soft bodies
- refraction
- volume rendering
- GPU picking
- custom material/shading plugins (their per-material bind group is group 3,
  so they need `max_bind_groups >= 4`)

The core mesh path works within two groups: lit rendering, shadows, HDR
post-processing, OIT transparency, and wireframe.

## How the gating works

At renderer construction, `deform_enabled = device.limits().max_bind_groups >= 3`
(`src/resources/init.rs`). On limited devices the mesh family compiles 2-group
"noop" pipeline layouts from the `_noop` shader variants, and every mesh-family
deform bind at draw time goes through the `bind_deform_group!` macro
(`src/renderer/types/mod.rs`), which no-ops when deform is disabled. Binding
group 2 on a 2-group device is a wgpu validation error, so the guard must stay
in that one macro; do not open-code a deform `set_bind_group(2, ...)` at a draw
site.

The invariant is enforced headlessly on every `cargo test` by
`two_bind_group_device_renders_without_validation_errors` (`tests/headless.rs`),
which renders the LDR, HDR/OIT, and wireframe paths on a real device capped at
two bind groups.