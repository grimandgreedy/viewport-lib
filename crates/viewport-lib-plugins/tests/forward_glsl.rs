//! The colour shaders of the item types that register on the forward tier,
//! adapted for that tier, translate to the GLSL ES 3.0 WebGL2 compiles.
//!
//! naga's GLSL writer is the path wgpu's GL backend takes, and it rejects some
//! WGSL that every other backend accepts: a texture read through two samplers,
//! or a `textureLoad` from a depth texture.

#![cfg(feature = "wgpu27")]

use viewport_lib::{Capabilities, Tier, gpu};

/// The shaders that draw on the forward tier. Their pick, outline and shadow
/// shaders run only on the full tier.
const FORWARD_SHADERS: &[&str] = &[
    "implicit.wgsl",
    "volume.wgsl",
    "streamtube.wgsl",
    "ribbon.wgsl",
];

fn webgl2_caps() -> Capabilities {
    let caps = Capabilities::from_parts(
        &gpu::Limits::downlevel_webgl2_defaults(),
        gpu::Features::empty(),
        Some(gpu::DownlevelFlags::empty()),
    )
    .with_backend(gpu::Backend::Gl);
    assert_eq!(caps.tier(), Tier::Forward);
    caps
}

fn translate(src: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("{e:?}"))?;
    for ep in &module.entry_points {
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &module,
            &info,
            Some((ep.stage, ep.name.as_str())),
            &Default::default(),
        )
        .map_err(|e| format!("{}: {e}", ep.name))?;
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };
        let pipeline = naga::back::glsl::PipelineOptions {
            shader_stage: ep.stage,
            entry_point: ep.name.clone(),
            multiview: None,
        };
        let mut out = String::new();
        naga::back::glsl::Writer::new(
            &mut out,
            &module,
            &info,
            &options,
            &pipeline,
            naga::proc::BoundsCheckPolicies::default(),
        )
        .and_then(|mut writer| writer.write())
        .map_err(|e| format!("{}: {e}", ep.name))?;
    }
    Ok(())
}

#[test]
fn forward_tier_shaders_translate_to_webgl2_glsl() {
    let caps = webgl2_caps();
    let sources = viewport_lib_plugins::shader_sources();
    let mut failures = Vec::new();
    for name in FORWARD_SHADERS {
        let (_, src) = sources
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} is not in shader_sources()"));
        let adapted = viewport_lib::plugin_api::builders::adapt_wgsl(src.as_str(), &caps);
        if let Err(e) = translate(&adapted) {
            failures.push(format!("{name}: {e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
