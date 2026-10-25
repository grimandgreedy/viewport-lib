//! The colour and selection mask shaders of the item types that register on
//! the compatibility tier, adapted for that tier, translate to the GLSL ES 3.0
//! WebGL2 compiles, and every plugin shader translates for a full-tier GL
//! device.
//!
//! naga's GLSL writer is the path wgpu's GL backend takes, and it rejects some
//! WGSL that every other backend accepts: a texture read through two samplers,
//! or a `textureLoad` from a depth texture.

#![cfg(feature = "wgpu27")]

use viewport_lib::{Capabilities, Tier, gpu};

/// The shaders that draw on the compatibility tier: each type's colour shader and
/// its selection mask. Their pick and shadow shaders run only on the full
/// tier.
const FORWARD_SHADERS: &[&str] = &[
    "implicit.wgsl",
    "implicit_outline_mask.wgsl",
    "volume.wgsl",
    "volume_outline_mask.wgsl",
    "streamtube.wgsl",
    "ribbon.wgsl",
    "curve_outline_mask.wgsl",
    "point_cloud_stream.wgsl",
    "point_disc_mask.wgsl",
    "vector_field_stream.wgsl",
    "vector_field_outline_mask_stream.wgsl",
    "mc_isosurface.wgsl",
    "gaussian_splat_packed.wgsl",
    "gaussian_splat_pack.wgsl",
];

fn webgl2_caps() -> Capabilities {
    let caps = Capabilities::from_parts(
        &gpu::Limits::downlevel_webgl2_defaults(),
        gpu::Features::empty(),
        Some(gpu::DownlevelFlags::empty()),
    )
    .with_backend(gpu::Backend::Gl);
    assert_eq!(caps.tier(), Tier::Compatibility);
    caps
}

/// The first function-local array `module` indexes with a run-time value, as
/// `` `function`: `array` ``. Adreno's GLES compiler miscompiles that pattern
/// with no error (an isosurface whose box corners came from one drew nothing on
/// an Adreno 650), so shaders that run below the full tier keep such tables at
/// module scope as a `const`, or compute them.
fn local_array_indexed_at_run_time(module: &naga::Module) -> Option<String> {
    let functions = module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|ep| &ep.function));
    for function in functions {
        for (_, expr) in function.expressions.iter() {
            let naga::Expression::Access { base, .. } = expr else {
                continue;
            };
            let (array, name) = match function.expressions[*base] {
                naga::Expression::LocalVariable(local) => {
                    let local = &function.local_variables[local];
                    (local.ty, local.name.clone())
                }
                naga::Expression::Compose { ty, .. } => (ty, None),
                _ => continue,
            };
            if matches!(module.types[array].inner, naga::TypeInner::Array { .. }) {
                return Some(format!(
                    "`{}`: `{}`",
                    function.name.as_deref().unwrap_or("?"),
                    name.as_deref().unwrap_or("an array constructed in place")
                ));
            }
        }
    }
    None
}

/// The first function of `module` that samples or gathers a depth texture
/// without a depth reference, as `` `function`: `texture` ``. naga declares
/// every depth texture a `sampler2DShadow` in GLSL and emits the lookup with no
/// reference value, which matches no GLSL overload, so the driver rejects the
/// program although naga reports no error.
fn depth_sampled_without_comparison(module: &naga::Module) -> Option<String> {
    let functions = module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|ep| &ep.function));
    for function in functions {
        for (_, expr) in function.expressions.iter() {
            let naga::Expression::ImageSample {
                image,
                depth_ref: None,
                ..
            } = expr
            else {
                continue;
            };
            let (ty, name) = match function.expressions[*image] {
                naga::Expression::GlobalVariable(var) => {
                    let var = &module.global_variables[var];
                    (var.ty, var.name.clone())
                }
                naga::Expression::FunctionArgument(i) => {
                    let arg = &function.arguments[i as usize];
                    (arg.ty, arg.name.clone())
                }
                _ => continue,
            };
            if matches!(
                module.types[ty].inner,
                naga::TypeInner::Image {
                    class: naga::ImageClass::Depth { .. },
                    ..
                }
            ) {
                return Some(format!(
                    "`{}`: `{}`",
                    function.name.as_deref().unwrap_or("?"),
                    name.as_deref().unwrap_or("?")
                ));
            }
        }
    }
    None
}

/// The entry points of `src` naga's GLSL writer cannot translate at `version`,
/// as `(entry point, error)`. An error that is not about one entry point
/// (parsing, validation, a uniform size) is the `Err`.
fn glsl_failures(
    src: &str,
    version: naga::back::glsl::Version,
) -> Result<Vec<(String, String)>, String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("{e:?}"))?;
    // wgpu's GL backend (no `BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED`) refuses a
    // pipeline whose uniform types are not a multiple of 16 bytes, a rule no
    // native device enforces, so it is checked here.
    for (_, var) in module.global_variables.iter() {
        if var.space == naga::AddressSpace::Uniform {
            let size = module.types[var.ty].inner.size(module.to_ctx());
            if size % 16 != 0 {
                return Err(format!(
                    "uniform `{}` is {size} bytes, not a multiple of 16",
                    var.name.as_deref().unwrap_or("?")
                ));
            }
        }
    }
    let mut failures = Vec::new();
    for ep in &module.entry_points {
        let result = naga::back::pipeline_constants::process_overrides(
            &module,
            &info,
            Some((ep.stage, ep.name.as_str())),
            &Default::default(),
        )
        .map_err(|e| e.to_string())
        .and_then(|(module, info)| {
            let options = naga::back::glsl::Options {
                version,
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
            .map(|_| ())
            .map_err(|e| e.to_string())
        });
        if let Err(e) = result {
            failures.push((ep.name.clone(), e));
        }
    }
    Ok(failures)
}

/// The run-time indexed local array in `src`, as an error.
fn local_array_error(src: &str) -> Option<String> {
    let module = naga::front::wgsl::parse_str(src).ok()?;
    local_array_indexed_at_run_time(&module).map(|at| {
        format!(
            "{at} is a function-local array indexed by a run-time value, which Adreno \
             miscompiles; make it a module-scope const or compute it"
        )
    })
}

/// A non-comparison sample of a depth texture in `src`, as an error.
fn depth_sample_error(src: &str) -> Option<String> {
    let module = naga::front::wgsl::parse_str(src).ok()?;
    depth_sampled_without_comparison(&module).map(|at| {
        format!(
            "{at} samples a depth texture without a comparison, which GLSL has no \
             lookup for; bind it as texture_2d<f32>"
        )
    })
}

fn translate(src: &str) -> Result<(), String> {
    if let Some(e) = local_array_error(src) {
        return Err(e);
    }
    if let Some(e) = depth_sample_error(src) {
        return Err(e);
    }
    let version = naga::back::glsl::Version::Embedded {
        version: 300,
        is_webgl: true,
    };
    match glsl_failures(src, version)?.into_iter().next() {
        Some((ep, e)) => Err(format!("{ep}: {e}")),
        None => Ok(()),
    }
}

#[test]
fn compatibility_tier_shaders_translate_to_webgl2_glsl() {
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

/// Plugin shader entry points the GLSL writer cannot translate yet for a
/// full-tier GL device, as `shader:entry point`. The test fails when one is
/// fixed and still listed, so the list stays exact.
const GL_KNOWN_FAILURES: &[&str] = &[
    // `textureLoad` from a depth texture.
    "decal.wgsl:fs_main",
    "decal_outline_mask.wgsl:fs_main",
    "scatter_refraction.wgsl:fs_main",
    "scatter_temporal_resolve.wgsl:fs_main",
    "scatter_volume.wgsl:fs_main",
    // A depth texture sampled without a comparison: naga emits a
    // `sampler2DShadow` lookup with no reference, which no GLSL overload
    // matches, and reports no error.
    "sprite.wgsl:depth-sample",
    "sprite_lit.wgsl:depth-sample",
    // A `@interpolate(linear)` varying, which GLSL ES has no qualifier for.
    "surface_contour.wgsl:fs_main",
    "surface_contour.wgsl:vs_main",
];

/// Every plugin shader, adapted for a desktop GL device at the full tier (cube
/// arrays, no depth-texture loads), translates at both GLSL versions wgpu's GL
/// backend emits: GLSL ES 3.2 on a GLES context and desktop GLSL 4.50 (the
/// driver's version, which wgpu caps at 450) on a desktop one.
#[test]
fn full_tier_shaders_translate_to_gl_glsl() {
    let mut caps = Capabilities::from_parts(&gpu::Limits::default(), gpu::Features::empty(), None)
        .with_backend(gpu::Backend::Gl);
    caps.cube_array_textures = true;
    let versions = [
        naga::back::glsl::Version::Embedded {
            version: 320,
            is_webgl: false,
        },
        naga::back::glsl::Version::Desktop(450),
    ];
    let mut failures = std::collections::BTreeMap::<String, String>::new();
    for (name, src) in viewport_lib_plugins::shader_sources() {
        let adapted = viewport_lib::plugin_api::builders::adapt_wgsl(src.as_str(), &caps);
        if let Some(e) = depth_sample_error(&adapted) {
            failures.insert(format!("{name}:depth-sample"), e);
        }
        for version in versions {
            match glsl_failures(&adapted, version) {
                Ok(eps) => {
                    for (ep, e) in eps {
                        let at = failures.entry(format!("{name}:{ep}")).or_default();
                        at.push_str(&format!("[{version:?}] {e} "));
                    }
                }
                Err(e) => {
                    failures.entry(format!("{name}:*")).or_insert(e);
                }
            }
        }
    }
    let new: Vec<String> = failures
        .iter()
        .filter(|(k, _)| !GL_KNOWN_FAILURES.contains(&k.as_str()))
        .map(|(k, e)| format!("{k}: {e}"))
        .collect();
    let fixed: Vec<&str> = GL_KNOWN_FAILURES
        .iter()
        .copied()
        .filter(|k| !failures.contains_key(*k))
        .collect();
    assert!(
        new.is_empty() && fixed.is_empty(),
        "new GL translation failures:\n{}\n\nlisted but now translate: {fixed:?}",
        new.join("\n")
    );
}
