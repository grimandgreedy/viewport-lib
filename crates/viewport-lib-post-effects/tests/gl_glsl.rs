//! Every post-effect shader translates at both GLSL versions wgpu's GL backend
//! emits: GLSL ES 3.2 on a GLES context and desktop GLSL 4.50 (the driver's
//! version, which wgpu caps at 450) on a desktop one.
//!
//! naga's GLSL writer rejects some WGSL that every other backend accepts, such
//! as a `textureLoad` from a depth texture.

#![cfg(feature = "wgpu27")]

/// Entry points the GLSL writer cannot translate yet, as `shader:entry point`.
/// The test fails when one is fixed and still listed, so the list stays exact.
const KNOWN_FAILURES: &[&str] = &[];

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

/// The entry points of `src` that fail to translate at `version`, as
/// `(entry point, error)`.
fn glsl_failures(src: &str, version: naga::back::glsl::Version) -> Vec<(String, String)> {
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => return vec![("*".into(), e.emit_to_string(src))],
    };
    let info = match naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    {
        Ok(info) => info,
        Err(e) => return vec![("*".into(), format!("{e:?}"))],
    };
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
    failures
}

#[test]
fn post_effect_shaders_translate_to_gl_glsl() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let versions = [
        naga::back::glsl::Version::Embedded {
            version: 320,
            is_webgl: false,
        },
        naga::back::glsl::Version::Desktop(450),
    ];
    let mut failures = std::collections::BTreeMap::<String, String>::new();
    for entry in std::fs::read_dir(&dir).expect("src/shaders") {
        let path = entry.expect("shader entry").path();
        if path.extension().is_none_or(|e| e != "wgsl") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // The depth passes compose the depth-texel helper in front of their
        // body; composing it in front of every shader is harmless.
        let src = format!(
            "{}\n{}",
            viewport_lib::plugin_api::shared_wgsl::SHARED_DEPTH_TEXEL_WGSL,
            std::fs::read_to_string(&path).expect("shader source")
        );
        if let Some(at) = naga::front::wgsl::parse_str(&src)
            .ok()
            .and_then(|m| depth_sampled_without_comparison(&m))
        {
            failures.insert(
                format!("{name}:depth-sample"),
                format!("{at} samples a depth texture without a comparison"),
            );
        }
        for version in versions {
            for (ep, e) in glsl_failures(&src, version) {
                let at = failures.entry(format!("{name}:{ep}")).or_default();
                at.push_str(&format!("[{version:?}] {e} "));
            }
        }
    }
    let new: Vec<String> = failures
        .iter()
        .filter(|(k, _)| !KNOWN_FAILURES.contains(&k.as_str()))
        .map(|(k, e)| format!("{k}: {e}"))
        .collect();
    let fixed: Vec<&str> = KNOWN_FAILURES
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
