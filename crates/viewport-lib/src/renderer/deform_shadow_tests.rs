//! The shadow passes have to run the deformers too.
//!
//! A mesh deformed on the GPU (a skinned character, a displaced surface) keeps its rest geometry
//! in its own vertex buffer: the deformation exists only inside the shader. So a pass that
//! rasterises that mesh without the deformer body spliced in draws the rest pose, and for a depth
//! pass the result is a shadow that never moves while the lit mesh animates perfectly.
//!
//! `shadow_point.wgsl` was in exactly that state: the contract, the context and both hook calls
//! were present, the pipeline carried the deform bind group, and the composition list did not name
//! it, so every hook call stayed an identity pass-through. These tests hold both shadow pipelines
//! to the rule the mesh pipelines already follow. Skips when no wgpu adapter is available.

use super::ViewportRenderer;
use crate::resources::mesh_sidecar::registry::{DeformStage, DeformerDesc};

fn headless_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
    let instance = crate::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &crate::gpu::RequestAdapterOptions {
            power_preference: crate::gpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            #[cfg(wgpu30)]
            apply_limit_buckets: false,
        },
    ))
    .ok()?;
    pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
        label: Some("deform_shadow_tests"),
        required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
        ..Default::default()
    }))
    .ok()
}

/// A deformer that moves a vertex, so a pass that runs it and a pass that does not disagree.
fn bulge() -> DeformerDesc {
    DeformerDesc {
        name: "test_bulge",
        stage: DeformStage::ObjectSpace,
        priority: 0,
        wgsl_body: "fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {\n    \
                    var out = v;\n    out.position.y = out.position.y + 1.0;\n    return out;\n}\n"
            .to_string(),
        per_vertex_stride: 4,
    }
}

/// Registering a deformer has to rebuild the **point**-shadow pipeline, not just the cascade one.
///
/// `ensure_point_shadow_pipeline` returns early once the slot is filled and builds from the
/// uncomposed source, so if the rebuild does not overwrite it the pipeline that draws every point
/// light's cube faces runs the identity deformer for the rest of the session.
#[test]
fn registering_a_deformer_rebuilds_both_shadow_pipelines() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping registering_a_deformer_rebuilds_both_shadow_pipelines: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
    if !renderer.resources().deform.enabled {
        eprintln!("skipping: device has too few bind groups for the deform path");
        return;
    }

    renderer
        .resources_mut()
        .register_deformer(&device, bulge())
        .expect("the bulge registers");
    renderer
        .resources_mut()
        .flush_mesh_pipeline_rebuild(&device);

    assert!(
        renderer.resources().shadow.pipeline.is_some(),
        "the cascade shadow pipeline was not composed"
    );
    assert!(
        renderer.resources().shadow.point_pipeline.is_some(),
        "the point shadow pipeline was not composed, so every point light's cube faces draw the \
         undeformed mesh"
    );
}

/// The composition itself: both shadow shaders carry the registered body, rather than the
/// identity pass-through they ship with. This is what the rebuilt pipelines are built from, and it
/// fails on the `shadow_point.wgsl` side alone if the shader leaves the composition list again.
#[test]
fn both_shadow_shaders_compose_the_registered_body() {
    use crate::resources::mesh_sidecar::registry::{MESH_FAMILY_SHADERS, compose_shader};

    for name in ["shadow.wgsl", "shadow_point.wgsl"] {
        assert!(
            MESH_FAMILY_SHADERS.contains(&name),
            "'{name}' is not composed, so its deform hooks stay identity pass-throughs"
        );
        let Some(base) = crate::resources::mesh_sidecar::registry::lookup_source(name) else {
            return;
        };
        assert!(
            !base.contains("test_bulge__deform"),
            "'{name}' must not ship the test body"
        );
        let composed = compose_shader(base, &[stored(bulge())]);
        assert!(
            composed.contains("test_bulge__deform"),
            "'{name}' did not receive the deformer body"
        );
    }
}

fn stored(desc: DeformerDesc) -> crate::resources::mesh_sidecar::registry::StoredDeformer {
    crate::resources::mesh_sidecar::registry::StoredDeformer { desc, slot: 0 }
}
