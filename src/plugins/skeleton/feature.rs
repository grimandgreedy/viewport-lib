//! One-call install for a GPU-skinned mesh.
//!
//! GPU skinning is three pieces wired to two objects: the skinning deformer on
//! the renderer, a pose-driving runtime plugin on the runtime, and a per-frame
//! palette upload the host runs after `step`. [`SkinnedMeshFeature`] installs
//! the first two in one call and returns a [`SkinnedMeshHandle`] that does the
//! third, so the host does not hand-wire the drain-and-attach loop itself.
//!
//! The palette upload cannot move into a `GpuPlugin`: it writes through the
//! renderer's `DeviceResources` (lazily creating and reallocating per-instance
//! buffers), which a device/queue-only plugin hook never sees. So the handle
//! keeps a per-frame [`apply`](SkinnedMeshHandle::apply) call the host makes
//! from wherever it holds both the runtime output and the renderer.

use std::collections::HashMap;

use crate::MeshId;
use crate::error::ViewportResult;
use crate::plugin_api::{PluginInstallCtx, ViewportPlugin};
use crate::plugins::skinning::{SkinWeights, SkinnedPoseUpdate, SkinningPlugin};
use crate::resources::DeviceResources;
use crate::runtime::{RuntimeOutput, RuntimePlugin};

/// Installs GPU skinning for one or more meshes in a single call.
///
/// `pose_plugin` is a runtime plugin that emits [`SkinnedPoseUpdate`] on the GPU
/// path: a [`SkeletonPlugin`](super::SkeletonPlugin) or
/// [`SkinnedActorPlugin`](super::SkinnedActorPlugin) built
/// `.with_path(SkinningPath::Gpu)`. Add each skinned mesh's weights with
/// [`with_weights`](Self::with_weights); the returned handle uploads them the
/// first time a pose update names the mesh.
pub struct SkinnedMeshFeature<P: RuntimePlugin> {
    pose_plugin: P,
    weights: HashMap<MeshId, SkinWeights>,
}

impl<P: RuntimePlugin> SkinnedMeshFeature<P> {
    /// Wrap a pose-driving runtime plugin. Add mesh weights with
    /// [`with_weights`](Self::with_weights) before installing.
    pub fn new(pose_plugin: P) -> Self {
        Self {
            pose_plugin,
            weights: HashMap::new(),
        }
    }

    /// Supply the skin weights for one mesh. The handle uploads them the first
    /// time it sees a pose update for that mesh.
    pub fn with_weights(mut self, mesh_id: MeshId, weights: SkinWeights) -> Self {
        self.weights.insert(mesh_id, weights);
        self
    }
}

impl<P: RuntimePlugin> ViewportPlugin for SkinnedMeshFeature<P> {
    type Handle = SkinnedMeshHandle;

    fn install(self, ctx: &mut PluginInstallCtx<'_>) -> ViewportResult<SkinnedMeshHandle> {
        // Check the runtime is present before touching the renderer, so a
        // missing runtime does not leave a deformer registered with no plugin
        // driving it.
        ctx.require_runtime("a ViewportRuntime for the skinned mesh feature")?;

        let skinning = SkinningPlugin::install(ctx.renderer.resources_mut(), ctx.device)?;

        ctx.runtime
            .as_deref_mut()
            .expect("runtime presence checked above")
            .add_plugin(self.pose_plugin);

        Ok(SkinnedMeshHandle {
            skinning,
            weights: self.weights,
        })
    }
}

/// Uploads joint palettes for the meshes a [`SkinnedMeshFeature`] drives.
///
/// Call [`apply`](Self::apply) once per frame with the runtime output and the
/// renderer's resources. It drains [`SkinnedPoseUpdate`] events, uploads a
/// mesh's skin weights the first time it sees the mesh, and writes the joint
/// palette for each update.
pub struct SkinnedMeshHandle {
    skinning: SkinningPlugin,
    weights: HashMap<MeshId, SkinWeights>,
}

impl SkinnedMeshHandle {
    /// The underlying deformer handle, for hosts that want to attach weights or
    /// palettes directly.
    pub fn skinning(&self) -> &SkinningPlugin {
        &self.skinning
    }

    /// Drain pose updates from `output` and upload them.
    ///
    /// A mesh's weights are uploaded the first time a pose update names it,
    /// using the map given to the feature. A mesh with no weights entry is
    /// skipped: its palette attach is a no-op until weights exist.
    pub fn apply(
        &mut self,
        resources: &mut DeviceResources,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output: &mut RuntimeOutput,
    ) {
        for update in output.events.drain::<SkinnedPoseUpdate>() {
            if !self.skinning.is_skinned_mesh(update.mesh_id) {
                if let Some(weights) = self.weights.get(&update.mesh_id) {
                    self.skinning
                        .attach_weights(resources, device, update.mesh_id, weights);
                }
            }
            self.skinning.attach_palette(
                resources,
                device,
                queue,
                update.mesh_id,
                update.instance_id,
                &update.joint_matrices,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ViewportError;
    use crate::plugins::skeleton::plugin::{SkeletonPlugin, SkinningPath};
    use crate::plugins::skeleton::skeleton::{Joint, Pose, Skeleton};
    use crate::renderer::ViewportRenderer;
    use crate::runtime::ViewportRuntime;
    use glam::Affine3A;

    fn headless() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
            required_limits: crate::renderer::ViewportRenderer::recommended_device_limits(&adapter),
            ..Default::default()
        }))
        .ok()
    }

    fn single_joint_skeleton() -> Skeleton {
        Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: Affine3A::IDENTITY,
        }])
    }

    fn gpu_skeleton_plugin(mesh_id: MeshId) -> SkeletonPlugin {
        let weights = SkinWeights {
            joint_indices: vec![[0, 0, 0, 0]],
            joint_weights: vec![[1.0, 0.0, 0.0, 0.0]],
        };
        SkeletonPlugin::new(
            single_joint_skeleton(),
            mesh_id,
            vec![[0.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 1.0]],
            weights,
        )
        .with_path(SkinningPath::Gpu)
    }

    #[test]
    fn install_registers_deformer_and_pose_plugin() {
        let Some((device, queue)) = headless() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        let mesh_id = MeshId::new(0, 0);
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let mut runtime = ViewportRuntime::new();

        let _handle = {
            let mut ctx = PluginInstallCtx::new(&device, &queue, Some(&mut runtime), &mut renderer);
            SkinnedMeshFeature::new(gpu_skeleton_plugin(mesh_id))
                .install(&mut ctx)
                .expect("install")
        };

        // Renderer half: the skinning deformer is registered.
        assert!(
            renderer
                .resources()
                .deformer_id_by_name("viewport_skin")
                .is_some(),
            "install registered the skinning deformer"
        );

        // Runtime half: with a Pose in scope, stepping emits a pose update.
        runtime.resources_mut().insert(Pose::identity(1));
        let mut scene = crate::scene::scene::Scene::new();
        let mut sel = crate::interaction::select::selection::Selection::new();
        let mut frame = crate::runtime::RuntimeFrameContext::default();
        frame.dt = 1.0 / 60.0;
        let mut output = runtime.step(&mut scene, &mut sel, &frame);
        let updates = output.events.drain::<SkinnedPoseUpdate>();
        assert_eq!(updates.len(), 1, "the pose plugin emitted one update");
        assert_eq!(updates[0].mesh_id, mesh_id);
    }

    #[test]
    fn install_without_runtime_leaves_renderer_untouched() {
        let Some((device, queue)) = headless() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);

        let mut ctx = PluginInstallCtx::new(&device, &queue, None, &mut renderer);
        let result =
            SkinnedMeshFeature::new(gpu_skeleton_plugin(MeshId::new(0, 0))).install(&mut ctx);

        assert!(matches!(
            result,
            Err(ViewportError::PluginInstallMissing { .. })
        ));
        assert!(
            renderer
                .resources()
                .deformer_id_by_name("viewport_skin")
                .is_none(),
            "a missing runtime leaves the deformer unregistered"
        );
    }

    #[test]
    fn apply_drains_pose_updates() {
        let Some((device, queue)) = headless() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };

        let mesh_id = MeshId::new(0, 0);
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let mut runtime = ViewportRuntime::new();

        // No weights given, so apply is a safe no-op on an unuploaded mesh: it
        // still drains the event, which is what we check here.
        let mut handle = {
            let mut ctx = PluginInstallCtx::new(&device, &queue, Some(&mut runtime), &mut renderer);
            SkinnedMeshFeature::new(gpu_skeleton_plugin(mesh_id))
                .install(&mut ctx)
                .expect("install")
        };

        let mut output = RuntimeOutput::default();
        output.events.emit(SkinnedPoseUpdate {
            mesh_id,
            instance_id: 0,
            joint_matrices: vec![glam::Mat4::IDENTITY],
        });

        handle.apply(renderer.resources_mut(), &device, &queue, &mut output);

        assert!(
            output.events.drain::<SkinnedPoseUpdate>().is_empty(),
            "apply drained the pose update"
        );
    }
}
