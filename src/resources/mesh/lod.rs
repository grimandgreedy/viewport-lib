//! Level-of-detail mesh groups.
//!
//! A LOD group bundles several meshes that are different-detail versions of one
//! object, ordered from full detail to crude. Each level carries the smallest
//! on-screen size at which it is used. The renderer measures how large an object
//! appears each frame and draws the matching level, so distant objects fall back
//! to cheaper geometry.
//!
//! The group holds geometry only: no material, transform, or per-object state.
//! Those live on the scene item, so swapping levels changes which mesh is drawn
//! and nothing else.

use crate::error::{ViewportError, ViewportResult};
use crate::renderer::RenderCamera;
use crate::resources::mesh::mesh_store::MeshId;
use crate::scene::aabb::Aabb;

/// Margin applied to a level boundary before a switch is allowed, as a fraction
/// of the threshold. Keeps an object sitting on a boundary from flipping level
/// every frame: it must move past the boundary by this much to switch.
const LOD_HYSTERESIS: f32 = 0.1;

/// Handle to a LOD group in the renderer's group registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LodGroupId(pub(crate) usize);

impl LodGroupId {
    /// The raw index into the group registry.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// One detail level of a [`LodGroup`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct LodLevel {
    /// Mesh drawn while this level is selected.
    pub mesh: MeshId,
    /// Smallest projected size, as a fraction of viewport height, at which this
    /// level is used. Levels are ordered full to crude with strictly decreasing
    /// values: the full level has the largest threshold, the crudest the
    /// smallest (often `0.0` so it always applies as the fallback).
    pub min_screen_size: f32,
}

impl LodLevel {
    /// Build a level from a mesh and its lower screen-size bound.
    pub fn new(mesh: MeshId, min_screen_size: f32) -> Self {
        Self {
            mesh,
            min_screen_size,
        }
    }
}

/// How a group moves between adjacent levels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum LodTransition {
    /// Swap meshes at the threshold with no blend.
    #[default]
    Discrete,
}

/// A set of meshes that are detail variants of one object, plus the thresholds
/// that decide which one to draw.
///
/// Upload the level meshes however you like (sync `upload_mesh_data` or the
/// async job path), then bundle them with
/// [`ViewportGpuResources::register_lod_group`].
///
/// [`ViewportGpuResources::register_lod_group`]: crate::ViewportGpuResources::register_lod_group
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LodGroup {
    /// Levels ordered full detail to crude.
    levels: Vec<LodLevel>,
    /// How to move between levels.
    transition: LodTransition,
    /// Stop drawing the object once its projected size drops below this, as a
    /// fraction of viewport height. `None` means never cull on size.
    cull_below: Option<f32>,
}

impl LodGroup {
    /// The levels, full detail first.
    pub fn levels(&self) -> &[LodLevel] {
        &self.levels
    }

    /// Number of levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Mesh for a level index, clamped to the crudest level if out of range.
    pub fn mesh_at(&self, level: usize) -> MeshId {
        let idx = level.min(self.levels.len().saturating_sub(1));
        self.levels[idx].mesh
    }

    /// Transition mode.
    pub fn transition(&self) -> LodTransition {
        self.transition
    }

    /// Size below which the object is culled, if set.
    pub fn cull_below(&self) -> Option<f32> {
        self.cull_below
    }

    /// Whether an object of this projected size should be culled entirely.
    pub fn should_cull(&self, screen_size: f32) -> bool {
        matches!(self.cull_below, Some(c) if screen_size < c)
    }

    /// The level an object of this size maps to, ignoring the current level.
    ///
    /// Returns the finest level whose `min_screen_size` the object clears, or
    /// the crudest level if it clears none.
    pub fn level_for_size(&self, screen_size: f32) -> usize {
        for (i, level) in self.levels.iter().enumerate() {
            if screen_size >= level.min_screen_size {
                return i;
            }
        }
        self.levels.len().saturating_sub(1)
    }

    /// The level to draw given the object's projected size and the level it drew
    /// last frame.
    ///
    /// This is [`level_for_size`](Self::level_for_size) plus hysteresis: a
    /// switch only happens once the size has moved past the current level's
    /// boundary by [`LOD_HYSTERESIS`]. An object oscillating inside that margin
    /// stays put.
    pub fn select(&self, screen_size: f32, current_level: usize) -> usize {
        let n = self.levels.len();
        if n == 0 {
            return 0;
        }
        let current = current_level.min(n - 1);
        let target = self.level_for_size(screen_size);
        if target == current {
            return current;
        }
        if target < current {
            // Want a finer level (object grew). The boundary to cross is the
            // start of the level just finer than the current one.
            let boundary = self.levels[current - 1].min_screen_size;
            if screen_size > boundary * (1.0 + LOD_HYSTERESIS) {
                return target;
            }
        } else {
            // Want a cruder level (object shrank). The boundary is the start of
            // the current level.
            let boundary = self.levels[current].min_screen_size;
            if screen_size < boundary * (1.0 - LOD_HYSTERESIS) {
                return target;
            }
        }
        current
    }
}

/// Projected size of an object, as a fraction of viewport height.
///
/// Takes the object's local-space AABB and model transform plus the camera, and
/// returns how much of the viewport's vertical extent the object's bounding
/// sphere covers. This is the metric LOD thresholds compare against: it folds in
/// distance and field of view, so it is independent of window resolution.
///
/// Assumes a perspective projection (uses the camera field of view). Objects at
/// or behind the near plane are clamped to the near distance.
pub fn projected_screen_size(local_aabb: &Aabb, model: &glam::Mat4, camera: &RenderCamera) -> f32 {
    let world = local_aabb.transformed(model);
    let center = world.center();
    let radius = (world.max - world.min).length() * 0.5;
    if radius <= 0.0 {
        return 0.0;
    }
    let eye = glam::Vec3::from(camera.eye_position);
    let distance = (center - eye).length().max(camera.near);
    let half_fov_tan = (camera.fov * 0.5).tan().max(1e-4);
    radius / (distance * half_fov_tan)
}

/// Registry of LOD groups owned by the renderer. Groups are append-only: a
/// `LodGroupId` stays valid for the life of the renderer.
pub(crate) struct LodGroupStore {
    groups: Vec<LodGroup>,
}

impl LodGroupStore {
    pub fn new() -> Self {
        Self { groups: Vec::new() }
    }

    pub fn insert(&mut self, group: LodGroup) -> LodGroupId {
        let id = LodGroupId(self.groups.len());
        self.groups.push(group);
        id
    }

    pub fn get(&self, id: LodGroupId) -> Option<&LodGroup> {
        self.groups.get(id.0)
    }

    pub fn get_mut(&mut self, id: LodGroupId) -> Option<&mut LodGroup> {
        self.groups.get_mut(id.0)
    }
}

impl crate::resources::ViewportGpuResources {
    /// Group meshes that are already uploaded into a LOD chain.
    ///
    /// `levels` lists the meshes full detail first; `min_screen_sizes` gives the
    /// matching lower screen-size bound for each, as a fraction of viewport
    /// height. The thresholds must strictly decrease.
    ///
    /// All levels must be drawable the same way: they need the same named
    /// attributes and the same deformer attachment, otherwise switching to a
    /// level would silently change or drop coloring, warp, or skinning. That is
    /// checked here so a mismatch fails at registration instead of at render
    /// time.
    ///
    /// # Errors
    ///
    /// - [`ViewportError::LodGroupEmpty`] if `levels` is empty.
    /// - [`ViewportError::LodLevelCountMismatch`] if the two lists differ in
    ///   length.
    /// - [`ViewportError::MeshSlotEmpty`] if a mesh id is not in the store.
    /// - [`ViewportError::LodThresholdsNotDescending`] if a threshold is not
    ///   smaller than the previous one.
    /// - [`ViewportError::LodLevelIncompatible`] if a level's attribute set or
    ///   deform attachment differs from level 0.
    pub fn register_lod_group(
        &mut self,
        levels: &[MeshId],
        min_screen_sizes: &[f32],
    ) -> ViewportResult<LodGroupId> {
        if levels.is_empty() {
            return Err(ViewportError::LodGroupEmpty);
        }
        if levels.len() != min_screen_sizes.len() {
            return Err(ViewportError::LodLevelCountMismatch {
                meshes: levels.len(),
                thresholds: min_screen_sizes.len(),
            });
        }

        for (i, &mesh) in levels.iter().enumerate() {
            if !self.mesh_store.contains(mesh) {
                return Err(ViewportError::MeshSlotEmpty {
                    index: mesh.index(),
                });
            }
            if i > 0 && min_screen_sizes[i] >= min_screen_sizes[i - 1] {
                return Err(ViewportError::LodThresholdsNotDescending { level: i });
            }
        }

        self.validate_lod_compatibility(levels)?;

        let lod_levels = levels
            .iter()
            .zip(min_screen_sizes)
            .map(|(&mesh, &size)| LodLevel::new(mesh, size))
            .collect();

        Ok(self.lod_groups.insert(LodGroup {
            levels: lod_levels,
            transition: LodTransition::Discrete,
            cull_below: None,
        }))
    }

    /// Look up a registered LOD group. The per-frame resolve pass uses this to
    /// turn a group id into the mesh for the chosen level.
    #[allow(dead_code)]
    pub(crate) fn lod_group(&self, id: LodGroupId) -> Option<&LodGroup> {
        self.lod_groups.get(id)
    }

    /// Set the screen size, as a fraction of viewport height, below which
    /// objects in this group stop drawing. `None` disables size culling.
    ///
    /// Applies to both `SceneRenderItem`s and individual `MeshInstanceItem`
    /// instances that use the group.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::LodGroupNotFound`] if `id` is not registered.
    pub fn set_lod_cull_below(
        &mut self,
        id: LodGroupId,
        cull_below: Option<f32>,
    ) -> ViewportResult<()> {
        match self.lod_groups.get_mut(id) {
            Some(group) => {
                group.cull_below = cull_below;
                Ok(())
            }
            None => Err(ViewportError::LodGroupNotFound { index: id.index() }),
        }
    }

    /// Check that every level draws the same way as level 0: same named
    /// attributes and same deformer attachment.
    fn validate_lod_compatibility(&self, levels: &[MeshId]) -> ViewportResult<()> {
        let base = levels[0];
        let base_attrs = self.mesh_attribute_names(base);
        let base_deformed = self.deform.meshes.contains_key(&base);

        for (i, &mesh) in levels.iter().enumerate().skip(1) {
            let attrs = self.mesh_attribute_names(mesh);
            if let Some(missing) = base_attrs.iter().find(|a| !attrs.contains(*a)) {
                return Err(ViewportError::LodLevelIncompatible {
                    level: i,
                    reason: format!("missing attribute '{missing}' present on level 0"),
                });
            }
            if let Some(extra) = attrs.iter().find(|a| !base_attrs.contains(*a)) {
                return Err(ViewportError::LodLevelIncompatible {
                    level: i,
                    reason: format!("has attribute '{extra}' absent from level 0"),
                });
            }
            if self.deform.meshes.contains_key(&mesh) != base_deformed {
                let reason = if base_deformed {
                    "level 0 has deformer data attached but this level does not"
                } else {
                    "this level has deformer data attached but level 0 does not"
                };
                return Err(ViewportError::LodLevelIncompatible {
                    level: i,
                    reason: reason.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Names of every attribute that drives a draw-time lookup on a mesh: scalar
    /// vertex/face attributes, face colours, and vector attributes. Empty if the
    /// mesh slot is gone.
    fn mesh_attribute_names(&self, mesh: MeshId) -> std::collections::BTreeSet<String> {
        let mut names = std::collections::BTreeSet::new();
        if let Some(m) = self.mesh_store.get(mesh) {
            names.extend(m.attribute_buffers.keys().cloned());
            names.extend(m.face_attribute_buffers.keys().cloned());
            names.extend(m.face_colour_buffers.keys().cloned());
            names.extend(m.vector_attribute_buffers.keys().cloned());
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(levels: Vec<LodLevel>, cull_below: Option<f32>) -> LodGroup {
        LodGroup {
            levels,
            transition: LodTransition::Discrete,
            cull_below,
        }
    }

    fn lvl(min_screen_size: f32) -> LodLevel {
        // Mesh id is irrelevant to the selection math.
        LodLevel::new(MeshId::INVALID, min_screen_size)
    }

    #[test]
    fn projected_size_shrinks_with_distance() {
        let aabb = Aabb {
            min: glam::Vec3::splat(-0.5),
            max: glam::Vec3::splat(0.5),
        };
        let mut camera = RenderCamera::default();
        camera.eye_position = [0.0, 0.0, 0.0];
        camera.fov = std::f32::consts::FRAC_PI_4;

        let near = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -5.0));
        let far = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -50.0));
        let near_size = projected_screen_size(&aabb, &near, &camera);
        let far_size = projected_screen_size(&aabb, &far, &camera);

        assert!(near_size > far_size);
        // Ten times the distance is roughly a tenth the size.
        assert!((near_size / far_size - 10.0).abs() < 0.5);
    }

    #[test]
    fn projected_size_shrinks_with_wider_fov() {
        let aabb = Aabb {
            min: glam::Vec3::splat(-0.5),
            max: glam::Vec3::splat(0.5),
        };
        let model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -10.0));
        let mut narrow = RenderCamera::default();
        narrow.eye_position = [0.0, 0.0, 0.0];
        narrow.fov = std::f32::consts::FRAC_PI_4;
        let mut wide = narrow.clone();
        wide.fov = std::f32::consts::FRAC_PI_2;

        assert!(
            projected_screen_size(&aabb, &model, &narrow)
                > projected_screen_size(&aabb, &model, &wide)
        );
    }

    #[test]
    fn level_for_size_picks_finest_cleared() {
        let g = group(vec![lvl(0.5), lvl(0.2), lvl(0.0)], None);
        assert_eq!(g.level_for_size(0.9), 0);
        assert_eq!(g.level_for_size(0.5), 0);
        assert_eq!(g.level_for_size(0.3), 1);
        assert_eq!(g.level_for_size(0.05), 2);
    }

    #[test]
    fn select_holds_level_inside_hysteresis_band() {
        let g = group(vec![lvl(0.5), lvl(0.2), lvl(0.0)], None);
        // Sitting just under the 0.5 boundary: from level 0 we should not yet
        // drop, because we have not fallen past 0.5 * (1 - 0.1) = 0.45.
        assert_eq!(g.select(0.48, 0), 0);
        // Coming up from level 1, we should not jump to 0 until past
        // 0.5 * (1 + 0.1) = 0.55.
        assert_eq!(g.select(0.52, 1), 1);
        // Clearly past the margin in each direction.
        assert_eq!(g.select(0.40, 0), 1);
        assert_eq!(g.select(0.60, 1), 0);
    }

    #[test]
    fn select_can_jump_multiple_levels() {
        let g = group(vec![lvl(0.5), lvl(0.2), lvl(0.0)], None);
        // A big shrink from level 0 lands directly on the crudest level.
        assert_eq!(g.select(0.01, 0), 2);
    }

    #[test]
    fn should_cull_respects_threshold() {
        let g = group(vec![lvl(0.5), lvl(0.0)], Some(0.05));
        assert!(g.should_cull(0.01));
        assert!(!g.should_cull(0.10));
        let no_cull = group(vec![lvl(0.5), lvl(0.0)], None);
        assert!(!no_cull.should_cull(0.0));
    }
}

#[cfg(test)]
mod registration_tests {
    use super::*;
    use crate::ViewportGpuResources;
    use crate::geometry::primitives;
    use crate::resources::{AttributeData, MeshData};

    fn try_make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn with_attr(mut data: MeshData, name: &str) -> MeshData {
        let count = data.positions.len();
        data.attributes
            .insert(name.to_string(), AttributeData::Vertex(vec![0.0; count]));
        data
    }

    /// Upload each level mesh, then register the group: the path a consumer
    /// takes, made one call for the tests.
    fn register(
        res: &mut ViewportGpuResources,
        device: &wgpu::Device,
        levels: &[(MeshData, f32)],
    ) -> ViewportResult<LodGroupId> {
        let mut ids = Vec::with_capacity(levels.len());
        let mut sizes = Vec::with_capacity(levels.len());
        for (data, size) in levels {
            ids.push(res.upload_mesh_data(device, data)?);
            sizes.push(*size);
        }
        res.register_lod_group(&ids, &sizes)
    }

    #[test]
    fn register_lod_group_round_trips() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut res = ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = register(
            &mut res,
            &device,
            &[
                (primitives::icosphere(1.0, 3), 0.5),
                (primitives::icosphere(1.0, 1), 0.2),
                (primitives::icosphere(1.0, 0), 0.0),
            ],
        )
        .expect("group should register");
        let group = res.lod_group(id).expect("group present");
        assert_eq!(group.level_count(), 3);
    }

    #[test]
    fn thresholds_must_descend() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut res = ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let err = register(
            &mut res,
            &device,
            &[(primitives::cube(1.0), 0.2), (primitives::cube(1.0), 0.5)],
        );
        assert!(matches!(
            err,
            Err(ViewportError::LodThresholdsNotDescending { level: 1 })
        ));
    }

    #[test]
    fn mismatched_attributes_are_rejected() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut res = ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let err = register(
            &mut res,
            &device,
            &[
                (with_attr(primitives::cube(1.0), "temperature"), 0.5),
                (primitives::cube(1.0), 0.0),
            ],
        );
        assert!(matches!(
            err,
            Err(ViewportError::LodLevelIncompatible { level: 1, .. })
        ));
    }

    #[test]
    fn empty_group_is_rejected() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut res = ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let err = res.register_lod_group(&[], &[]);
        assert!(matches!(err, Err(ViewportError::LodGroupEmpty)));
    }
}
