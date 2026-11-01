//! The view-depth range the directional shadow cascades cover, fit to the
//! mesh instances' bounds.
//!
//! The instance bounds are bucketed into a coarse grid of cells, each keeping
//! the union of its casters and of all its instances, so far-apart groups of
//! instances do not merge into one box spanning the empty space between
//! them. The cells are rebuilt only when the instance bounds change; each
//! frame sweeps the caster cells away from the light, clips them to the
//! receiver cells and the view frustum, and takes the depth span of what is
//! left.

use crate::renderer::RenderCamera;
use crate::resources::InstanceAabb;

/// Grid cells per axis the instance bounds are bucketed into.
const GRID: usize = 4;

/// World-space axis-aligned bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) min: glam::Vec3,
    pub(crate) max: glam::Vec3,
}

impl Bounds {
    fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn intersection(self, other: Self) -> Option<Self> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        min.cmple(max).all().then_some(Self { min, max })
    }

    pub(crate) fn centre(self) -> glam::Vec3 {
        (self.min + self.max) * 0.5
    }

    fn contains(self, p: glam::Vec3) -> bool {
        self.min.cmple(p).all() && p.cmple(self.max).all()
    }

    fn corners(self) -> [glam::Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [
            glam::Vec3::new(a.x, a.y, a.z),
            glam::Vec3::new(b.x, a.y, a.z),
            glam::Vec3::new(b.x, b.y, a.z),
            glam::Vec3::new(a.x, b.y, a.z),
            glam::Vec3::new(a.x, a.y, b.z),
            glam::Vec3::new(b.x, a.y, b.z),
            glam::Vec3::new(b.x, b.y, b.z),
            glam::Vec3::new(a.x, b.y, b.z),
        ]
    }

    /// Half the widest side of these bounds seen from a directional light
    /// shining along `-light_dir`, in the light basis `compute_shadow_matrix`
    /// uses.
    pub(crate) fn light_space_half_extent(self, light_dir: glam::Vec3) -> f32 {
        let dir = light_dir.normalize();
        let up = if dir.z.abs() > 0.99 {
            glam::Vec3::X
        } else {
            glam::Vec3::Z
        };
        let view = glam::Mat4::look_at_rh(dir, glam::Vec3::ZERO, up);
        let (mut lo, mut hi) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
        for c in self.corners() {
            let p = view.transform_point3(c);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        ((hi - lo) * 0.5).max_element().max(1e-3)
    }
}

/// One grid cell's instances.
#[derive(Clone, Copy, Debug)]
struct Cell {
    /// Union of the cell's instances that cast shadows.
    casters: Option<Bounds>,
    /// Union of all the cell's instances: the receivers.
    all: Bounds,
}

/// The caster and scene bounds of the mesh instances, cached against
/// `InstancingState::aabbs_gen` so a frame whose instance bounds did not
/// change reuses them.
#[derive(Default)]
pub(crate) struct CascadeBounds {
    key: Option<u64>,
    /// Union of the instances that cast shadows.
    pub(crate) casters: Option<Bounds>,
    /// Union of every instance.
    pub(crate) scene: Option<Bounds>,
    cells: Vec<Cell>,
}

impl CascadeBounds {
    /// Rebuild the bounds from `aabbs` unless they were built at
    /// `generation`. Returns whether they were rebuilt.
    pub(crate) fn refresh(&mut self, generation: u64, aabbs: &[InstanceAabb]) -> bool {
        if self.key == Some(generation) {
            return false;
        }
        self.key = Some(generation);
        self.casters = None;
        self.scene = None;
        self.cells.clear();
        let finite = |a: &&InstanceAabb| {
            glam::Vec3::from(a.min).is_finite() && glam::Vec3::from(a.max).is_finite()
        };
        let bounds = |a: &InstanceAabb| Bounds {
            min: a.min.into(),
            max: a.max.into(),
        };
        for a in aabbs.iter().filter(finite) {
            let b = bounds(a);
            self.scene = Some(self.scene.map_or(b, |s| s.union(b)));
            if a.cast_shadows != 0 {
                self.casters = Some(self.casters.map_or(b, |s| s.union(b)));
            }
        }
        let Some(scene) = self.scene else {
            return true;
        };
        let mut grid: [Option<Cell>; GRID * GRID * GRID] = [None; GRID * GRID * GRID];
        let cell_size = ((scene.max - scene.min) / GRID as f32).max(glam::Vec3::splat(1e-6));
        for a in aabbs.iter().filter(finite) {
            let b = bounds(a);
            let c = ((b.centre() - scene.min) / cell_size)
                .floor()
                .clamp(glam::Vec3::ZERO, glam::Vec3::splat((GRID - 1) as f32));
            let i = c.x as usize + GRID * (c.y as usize + GRID * c.z as usize);
            let caster = (a.cast_shadows != 0).then_some(b);
            grid[i] = Some(match grid[i] {
                None => Cell {
                    casters: caster,
                    all: b,
                },
                Some(cell) => Cell {
                    casters: match (cell.casters, caster) {
                        (Some(x), Some(y)) => Some(x.union(y)),
                        (x, y) => x.or(y),
                    },
                    all: cell.all.union(b),
                },
            });
        }
        self.cells.extend(grid.into_iter().flatten());
        true
    }
}

/// The view-depth range the cascades cover when there is nothing to fit
/// them to. `typical` stands in for the orbit radius the formula was first
/// written against: the log-space midpoint of the depth range.
pub(super) fn fallback_range(camera_near: f32, camera_far: f32, extent: f32) -> (f32, f32) {
    let typical = (camera_near.max(1e-6) * camera_far).sqrt();
    let near = (typical * 0.1).max(camera_near);
    let far = (typical * 1.5).max(extent).min(camera_far);
    (near, far)
}

/// The view-depth range in which a visible fragment can be in a cast shadow.
///
/// Casters shadow what lies behind them along the light, so each cell's
/// casters are swept away from the light (`light_dir` points towards it) far
/// enough to cross the scene, or to reach the ground plane at `ground` when
/// one is drawn, and clipped to each cell's receivers and to the ground
/// plane. The range is the depth span along the view of the parts of those
/// regions inside the view frustum, clamped to `[2 near, far]` of the camera.
/// `None` when no part of them is visible.
pub(super) fn fitted_range(
    camera: &RenderCamera,
    bounds: &CascadeBounds,
    light_dir: glam::Vec3,
    ground: Option<f32>,
) -> Option<(f32, f32)> {
    let scene = bounds.scene?;
    let (camera_near, camera_far) = camera.depth_range();
    let dir = light_dir.normalize_or_zero();
    let mut sweep = (scene.max - scene.min).length();
    let ground_plane = ground.map(|height| Bounds {
        min: glam::Vec3::new(f32::MIN, f32::MIN, height),
        max: glam::Vec3::new(f32::MAX, f32::MAX, height),
    });
    if let (Some(height), Some(casters)) = (ground, bounds.casters) {
        if dir.z > 1e-3 {
            sweep = sweep.max((casters.max.z - height) / dir.z);
        }
    }
    let sweep = sweep.min(camera_far * 2.0);

    let frustum = camera.culling_frustum();
    let p = &frustum.planes;
    let planes = [0, 1, 2, 3].map(|i| (p[i].normal, p[i].d));
    let eye = camera.eye_position().as_vec3();
    let mut span = (f32::MAX, f32::MIN);
    for casters in bounds.cells.iter().filter_map(|c| c.casters) {
        let shifted = Bounds {
            min: casters.min - dir * sweep,
            max: casters.max - dir * sweep,
        };
        let swept = casters.union(shifted);
        let receivers = bounds.cells.iter().map(|c| c.all).chain(ground_plane);
        for region in receivers.filter_map(|r| swept.intersection(r)) {
            visible_depth_span(camera, &planes, region, &mut span);
            if region.contains(eye) {
                span.0 = span.0.min(0.0);
            }
        }
    }
    let (lo, hi) = span;
    let near_limit = camera_near * 2.0;
    if lo > hi || hi <= near_limit || lo >= camera_far {
        return None;
    }
    let near = lo.max(near_limit);
    let far = hi.min(camera_far);
    (far > near * 1.001).then_some((near, far))
}

impl CascadeBounds {
    /// The highest light-view z of the casters inside a cascade's light-space
    /// `rect` (`[min_x, min_y, max_x, max_y]` in `light_view`), the casters
    /// that can shadow it. Each grid cell's caster box is clipped to the
    /// rect, so a large caster only counts where it overlaps. `None` with no
    /// caster in the rect.
    pub(super) fn caster_max_z(&self, light_view: glam::Mat4, rect: [f32; 4]) -> Option<f32> {
        let planes = [
            (glam::Vec3::X, -rect[0]),
            (glam::Vec3::Y, -rect[1]),
            (-glam::Vec3::X, rect[2]),
            (-glam::Vec3::Y, rect[3]),
        ];
        let mut top = None::<f32>;
        for casters in self.cells.iter().filter_map(|c| c.casters) {
            let corners = casters.corners().map(|c| light_view.transform_point3(c));
            clip_box_faces(&corners, &planes, |p| {
                top = Some(top.map_or(p.z, |t| t.max(p.z)));
            });
        }
        top
    }
}

/// Widen `span` by the view depths of `region`'s part inside the frustum
/// side `planes`: each face of the box clipped to the planes.
fn visible_depth_span(
    camera: &RenderCamera,
    planes: &[(glam::Vec3, f32)],
    region: Bounds,
    span: &mut (f32, f32),
) {
    clip_box_faces(&region.corners(), planes, |p| {
        let depth = -camera.view().transform_point3(p).z;
        span.0 = span.0.min(depth);
        span.1 = span.1.max(depth);
    });
}

/// Clip each face of the box with `corners` (in [`Bounds::corners`] order)
/// to the half-spaces `normal . p + d >= 0` of `planes`, and hand every
/// vertex of what is left to `visit`.
fn clip_box_faces(
    corners: &[glam::Vec3; 8],
    planes: &[(glam::Vec3, f32)],
    mut visit: impl FnMut(glam::Vec3),
) {
    const FACES: [[usize; 4]; 6] = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    // A quad clipped by four planes keeps at most eight vertices.
    const MAX: usize = 12;
    for face in FACES {
        let mut poly = [glam::Vec3::ZERO; MAX];
        let mut n = 4;
        for (slot, &i) in poly.iter_mut().zip(&face) {
            *slot = corners[i];
        }
        for &(normal, d) in planes {
            let dist = |p: glam::Vec3| normal.dot(p) + d;
            let mut out = [glam::Vec3::ZERO; MAX];
            let mut m = 0;
            for k in 0..n {
                let (a, b) = (poly[k], poly[(k + 1) % n]);
                let (da, db) = (dist(a), dist(b));
                if da >= 0.0 && m < MAX {
                    out[m] = a;
                    m += 1;
                }
                if (da >= 0.0) != (db >= 0.0) && m < MAX {
                    out[m] = a + (b - a) * (da / (da - db));
                    m += 1;
                }
            }
            poly = out;
            n = m;
            if n == 0 {
                break;
            }
        }
        for p in &poly[..n] {
            visit(*p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn aabb(min: [f32; 3], max: [f32; 3], cast: bool) -> InstanceAabb {
        InstanceAabb {
            min,
            batch_index: 0,
            max,
            cast_shadows: cast as u32,
        }
    }

    fn bounds_of(aabbs: &[InstanceAabb]) -> CascadeBounds {
        let mut b = CascadeBounds::default();
        b.refresh(0, aabbs);
        b
    }

    /// Eye at the origin looking along +Y, Z up.
    fn camera() -> RenderCamera {
        let cam = crate::camera::Camera::orbit(
            glam::DVec3::Y * 0.01,
            0.01,
            glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        );
        RenderCamera::from_camera(&cam)
    }

    #[test]
    fn range_spans_the_casters_and_their_shadows_within_the_scene() {
        let b = bounds_of(&[
            aabb([-5.0, 0.0, 0.0], [5.0, 40.0, 0.1], false),
            aabb([-1.0, 9.0, 0.0], [1.0, 11.0, 2.0], true),
        ]);
        // Light from straight overhead: shadows fall under the casters.
        let (near, far) = fitted_range(&camera(), &b, Vec3::Z, None).unwrap();
        assert!(
            (near - 9.0).abs() < 1e-3 && (far - 11.0).abs() < 1e-3,
            "{near} {far}"
        );
        // A low light from -Y throws them on along +Y, as far as the floor goes.
        let low = Vec3::new(0.0, -1.0, 0.2);
        let (near, far) = fitted_range(&camera(), &b, low, None).unwrap();
        assert!(
            (near - 9.0).abs() < 1e-3 && (far - 40.0).abs() < 1e-3,
            "{near} {far}"
        );
    }

    #[test]
    fn a_ground_plane_receives_shadows_past_the_scene() {
        let b = bounds_of(&[aabb([-1.0, 9.0, 0.0], [1.0, 11.0, 2.0], true)]);
        let low = Vec3::new(0.0, -1.0, 0.2).normalize();
        let (_, far) = fitted_range(&camera(), &b, low, None).unwrap();
        assert!((far - 11.0).abs() < 1e-3, "{far}");
        // The top of the caster throws its shadow 2 / 0.2 = 10 m along +Y.
        let (_, far) = fitted_range(&camera(), &b, low, Some(0.0)).unwrap();
        assert!((far - 21.0).abs() < 1e-2, "{far}");
    }

    #[test]
    fn casters_behind_the_camera_fit_nothing() {
        let b = bounds_of(&[aabb([-1.0, -11.0, 0.0], [1.0, -9.0, 2.0], true)]);
        assert!(fitted_range(&camera(), &b, Vec3::Z, None).is_none());
    }

    #[test]
    fn far_away_instances_do_not_stretch_the_range() {
        // A caster in view, and another far off to the side and below, out of
        // view: one box around both would reach hundreds of metres deep.
        let b = bounds_of(&[
            aabb([-1.0, 9.0, 0.0], [1.0, 11.0, 2.0], true),
            aabb([-500.0, -500.0, -500.0], [-499.0, -499.0, -499.0], true),
        ]);
        let (near, far) = fitted_range(&camera(), &b, Vec3::Z, None).unwrap();
        assert!(
            (near - 9.0).abs() < 1e-3 && (far - 11.0).abs() < 1e-3,
            "{near} {far}"
        );
    }

    #[test]
    fn only_the_visible_part_of_a_receiver_counts() {
        // A long wall beside the eye, mostly out of a 45 degree view.
        let b = bounds_of(&[aabb([50.0, -100.0, 0.0], [51.0, 100.0, 2.0], true)]);
        let (near, far) = fitted_range(&camera(), &b, Vec3::Z, None).unwrap();
        assert!(near > 50.0 && far <= 100.0 + 1e-3, "{near} {far}");
    }

    #[test]
    fn bounds_rebuild_once_per_generation() {
        let mut b = CascadeBounds::default();
        assert!(b.refresh(
            0,
            &[
                aabb([0.0; 3], [1.0; 3], true),
                aabb([5.0, 0.0, 0.0], [6.0, 1.0, 1.0], false)
            ]
        ));
        let unit = Bounds {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        };
        assert_eq!(b.casters, Some(unit));
        assert_eq!(b.scene.map(|s| s.max), Some(Vec3::new(6.0, 1.0, 1.0)));
        assert!(!b.refresh(0, &[]));
        assert!(b.refresh(1, &[aabb([5.0, 0.0, 0.0], [6.0, 1.0, 1.0], false)]));
        assert_eq!(b.casters, None);
    }
}
