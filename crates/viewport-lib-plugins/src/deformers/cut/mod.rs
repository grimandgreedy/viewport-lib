//! Remove part of a mesh: a plane, an oriented box, a sphere, or a range of a
//! per-vertex scalar. For section views and thresholds.
//!
//! The cut is a deformer, so it holds in every pass the mesh is drawn in: the
//! removed part is not drawn, casts no shadow, gets no selection outline and is
//! not hit by a GPU pick, which passes through to what is behind. The mesh data
//! is never modified, and the edge of the removed region follows the cut
//! through each triangle rather than dropping whole triangles.
//!
//! ```no_run
//! use viewport_lib_plugins::deformers::cut::{Cut, CutDeformer};
//! # let mut renderer: viewport_lib::renderer::ViewportRenderer = unimplemented!();
//! # let device: &viewport_lib::gpu::Device = unimplemented!();
//! # let queue: &viewport_lib::gpu::Queue = unimplemented!();
//! # let mesh_id: viewport_lib::MeshId = unimplemented!();
//! # let mut item = viewport_lib::SceneRenderItem::default();
//! // Once.
//! let cut = CutDeformer::install(renderer.resources_mut(), device)?;
//!
//! // Keep the part of the mesh above z = 0.5 for one item.
//! cut.set(renderer.resources_mut(), device, queue, mesh_id, 1, &[Cut::plane([0.0, 0.0, 1.0], 0.5)]);
//! item.deform_instance = Some(1);
//! # Ok::<(), viewport_lib::error::ViewportError>(())
//! ```
//!
//! Things to know:
//!
//! - **The item selects the cut.** Cuts are stored per (mesh, instance id), and
//!   an item uses them by setting `deform_instance` to that id. An item that
//!   leaves it at `None` is drawn whole, with no error. Two items sharing a
//!   mesh can be cut differently, or one cut and one not.
//! - **Shapes are in world space** and are tested against the mesh after every
//!   other deformer has moved it.
//! - **A range needs a field** on the mesh: [`set_field`](CutDeformer::set_field)
//!   from values, [`set_field_source`](CutDeformer::set_field_source) from a
//!   buffer you write, or
//!   [`set_field_from_attribute`](CutDeformer::set_field_from_attribute) from a
//!   scalar attribute the mesh was uploaded with. Without one a range keeps
//!   everything.
//! - **Curved edges are approximate on coarse meshes.** The kept value is
//!   computed per vertex and interpolated, so a plane and a box face cut
//!   exactly, while a sphere or a non-linear field is followed as closely as
//!   the triangles allow.
//! - **A cut item draws on its own**, not in an instanced batch, like any item
//!   with deformer data.
//! - **A skinned item that is also cut** uses the same `deform_instance` for
//!   its palette and its cuts.
//! - **CPU picking ignores the cut**: it reads the mesh's own geometry. So does
//!   the per-pixel vertex and edge refinement of a GPU pick.
//! - **Deformers need a capable device**: `install` fails on one created
//!   without `ViewportRenderer::recommended_device_limits`.

use viewport_lib::MeshId;
use viewport_lib::error::ViewportResult;
use viewport_lib::gpu;
use viewport_lib::resources::{DeformStage, DeformerDesc, DeformerId, DeviceResources};

/// The most cuts one item can carry.
pub const MAX_CUTS: usize = 8;

/// The name the deformer registers under.
pub const DEFORMER_NAME: &str = "vpl_cut";

/// Words in one per-instance element: a cut, or the leading count.
const ELEMENT_WORDS: usize = 20;

/// One cut: the region it keeps.
///
/// Every constructor keeps one side; [`flipped`](Self::flipped) keeps the other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cut {
    shape: Shape,
    flipped: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    Plane {
        normal: [f32; 3],
        distance: f32,
    },
    Box {
        centre: [f32; 3],
        axes: [[f32; 3]; 3],
        half_extents: [f32; 3],
    },
    Sphere {
        centre: [f32; 3],
        radius: f32,
    },
    Range {
        min: f32,
        max: f32,
    },
}

impl Cut {
    /// Keeps the side of a plane where `dot(p, normal) >= distance`. `normal`
    /// need not be unit length, but the edge's softness scales with it.
    pub fn plane(normal: [f32; 3], distance: f32) -> Self {
        Self::from(Shape::Plane { normal, distance })
    }

    /// Keeps the inside of an axis-aligned box.
    pub fn aabb(min: [f32; 3], max: [f32; 3]) -> Self {
        let centre = std::array::from_fn(|i| (min[i] + max[i]) * 0.5);
        let half_extents = std::array::from_fn(|i| (max[i] - min[i]).abs() * 0.5);
        Self::oriented_box(
            centre,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            half_extents,
        )
    }

    /// Keeps the inside of a box with the given centre, unit axes and
    /// half-extent along each axis.
    pub fn oriented_box(centre: [f32; 3], axes: [[f32; 3]; 3], half_extents: [f32; 3]) -> Self {
        Self::from(Shape::Box {
            centre,
            axes,
            half_extents,
        })
    }

    /// Keeps the inside of a sphere.
    pub fn sphere(centre: [f32; 3], radius: f32) -> Self {
        Self::from(Shape::Sphere { centre, radius })
    }

    /// Keeps the vertices whose field value is in `min..=max`. Needs a field
    /// on the mesh; see the module docs.
    pub fn range(min: f32, max: f32) -> Self {
        Self::from(Shape::Range { min, max })
    }

    /// The same cut, keeping the other side.
    pub fn flipped(self) -> Self {
        Self {
            flipped: !self.flipped,
            ..self
        }
    }

    fn from(shape: Shape) -> Self {
        Self {
            shape,
            flipped: false,
        }
    }

    fn encode(&self) -> [u32; ELEMENT_WORDS] {
        let mut w = [0u32; ELEMENT_WORDS];
        let mut put = |k: usize, v: f32| w[k] = v.to_bits();
        let kind = match self.shape {
            Shape::Plane { normal, distance } => {
                put(4, normal[0]);
                put(5, normal[1]);
                put(6, normal[2]);
                put(7, distance);
                0
            }
            Shape::Box {
                centre,
                axes,
                half_extents,
            } => {
                put(4, centre[0]);
                put(5, centre[1]);
                put(6, centre[2]);
                for (a, (axis, half)) in axes.iter().zip(half_extents).enumerate() {
                    let k = 8 + a * 4;
                    put(k, axis[0]);
                    put(k + 1, axis[1]);
                    put(k + 2, axis[2]);
                    put(k + 3, half);
                }
                1
            }
            Shape::Sphere { centre, radius } => {
                put(4, centre[0]);
                put(5, centre[1]);
                put(6, centre[2]);
                put(7, radius);
                2
            }
            Shape::Range { min, max } => {
                put(4, min);
                put(5, max);
                3
            }
        };
        w[0] = kind;
        w[1] = self.flipped as u32;
        w
    }
}

/// The cut deformer, registered with a renderer's resources.
///
/// Cheap to copy; every call takes the resources it acts on.
#[derive(Clone, Copy, Debug)]
pub struct CutDeformer {
    id: DeformerId,
}

impl CutDeformer {
    /// Register the deformer, or find it if it is already registered.
    ///
    /// Takes one of the renderer's deformer slots, whatever the number of cut
    /// items. Fails on a device without deformer support, or when every slot
    /// is taken.
    pub fn install(resources: &mut DeviceResources, device: &gpu::Device) -> ViewportResult<Self> {
        if let Some(id) = resources.deformer_id_by_name(DEFORMER_NAME) {
            return Ok(Self { id });
        }
        let id = resources.register_deformer(
            device,
            DeformerDesc {
                name: DEFORMER_NAME,
                stage: DeformStage::WorldSpace,
                // `deform` moves nothing and `keep` runs after every stage, so
                // the cut sees the final position whatever its place here.
                priority: 0,
                wgsl_body: include_str!("cut.wgsl").to_string(),
                per_vertex_stride: 4,
            },
        )?;
        Ok(Self { id })
    }

    /// The registered deformer.
    pub fn deformer_id(&self) -> DeformerId {
        self.id
    }

    /// Cut the items that draw `mesh_id` with `deform_instance == Some(instance)`
    /// by `cuts`, replacing any cuts they had. The item keeps what every cut
    /// keeps. An empty slice keeps everything; [`clear`](Self::clear) also
    /// releases the data.
    ///
    /// # Panics
    ///
    /// When `cuts` holds more than [`MAX_CUTS`].
    pub fn set(
        &self,
        resources: &mut DeviceResources,
        device: &gpu::Device,
        queue: &gpu::Queue,
        mesh_id: MeshId,
        instance: u32,
        cuts: &[Cut],
    ) {
        assert!(
            cuts.len() <= MAX_CUTS,
            "{} cuts given, at most {MAX_CUTS} per item",
            cuts.len()
        );
        let mut words = vec![0u32; ELEMENT_WORDS * (cuts.len() + 1)];
        words[0] = cuts.len() as u32;
        for (i, cut) in cuts.iter().enumerate() {
            let start = ELEMENT_WORDS * (i + 1);
            words[start..start + ELEMENT_WORDS].copy_from_slice(&cut.encode());
        }
        resources.attach_deform_slot_instance(
            device,
            queue,
            mesh_id,
            instance,
            self.id.slot(),
            (ELEMENT_WORDS * 4) as u32,
            bytemuck::cast_slice(&words),
        );
    }

    /// Remove the cuts of `(mesh_id, instance)`, so its items draw whole.
    pub fn clear(
        &self,
        resources: &mut DeviceResources,
        device: &gpu::Device,
        queue: &gpu::Queue,
        mesh_id: MeshId,
        instance: u32,
    ) {
        resources.detach_deform_slot_instance(device, queue, mesh_id, instance, self.id.slot());
    }

    /// The scalar field a [`Cut::range`] tests, one value per vertex of
    /// `mesh_id`.
    pub fn set_field(
        &self,
        resources: &mut DeviceResources,
        device: &gpu::Device,
        mesh_id: MeshId,
        values: &[f32],
    ) {
        resources.attach_deform_slot(
            device,
            mesh_id,
            self.id.slot(),
            4,
            bytemuck::cast_slice(values),
        );
    }

    /// Read the field from a buffer you own and write, one `f32` per vertex.
    /// It is copied each frame before the mesh is drawn, so whatever your
    /// compute last wrote is what cuts. The buffer needs `COPY_SRC`.
    pub fn set_field_source(
        &self,
        resources: &mut DeviceResources,
        device: &gpu::Device,
        mesh_id: MeshId,
        buffer: gpu::Buffer,
    ) -> ViewportResult<()> {
        resources.set_deform_slot_source_buffer(device, mesh_id, self.id.slot(), buffer, 4)
    }

    /// Read the field from a scalar attribute `mesh_id` was uploaded with,
    /// with no second upload. A later `replace_attribute` reaches the cut.
    pub fn set_field_from_attribute(
        &self,
        resources: &mut DeviceResources,
        device: &gpu::Device,
        mesh_id: MeshId,
        name: &str,
    ) -> ViewportResult<()> {
        resources.set_deform_slot_source_attribute(device, mesh_id, self.id.slot(), name)
    }

    /// Remove the field of `mesh_id`, from whichever source it came.
    pub fn clear_field(
        &self,
        resources: &mut DeviceResources,
        device: &gpu::Device,
        mesh_id: MeshId,
    ) {
        resources.clear_deform_slot_source(device, mesh_id, self.id.slot());
        resources.detach_deform_slot(device, mesh_id, self.id.slot());
    }
}
