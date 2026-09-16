use crate::scene::material::ItemSettings;

/// One GPU implicit surface draw item submitted via [`SceneFrame::gpu_implicit`].
///
/// Up to 16 [`ImplicitPrimitive`] entries are supported per item.
///
/// # Example
/// ```no_run
/// use viewport_lib::{GpuImplicitItem, GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive};
///
/// let mut prim = ImplicitPrimitive::zeroed();
/// prim.kind   = 1;  // sphere
/// prim.blend  = 0.9;
/// prim.params = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];  // center=origin, radius=1
/// prim.colour  = [1.0, 0.5, 0.2, 1.0].into();
///
/// let mut item = GpuImplicitItem::default();
/// item.primitives    = vec![prim];
/// item.blend_mode    = ImplicitBlendMode::SmoothUnion;
/// item.march_options = GpuImplicitOptions::default();
/// ```
#[non_exhaustive]
#[derive(Clone)]
pub struct GpuImplicitItem {
    /// Primitive descriptors (max 16 entries; excess entries are ignored).
    pub primitives: Vec<ImplicitPrimitive>,
    /// How the primitives are combined.
    pub blend_mode: ImplicitBlendMode,
    /// Ray-march quality settings.
    pub march_options: GpuImplicitOptions,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for GpuImplicitItem {
    fn default() -> Self {
        Self {
            primitives: Vec::new(),
            blend_mode: ImplicitBlendMode::Union,
            march_options: GpuImplicitOptions::default(),
            settings: ItemSettings::default(),
        }
    }
}

/// Primitive descriptor for the GPU implicit SDF.
///
/// The shader evaluates each primitive independently and combines them
/// according to the item's [`ImplicitBlendMode`].
///
/// # Primitive kinds and `params` layout
///
/// | `kind` | Primitive | `params[0..4]`   | `params[4..8]`    |
/// |--------|-----------|-----------------|-------------------|
/// | 1      | Sphere    | cx,cy,cz,radius | unused            |
/// | 2      | Box       | cx,cy,cz,_      | hx,hy,hz,_ (half-extents) |
/// | 3      | Plane     | nx,ny,nz,d      | unused (normal + offset)  |
/// | 4      | Capsule   | ax,ay,az,radius | bx,by,bz,_ (endpoints)   |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImplicitPrimitive {
    /// Primitive type discriminant (1=sphere, 2=box, 3=plane, 4=capsule).
    pub kind: u32,
    /// Smooth-min blend radius used when the item's blend mode is `SmoothUnion`.
    /// Zero produces a hard union.
    pub blend: f32,
    #[doc(hidden)]
    pub _pad: [f32; 2],
    /// Kind-specific parameters, first four floats.
    pub params: [f32; 8],
    /// Linear RGBA colour for this primitive.
    /// Colours are blended by proximity weight at the hit point.
    pub colour: crate::Colour,
}

/// How multiple primitives are combined into a single SDF.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImplicitBlendMode {
    /// Hard min() union (sharp junctions between primitives)
    #[default]
    Union,
    /// Smooth-min union (primitives fuse organically; uses per-primitive `blend` radius)
    SmoothUnion,
    /// Max() intersection (only the region inside all primitives is visible)
    Intersection,
}

/// March configuration for a [`GpuImplicitItem`].
#[derive(Clone, Copy, Debug)]
pub struct GpuImplicitOptions {
    /// Maximum ray-march steps before the ray is considered a miss. Default: 128.
    pub max_steps: u32,
    /// Step-scale applied to the SDF distance each iteration (< 1 improves thin-feature quality).
    /// Default: 0.85.
    pub step_scale: f32,
    /// Distance threshold for a ray-surface hit. Default: 5e-4.
    pub hit_threshold: f32,
    /// Maximum ray length before miss. Default: 40.0.
    pub max_distance: f32,
}

impl Default for GpuImplicitOptions {
    fn default() -> Self {
        Self {
            max_steps: 128,
            step_scale: 0.85,
            hit_threshold: 5e-4,
            max_distance: 40.0,
        }
    }
}

impl ImplicitPrimitive {
    /// Return a zeroed primitive with all fields set to zero.
    pub fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}
