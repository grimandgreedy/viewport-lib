use crate::resources::{GpuImplicitOptions, ImplicitBlendMode, ImplicitPrimitive};
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
/// prim.colour  = [1.0, 0.5, 0.2, 1.0];
///
/// let mut item = GpuImplicitItem::default();
/// item.primitives    = vec![prim];
/// item.blend_mode    = ImplicitBlendMode::SmoothUnion;
/// item.march_options = GpuImplicitOptions::default();
/// ```
#[non_exhaustive]
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
            blend_mode: crate::resources::ImplicitBlendMode::Union,
            march_options: GpuImplicitOptions::default(),
            settings: ItemSettings::default(),
        }
    }
}
