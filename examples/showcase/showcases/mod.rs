//! The showcase registry. Add a numbered file here and push it in `all()`.
//!
//! Files are numbered for ordering (`1_objects.rs`, ...). Module identifiers
//! cannot start with a digit, so each is attached with `#[path]`.

#[path = "1_objects.rs"]
pub mod objects;
#[path = "2_overlays.rs"]
pub mod overlays;
#[path = "3_materials.rs"]
pub mod materials;

use crate::showcase::Showcase;

/// Every showcase, in selector order.
pub fn all() -> Vec<Box<dyn Showcase>> {
    vec![
        Box::new(objects::ObjectsShowcase::new()),
        Box::new(overlays::OverlaysShowcase::new()),
        Box::new(materials::MaterialsShowcase::new()),
    ]
}
