//! The showcase registry. Add a numbered file here and push it in `all()`.
//!
//! Files are numbered for ordering (`01_objects.rs`, ...). Module identifiers
//! cannot start with a digit, so each is attached with `#[path]`.

#[path = "03_materials.rs"]
pub mod materials;
#[path = "01_objects.rs"]
pub mod objects;
#[path = "02_overlays.rs"]
pub mod overlays;
#[path = "04_picking.rs"]
pub mod picking;

use crate::showcase::Showcase;

/// Every showcase, in selector order.
pub fn all() -> Vec<Box<dyn Showcase>> {
    vec![
        Box::new(objects::ObjectsShowcase::new()),
        Box::new(overlays::OverlaysShowcase::new()),
        Box::new(materials::MaterialsShowcase::new()),
        Box::new(picking::PickingShowcase::new()),
    ]
}
