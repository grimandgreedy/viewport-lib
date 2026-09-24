//! How a shader in this crate is assembled.
//!
//! A `.wgsl` file here holds only what the item type owns: its own bind groups,
//! its structs, and its entry points. It never declares the group-0 camera,
//! light or clip bindings, and it never carries an include directive. The
//! shared declarations and helper functions are string constants viewport-lib
//! publishes, and a pipeline concatenates the ones it needs in front of the
//! body:
//!
//! ```ignore
//! let source = scene_shader(&[], wgsl_source!("point_cloud"));
//! ```
//!
//! That is the whole substitute for the renderer's include preprocessor. The
//! helpers a body used to include resolve to published functions:
//! `clip_volume_test` is `viewport_pass_clip_volumes`, the hand-rolled
//! clip-plane loop is `viewport_pass_clip_planes`, and the two together are
//! `viewport_clip_test`.

use viewport_lib::plugin_api::shared_wgsl;

/// The source of a shader that draws into the scene pass: the shared group-0
/// declarations, then any further shared sections the body needs, then the
/// body.
///
/// `extra` names the additional catalogue constants to splice in, in order.
/// Pass `&[shared_wgsl::SHARED_PBR_WGSL]` for a lit pipeline, or an empty
/// slice when the body shades for itself.
pub(crate) fn scene_shader(extra: &[&str], body: &str) -> String {
    let mut out = String::with_capacity(
        shared_wgsl::SHARED_BINDINGS_WGSL.len()
            + extra.iter().map(|s| s.len()).sum::<usize>()
            + body.len()
            + 8,
    );
    out.push_str(shared_wgsl::SHARED_BINDINGS_WGSL);
    for section in extra {
        out.push('\n');
        out.push_str(section);
    }
    out.push('\n');
    out.push_str(body);
    out
}

/// The text of one `.wgsl` file in this crate, by bare file name and without
/// the extension. `build.rs` flattens every shader into `OUT_DIR`, so the name
/// is unique across the crate.
macro_rules! wgsl_source {
    ($name:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/", $name, ".wgsl"))
    };
}
pub(crate) use wgsl_source;
