//! Uploaded RGBA textures get a full mip chain. Uploads used to create a
//! single mip level, so minified sampling fetched full-resolution texels
//! (texture-cache thrashing on large textured meshes) and aliased.

use viewport_lib_testkit::{Harness, textures};

#[test]
fn uploaded_textures_carry_a_mip_chain() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let size = 256u32;
    let tex = textures::checker(size, 8, [255, 255, 255], [0, 0, 0]);
    let _id = h
        .renderer
        .resources_mut()
        .upload_texture(&h.device, &h.queue, size, size, &tex.rgba)
        .expect("upload");

    // A full RGBA8 chain for 256x256 is 349,524 bytes; a single level is
    // 262,144. The accounting reflects what was allocated.
    let stats = h.renderer.resources().texture_memory_stats();
    let single_level = (size * size * 4) as u64;
    assert!(
        stats.used_bytes > single_level,
        "texture accounting should include the mip chain: got {} bytes for a {} byte base",
        stats.used_bytes,
        single_level
    );
}
