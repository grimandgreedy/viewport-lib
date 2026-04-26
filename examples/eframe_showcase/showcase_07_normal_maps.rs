//! Showcase 7: Normal Maps + AO Maps : build method.
//!
//! Demonstrates normal-mapped and AO-mapped surfaces on a variety of shapes:
//! a sphere, a cube, and a flat wall panel, all on a tiled ground plane.
//! Toggles let you enable/disable normal maps and AO maps across all objects
//! to see the difference they make.

use crate::App;
use crate::geometry::{
    make_box_with_uvs, make_brick_ao_map, make_brick_normal_map, make_tile_ao_map,
    make_tile_normal_map, make_uv_sphere,
};
use viewport_lib::{BackfacePolicy, Material, ViewportRenderer, scene::Scene};

impl App {
    /// Build Showcase 7: Normal Maps + AO Maps demo.
    pub(crate) fn build_nm_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.nm_scene = Scene::new();
        self.nm_mapped_nodes.clear();

        // Upload maps at 128x128 for better detail.
        let brick_nm_data = make_brick_normal_map(128, 128);
        let brick_nm_id = renderer
            .resources_mut()
            .upload_normal_map(&self.device, &self.queue, 128, 128, &brick_nm_data)
            .expect("brick normal map upload");

        let brick_ao_data = make_brick_ao_map(128, 128);
        let brick_ao_id = renderer
            .resources_mut()
            .upload_texture(&self.device, &self.queue, 128, 128, &brick_ao_data)
            .expect("brick ao map upload");

        let tile_nm_data = make_tile_normal_map(128, 128);
        let tile_nm_id = renderer
            .resources_mut()
            .upload_normal_map(&self.device, &self.queue, 128, 128, &tile_nm_data)
            .expect("tile normal map upload");

        let tile_ao_data = make_tile_ao_map(128, 128);
        let tile_ao_id = renderer
            .resources_mut()
            .upload_texture(&self.device, &self.queue, 128, 128, &tile_ao_data)
            .expect("tile ao map upload");

        // --- Meshes ---
        let sphere = make_uv_sphere(48, 24, 1.0);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("sphere mesh upload");

        let cube_mesh = make_box_with_uvs(1.6, 1.6, 1.6);
        let cube_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &cube_mesh)
            .expect("nm cube mesh upload");

        // Flat wall panel to show brick normal map on a flat surface.
        let wall_mesh = make_box_with_uvs(4.0, 0.3, 3.0);
        let wall_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &wall_mesh)
            .expect("wall mesh upload");

        // Ground plane with tile pattern.
        let ground_mesh = make_box_with_uvs(12.0, 12.0, 0.15);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("ground mesh upload");

        // --- Scene objects ---

        // Ground : tiled normal + AO.
        let ground_node = self.nm_scene.add_named(
            "Ground (Tile)",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.075)),
            {
                let mut mat = Material::pbr([0.85, 0.85, 0.85], 0.0, 0.85);
                mat.normal_map_id = Some(tile_nm_id);
                mat.ao_map_id = Some(tile_ao_id);
                mat.backface_policy = BackfacePolicy::Identical;
                mat
            },
        );
        self.nm_mapped_nodes
            .push((ground_node, tile_nm_id, tile_ao_id));

        // Brick sphere : left.
        let sphere_node = self.nm_scene.add_named(
            "Sphere (Brick)",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 1.0)),
            {
                let mut mat = Material::pbr([0.9, 0.88, 0.85], 0.0, 0.5);
                mat.normal_map_id = Some(brick_nm_id);
                mat.ao_map_id = Some(brick_ao_id);
                mat.backface_policy = BackfacePolicy::Identical;
                mat
            },
        );
        self.nm_mapped_nodes
            .push((sphere_node, brick_nm_id, brick_ao_id));

        // Tile cube : right.
        let cube_node = self.nm_scene.add_named(
            "Cube (Tile)",
            Some(cube_id),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.0, 0.8)),
            {
                let mut mat = Material::pbr([0.85, 0.87, 0.9], 0.1, 0.6);
                mat.normal_map_id = Some(tile_nm_id);
                mat.ao_map_id = Some(tile_ao_id);
                mat.backface_policy = BackfacePolicy::Identical;
                mat
            },
        );
        self.nm_mapped_nodes
            .push((cube_node, tile_nm_id, tile_ao_id));

        // Brick wall panel : behind.
        let wall_node = self.nm_scene.add_named(
            "Wall (Brick)",
            Some(wall_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, -2.0, 3.5)),
            {
                let mut mat = Material::pbr([0.92, 0.9, 0.87], 0.0, 0.7);
                mat.normal_map_id = Some(brick_nm_id);
                mat.ao_map_id = Some(brick_ao_id);
                mat.backface_policy = BackfacePolicy::Identical;
                mat
            },
        );
        self.nm_mapped_nodes
            .push((wall_node, brick_nm_id, brick_ao_id));

        // Plain sphere for comparison : separate mesh upload to avoid per-object
        // uniform clobbering (two items sharing a mesh_index would overwrite
        // each other's model matrix and texture bindings in the uniform buffer).
        let plain_sphere = make_uv_sphere(48, 24, 1.0);
        let plain_sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &plain_sphere)
            .expect("plain sphere mesh upload");
        self.nm_scene.add_named(
            "Sphere (No Maps)",
            Some(plain_sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(-3.0, 2.5, 1.0)),
            {
                let mut mat = Material::pbr([0.9, 0.88, 0.85], 0.0, 0.5);
                mat.backface_policy = BackfacePolicy::Identical;
                mat
            },
        );

        self.nm_normal_on = true;
        self.nm_ao_on = true;
        self.nm_clip_enabled = false;
        self.nm_built = true;
    }
}
