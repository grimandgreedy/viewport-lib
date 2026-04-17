//! Showcase 7: Normal Maps + AO maps - build method.

use crate::AppState;
use crate::geometry::{
    make_box_with_uvs, make_brick_ao_map, make_brick_normal_map, make_tile_ao_map,
    make_tile_normal_map, make_uv_sphere,
};
use viewport_lib::{Material, MeshId, scene::Scene};

impl AppState {
    /// Build Showcase 7: Normal Maps + AO maps demo.
    ///
    /// Creates a sphere with a procedural brick-like normal map and an AO map,
    /// plus a flat-shaded reference sphere without normal mapping for comparison.
    pub(crate) fn build_nm_scene(&mut self) {
        self.nm_scene = Scene::new();

        // Build sphere mesh with UVs so tangents can be auto-computed.
        let sphere = make_uv_sphere(32, 16, 1.0);
        let sphere_idx = self
            .renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("sphere mesh upload");
        let sphere_id = MeshId::from_index(sphere_idx);

        // Procedural 64x64 normal map: simple bumpy brick pattern.
        let nm_data = make_brick_normal_map(64, 64);
        let nm_id = self
            .renderer
            .resources_mut()
            .upload_normal_map(&self.device, &self.queue, 64, 64, &nm_data)
            .expect("normal map upload");

        // Procedural 64x64 AO map: darker in the mortar gaps of the brick pattern.
        let ao_data = make_brick_ao_map(64, 64);
        let ao_id = self
            .renderer
            .resources_mut()
            .upload_texture(&self.device, &self.queue, 64, 64, &ao_data)
            .expect("ao map upload");

        self.nm_tex_ids = [nm_id, nm_id, ao_id];

        // Left sphere: normal map + AO (PBR).
        let nm_node = self.nm_scene.add_named(
            "Sphere (Normal Map + AO)",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(-1.5, 0.0, 0.0)),
            Material {
                base_color: [0.8, 0.6, 0.4],
                use_pbr: true,
                metallic: 0.0,
                roughness: 0.5,
                normal_map_id: Some(nm_id),
                ao_map_id: Some(ao_id),
                ..Material::default()
            },
        );
        self.nm_node = Some(nm_node);
        self.nm_normal_on = true;
        self.nm_ao_on = true;
        self.nm_clip_enabled = false;

        // Right sphere: plain Blinn-Phong, no normal map (reference).
        self.nm_scene.add_named(
            "Sphere (No Normal Map)",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(1.5, 0.0, 0.0)),
            Material {
                base_color: [0.8, 0.6, 0.4],
                ..Material::default()
            },
        );

        // Cube with tile normal map + AO (stone/concrete tile pattern).
        let box_mesh = make_box_with_uvs(1.4, 1.4, 1.4);
        let box_idx = self
            .renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &box_mesh)
            .expect("box mesh upload");
        let box_id = MeshId::from_index(box_idx);

        let tile_nm_data = make_tile_normal_map(64, 64);
        let tile_nm_id = self
            .renderer
            .resources_mut()
            .upload_normal_map(&self.device, &self.queue, 64, 64, &tile_nm_data)
            .expect("tile normal map upload");

        let tile_ao_data = make_tile_ao_map(64, 64);
        let tile_ao_id = self
            .renderer
            .resources_mut()
            .upload_texture(&self.device, &self.queue, 64, 64, &tile_ao_data)
            .expect("tile ao map upload");

        self.nm_tile_tex_ids = [tile_nm_id, tile_ao_id];

        let cube_node = self.nm_scene.add_named(
            "Cube (Tile Normal Map + AO)",
            Some(box_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 2.2, 0.0)),
            Material {
                base_color: [0.75, 0.78, 0.82],
                use_pbr: true,
                metallic: 0.1,
                roughness: 0.7,
                normal_map_id: Some(tile_nm_id),
                ao_map_id: Some(tile_ao_id),
                ..Material::default()
            },
        );
        self.nm_cube_node = Some(cube_node);

        self.nm_built = true;
    }
}
