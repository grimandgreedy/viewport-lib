//! The showcase table: number and title for every showcase, listed in menu
//! order.
//!
//! `main.rs` reads this for the selector, the controls-panel heading, and
//! Ctrl+[ / Ctrl+] cycling, so adding a showcase means adding one row here
//! rather than editing three separate lists.

use crate::ShowcaseMode;

/// One row of the table. `number` is the showcase's number, which matches its
/// `showcase_NN_*.rs` file name; the order of [`SHOWCASES`] drives both the
/// selector and the cycle keys.
pub(crate) struct Entry {
    pub(crate) number: u32,
    pub(crate) title: &'static str,
    pub(crate) mode: ShowcaseMode,
    /// The showcase's stateless handle: every per-frame hook goes through it.
    pub(crate) showcase: &'static dyn crate::Showcase,
}

impl Entry {
    /// The text shown in the selector and as the controls-panel heading.
    pub(crate) fn label(&self) -> String {
        format!("{}: {}", self.number, self.title)
    }
}

const fn entry(
    number: u32,
    title: &'static str,
    mode: ShowcaseMode,
    showcase: &'static dyn crate::Showcase,
) -> Entry {
    Entry {
        number,
        title,
        mode,
        showcase,
    }
}

use ShowcaseMode as M;

/// Every showcase, in the order the selector lists them and the cycle keys
/// walk them.
pub(crate) const SHOWCASES: [Entry; 59] = [
    entry(
        1,
        "Rendering Basics",
        M::Basic,
        &crate::showcase_01_basic::SHOWCASE,
    ),
    entry(
        2,
        "Scene Graph",
        M::SceneGraph,
        &crate::showcase_02_scene_graph::SHOWCASE,
    ),
    entry(
        3,
        "Ground Plane",
        M::GroundPlane,
        &crate::showcase_03_ground_plane::SHOWCASE,
    ),
    entry(
        4,
        "Interaction",
        M::Interaction,
        &crate::showcase_04_interaction::SHOWCASE,
    ),
    entry(
        5,
        "Materials and Visibility",
        M::MaterialsVisibility,
        &crate::showcase_05_materials_and_visibility::SHOWCASE,
    ),
    entry(
        6,
        "Post-Processing",
        M::PostProcess,
        &crate::showcase_06_post_process::SHOWCASE,
    ),
    entry(
        7,
        "Normal Maps",
        M::NormalMaps,
        &crate::showcase_07_normal_maps::SHOWCASE,
    ),
    entry(
        8,
        "Shadows",
        M::Shadows,
        &crate::showcase_08_shadows::SHOWCASE,
    ),
    entry(
        9,
        "Annotations",
        M::Annotation,
        &crate::showcase_09_annotation::SHOWCASE,
    ),
    entry(
        10,
        "Camera Tools",
        M::CameraTools,
        &crate::showcase_10_camera_tools::SHOWCASE,
    ),
    entry(
        11,
        "Lights",
        M::Lights,
        &crate::showcase_11_lights::SHOWCASE,
    ),
    entry(
        12,
        "Scalar Fields",
        M::ScalarFields,
        &crate::showcase_12_scalar_fields::SHOWCASE,
    ),
    entry(
        13,
        "Multi-Viewport",
        M::MultiViewport,
        &crate::showcase_13_multi_viewport::SHOWCASE,
    ),
    entry(
        14,
        "Isolines & Contours",
        M::Isolines,
        &crate::showcase_14_isolines::SHOWCASE,
    ),
    entry(
        15,
        "Point Clouds & Glyphs",
        M::PointClouds,
        &crate::showcase_15_point_clouds::SHOWCASE,
    ),
    entry(
        16,
        "Streamlines & Tubes",
        M::Streamlines,
        &crate::showcase_16_streamlines::SHOWCASE,
    ),
    entry(
        17,
        "Volume & Isosurface",
        M::Volume,
        &crate::showcase_17_volume::SHOWCASE,
    ),
    entry(
        18,
        "Clip Volumes",
        M::ClipVolumes,
        &crate::showcase_18_clip_volumes::SHOWCASE,
    ),
    entry(
        19,
        "Matcap Shading",
        M::Matcap,
        &crate::showcase_19_matcap::SHOWCASE,
    ),
    entry(
        20,
        "Face Attributes",
        M::FaceAttributes,
        &crate::showcase_20_face_attributes::SHOWCASE,
    ),
    entry(
        21,
        "Textures",
        M::Textures,
        &crate::showcase_21_textures::SHOWCASE,
    ),
    entry(
        22,
        "UV Parameterization",
        M::ParamVis,
        &crate::showcase_22_parameterization::SHOWCASE,
    ),
    entry(
        23,
        "Performance",
        M::Performance,
        &crate::showcase_23_performance::SHOWCASE,
    ),
    entry(
        24,
        "Backface Policy",
        M::BackfacePolicy,
        &crate::showcase_24_backface_policy::SHOWCASE,
    ),
    entry(
        25,
        "Surface Vectors",
        M::SurfaceVectors,
        &crate::showcase_25_surface_vectors::SHOWCASE,
    ),
    entry(
        26,
        "Volume Meshes",
        M::VolumeMesh,
        &crate::showcase_26_volume_mesh::SHOWCASE,
    ),
    entry(
        27,
        "Camera Framing & HUD",
        M::Auxiliary,
        &crate::showcase_27_camera_framing::SHOWCASE,
    ),
    entry(
        28,
        "Curve Network Quantities",
        M::CurveNetworkQuantities,
        &crate::showcase_28_curve_network_quantities::SHOWCASE,
    ),
    entry(
        29,
        "Depth-Composited Images",
        M::DepthCompositeImages,
        &crate::showcase_29_depth_composite_images::SHOWCASE,
    ),
    entry(
        30,
        "Implicit Surfaces",
        M::ImplicitSurface,
        &crate::showcase_30_implicit_surface::SHOWCASE,
    ),
    entry(
        31,
        "Sparse Volume Grid",
        M::SparseVolumeGrid,
        &crate::showcase_31_sparse_volume_grid::SHOWCASE,
    ),
    entry(
        32,
        "Extended Quantities",
        M::ExtendedQuantities,
        &crate::showcase_32_extended_quantities::SHOWCASE,
    ),
    entry(
        33,
        "Picking Levels",
        M::PickLevels,
        &crate::showcase_33_picking_levels::SHOWCASE,
    ),
    entry(
        34,
        "Labels",
        M::Labels,
        &crate::showcase_34_labels::SHOWCASE,
    ),
    entry(
        35,
        "Overlay Composition",
        M::Overlay,
        &crate::showcase_35_overlay::SHOWCASE,
    ),
    entry(
        36,
        "Playback Runtime Control",
        M::PlaybackRuntime,
        &crate::showcase_36_playback_runtime::SHOWCASE,
    ),
    entry(
        37,
        "Probe Widgets",
        M::ProbeWidgets,
        &crate::showcase_37_probe_widgets::SHOWCASE,
    ),
    entry(
        38,
        "Surface LIC",
        M::SurfaceLIC,
        &crate::showcase_38_surface_lic::SHOWCASE,
    ),
    entry(
        39,
        "Tensor Glyphs",
        M::TensorGlyphs,
        &crate::showcase_39_tensor_glyphs::SHOWCASE,
    ),
    entry(
        40,
        "GPU Vertex Warp",
        M::VertexWarp,
        &crate::showcase_40_vertex_warp::SHOWCASE,
    ),
    entry(
        41,
        "Sprites & Particles",
        M::Sprites,
        &crate::showcase_41_sprites::SHOWCASE,
    ),
    entry(
        42,
        "Gaussian Splats",
        M::GaussianSplats,
        &crate::showcase_42_gaussian_splats::SHOWCASE,
    ),
    entry(
        43,
        "Scene Runtime",
        M::SceneRuntime,
        &crate::showcase_43_scene_runtime::SHOWCASE,
    ),
    entry(
        44,
        "Debug Draw",
        M::DebugDraw,
        &crate::showcase_44_debug_draw::SHOWCASE,
    ),
    entry(
        45,
        "Skeletal Animation",
        M::SkinnedAnimation,
        &crate::showcase_45_skinned_animation::SHOWCASE,
    ),
    entry(
        46,
        "Decals",
        M::Decals,
        &crate::showcase_46_decals::SHOWCASE,
    ),
    entry(
        47,
        "Lighting Consistency",
        M::LightingConsistency,
        &crate::showcase_47_lighting_consistency::SHOWCASE,
    ),
    entry(
        48,
        "Scatter Volumes",
        M::ScatterVolumes,
        &crate::showcase_48_scatter_volumes::SHOWCASE,
    ),
    entry(
        49,
        "Scene Lights",
        M::SceneLights,
        &crate::showcase_49_scene_lights::SHOWCASE,
    ),
    entry(
        50,
        "GPU Wave (compute plugin)",
        M::GpuWave,
        &crate::showcase_50_gpu_wave::SHOWCASE,
    ),
    entry(
        51,
        "Async Asset Streaming",
        M::AsyncUploads,
        &crate::showcase_51_async_uploads::SHOWCASE,
    ),
    entry(
        52,
        "Level of Detail",
        M::Lod,
        &crate::showcase_52_lod::SHOWCASE,
    ),
    entry(
        53,
        "Vertex Colours & Painting",
        M::VertexColours,
        &crate::showcase_53_vertex_colours::SHOWCASE,
    ),
    entry(
        54,
        "Custom Shading Plugins",
        M::CustomShading,
        &crate::showcase_54_custom_shading::SHOWCASE,
    ),
    entry(
        55,
        "Foreground Composite Pass",
        M::Foreground,
        &crate::showcase_55_foreground_pass::SHOWCASE,
    ),
    entry(
        56,
        "Submesh Materials",
        M::SubmeshMaterials,
        &crate::showcase_56_submesh_materials::SHOWCASE,
    ),
    entry(
        57,
        "Photometric Lighting",
        M::PhotometricLighting,
        &crate::showcase_57_photometric_lighting::SHOWCASE,
    ),
    entry(
        58,
        "Physically-Based Surfaces",
        M::PhysicallyBasedSurfaces,
        &crate::showcase_58_physically_based_surfaces::SHOWCASE,
    ),
    entry(
        59,
        "Vector Art (SVG)",
        M::VectorArt,
        &crate::showcase_59_vector_art::SHOWCASE,
    ),
];

impl ShowcaseMode {
    /// The table row for this mode. Every variant has exactly one row.
    pub(crate) fn entry(self) -> &'static Entry {
        SHOWCASES
            .iter()
            .find(|e| e.mode == self)
            .expect("every ShowcaseMode has a registry entry")
    }

    /// `"23: Performance"`, for the selector and the controls-panel heading.
    pub(crate) fn label(self) -> String {
        self.entry().label()
    }

    /// The stateless handle every per-frame hook is dispatched through.
    pub(crate) fn showcase(self) -> &'static dyn crate::Showcase {
        self.entry().showcase
    }
}
