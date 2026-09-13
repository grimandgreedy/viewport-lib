//! The showcase table: number, title, and menu group for every showcase, listed
//! in menu order.
//!
//! `main.rs` reads this for the selector dropdown, the controls-panel heading,
//! and Ctrl+[ / Ctrl+] cycling, so adding a showcase means adding one row here
//! rather than editing three separate lists.

use crate::ShowcaseMode;

/// Menu grouping for the selector. Showcases are listed under these headings so
/// related demos sit together: someone looking for "how do I draw a vector
/// field" sees the four candidates in one place instead of hunting the whole
/// list.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Group {
    GettingStarted,
    CamerasAndInteraction,
    LightingAndShading,
    MaterialsAndTextures,
    ScalarAndVectorQuantities,
    VolumesAndImplicitSurfaces,
    AnnotationAndOverlay,
    ParticlesAndSprites,
    RuntimeAndAnimation,
    PickingAndPerformance,
}

impl Group {
    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::GettingStarted => "Getting started",
            Self::CamerasAndInteraction => "Cameras & interaction",
            Self::LightingAndShading => "Lighting & shading",
            Self::MaterialsAndTextures => "Materials, textures & decals",
            Self::ScalarAndVectorQuantities => "Scalar & vector quantities",
            Self::VolumesAndImplicitSurfaces => "Volumes & implicit surfaces",
            Self::AnnotationAndOverlay => "Annotation & overlay",
            Self::ParticlesAndSprites => "Particles & sprites",
            Self::RuntimeAndAnimation => "Runtime, animation & GPU plugins",
            Self::PickingAndPerformance => "Picking & performance",
        }
    }
}

/// One row of the table. `number` is the showcase's number, which matches its
/// `showcase_NN_*.rs` file name; it is display-only, and the order of
/// [`SHOWCASES`] is what drives the menu and the cycle keys.
pub(crate) struct Entry {
    pub(crate) number: u32,
    pub(crate) title: &'static str,
    pub(crate) group: Group,
    pub(crate) mode: ShowcaseMode,
}

impl Entry {
    /// The text shown in the selector and as the controls-panel heading.
    pub(crate) fn label(&self) -> String {
        format!("{}: {}", self.number, self.title)
    }
}

const fn entry(number: u32, title: &'static str, group: Group, mode: ShowcaseMode) -> Entry {
    Entry {
        number,
        title,
        group,
        mode,
    }
}

use Group::*;
use ShowcaseMode as M;

/// Every showcase in menu order. Rows of the same group must be contiguous: the
/// selector emits a heading whenever the group changes as it walks this list.
pub(crate) const SHOWCASES: [Entry; 59] = [
    // --- Getting started ---
    entry(1, "Rendering Basics", GettingStarted, M::Basic),
    entry(2, "Scene Graph", GettingStarted, M::SceneGraph),
    entry(
        5,
        "Materials and Visibility",
        GettingStarted,
        M::MaterialsVisibility,
    ),
    entry(22, "UV Parameterization", GettingStarted, M::ParamVis),
    entry(24, "Backface Policy", GettingStarted, M::BackfacePolicy),
    // --- Cameras & interaction ---
    entry(4, "Interaction", CamerasAndInteraction, M::Interaction),
    entry(10, "Camera Tools", CamerasAndInteraction, M::CameraTools),
    entry(
        13,
        "Multi-Viewport",
        CamerasAndInteraction,
        M::MultiViewport,
    ),
    entry(
        27,
        "Camera Framing & HUD",
        CamerasAndInteraction,
        M::Auxiliary,
    ),
    entry(37, "Probe Widgets", CamerasAndInteraction, M::ProbeWidgets),
    // --- Lighting & shading ---
    entry(3, "Ground Plane", LightingAndShading, M::GroundPlane),
    entry(6, "Post-Processing", LightingAndShading, M::PostProcess),
    entry(
        55,
        "Foreground Composite Pass",
        LightingAndShading,
        M::Foreground,
    ),
    entry(7, "Normal Maps", LightingAndShading, M::NormalMaps),
    entry(8, "Shadows", LightingAndShading, M::Shadows),
    entry(11, "Lights", LightingAndShading, M::Lights),
    entry(19, "Matcap Shading", LightingAndShading, M::Matcap),
    entry(
        47,
        "Lighting Consistency",
        LightingAndShading,
        M::LightingConsistency,
    ),
    entry(49, "Scene Lights", LightingAndShading, M::SceneLights),
    entry(
        57,
        "Photometric Lighting",
        LightingAndShading,
        M::PhotometricLighting,
    ),
    // --- Materials, textures & decals ---
    entry(21, "Textures", MaterialsAndTextures, M::Textures),
    entry(46, "Decals", MaterialsAndTextures, M::Decals),
    entry(
        53,
        "Vertex Colours & Painting",
        MaterialsAndTextures,
        M::VertexColours,
    ),
    entry(
        56,
        "Submesh Materials",
        MaterialsAndTextures,
        M::SubmeshMaterials,
    ),
    entry(
        58,
        "Physically-Based Surfaces",
        MaterialsAndTextures,
        M::PhysicallyBasedSurfaces,
    ),
    // --- Scalar & vector quantities ---
    entry(
        12,
        "Scalar Fields",
        ScalarAndVectorQuantities,
        M::ScalarFields,
    ),
    entry(
        14,
        "Isolines & Contours",
        ScalarAndVectorQuantities,
        M::Isolines,
    ),
    entry(
        15,
        "Point Clouds & Glyphs",
        ScalarAndVectorQuantities,
        M::PointClouds,
    ),
    entry(
        16,
        "Streamlines & Tubes",
        ScalarAndVectorQuantities,
        M::Streamlines,
    ),
    entry(
        20,
        "Face Attributes",
        ScalarAndVectorQuantities,
        M::FaceAttributes,
    ),
    entry(
        25,
        "Surface Vectors",
        ScalarAndVectorQuantities,
        M::SurfaceVectors,
    ),
    entry(
        28,
        "Curve Network Quantities",
        ScalarAndVectorQuantities,
        M::CurveNetworkQuantities,
    ),
    entry(
        32,
        "Extended Quantities",
        ScalarAndVectorQuantities,
        M::ExtendedQuantities,
    ),
    entry(38, "Surface LIC", ScalarAndVectorQuantities, M::SurfaceLIC),
    entry(
        39,
        "Tensor Glyphs",
        ScalarAndVectorQuantities,
        M::TensorGlyphs,
    ),
    // --- Volumes & implicit surfaces ---
    entry(
        17,
        "Volume & Isosurface",
        VolumesAndImplicitSurfaces,
        M::Volume,
    ),
    entry(
        18,
        "Clip Volumes",
        VolumesAndImplicitSurfaces,
        M::ClipVolumes,
    ),
    entry(
        26,
        "Volume Meshes",
        VolumesAndImplicitSurfaces,
        M::VolumeMesh,
    ),
    entry(
        30,
        "Implicit Surfaces",
        VolumesAndImplicitSurfaces,
        M::ImplicitSurface,
    ),
    entry(
        31,
        "Sparse Volume Grid",
        VolumesAndImplicitSurfaces,
        M::SparseVolumeGrid,
    ),
    entry(
        48,
        "Scatter Volumes",
        VolumesAndImplicitSurfaces,
        M::ScatterVolumes,
    ),
    // --- Annotation & overlay ---
    entry(9, "Annotations", AnnotationAndOverlay, M::Annotation),
    entry(
        29,
        "Depth-Composited Images",
        AnnotationAndOverlay,
        M::DepthCompositeImages,
    ),
    entry(34, "Labels", AnnotationAndOverlay, M::Labels),
    entry(35, "Overlay Composition", AnnotationAndOverlay, M::Overlay),
    entry(59, "Vector Art (SVG)", AnnotationAndOverlay, M::VectorArt),
    // --- Particles & sprites ---
    entry(41, "Sprites & Particles", ParticlesAndSprites, M::Sprites),
    entry(
        42,
        "Gaussian Splats",
        ParticlesAndSprites,
        M::GaussianSplats,
    ),
    // --- Runtime, animation & GPU plugins ---
    entry(
        36,
        "Playback Runtime Control",
        RuntimeAndAnimation,
        M::PlaybackRuntime,
    ),
    entry(40, "GPU Vertex Warp", RuntimeAndAnimation, M::VertexWarp),
    entry(43, "Scene Runtime", RuntimeAndAnimation, M::SceneRuntime),
    entry(44, "Debug Draw", RuntimeAndAnimation, M::DebugDraw),
    entry(
        45,
        "Skeletal Animation",
        RuntimeAndAnimation,
        M::SkinnedAnimation,
    ),
    entry(
        50,
        "GPU Wave (compute plugin)",
        RuntimeAndAnimation,
        M::GpuWave,
    ),
    entry(
        54,
        "Custom Shading Plugins",
        RuntimeAndAnimation,
        M::CustomShading,
    ),
    // --- Picking & performance ---
    entry(23, "Performance", PickingAndPerformance, M::Performance),
    entry(33, "Picking Levels", PickingAndPerformance, M::PickLevels),
    entry(
        51,
        "Async Asset Streaming",
        PickingAndPerformance,
        M::AsyncUploads,
    ),
    entry(52, "Level of Detail", PickingAndPerformance, M::Lod),
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
}
