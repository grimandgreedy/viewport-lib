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
}

impl Entry {
    /// The text shown in the selector and as the controls-panel heading.
    pub(crate) fn label(&self) -> String {
        format!("{}: {}", self.number, self.title)
    }
}

const fn entry(number: u32, title: &'static str, mode: ShowcaseMode) -> Entry {
    Entry {
        number,
        title,
        mode,
    }
}

use ShowcaseMode as M;

/// Every showcase, in the order the selector lists them and the cycle keys
/// walk them.
pub(crate) const SHOWCASES: [Entry; 59] = [
    entry(1, "Rendering Basics", M::Basic),
    entry(2, "Scene Graph", M::SceneGraph),
    entry(3, "Ground Plane", M::GroundPlane),
    entry(4, "Interaction", M::Interaction),
    entry(5, "Materials and Visibility", M::MaterialsVisibility),
    entry(6, "Post-Processing", M::PostProcess),
    entry(7, "Normal Maps", M::NormalMaps),
    entry(8, "Shadows", M::Shadows),
    entry(9, "Annotations", M::Annotation),
    entry(10, "Camera Tools", M::CameraTools),
    entry(11, "Lights", M::Lights),
    entry(12, "Scalar Fields", M::ScalarFields),
    entry(13, "Multi-Viewport", M::MultiViewport),
    entry(14, "Isolines & Contours", M::Isolines),
    entry(15, "Point Clouds & Glyphs", M::PointClouds),
    entry(16, "Streamlines & Tubes", M::Streamlines),
    entry(17, "Volume & Isosurface", M::Volume),
    entry(18, "Clip Volumes", M::ClipVolumes),
    entry(19, "Matcap Shading", M::Matcap),
    entry(20, "Face Attributes", M::FaceAttributes),
    entry(21, "Textures", M::Textures),
    entry(22, "UV Parameterization", M::ParamVis),
    entry(23, "Performance", M::Performance),
    entry(24, "Backface Policy", M::BackfacePolicy),
    entry(25, "Surface Vectors", M::SurfaceVectors),
    entry(26, "Volume Meshes", M::VolumeMesh),
    entry(27, "Camera Framing & HUD", M::Auxiliary),
    entry(28, "Curve Network Quantities", M::CurveNetworkQuantities),
    entry(29, "Depth-Composited Images", M::DepthCompositeImages),
    entry(30, "Implicit Surfaces", M::ImplicitSurface),
    entry(31, "Sparse Volume Grid", M::SparseVolumeGrid),
    entry(32, "Extended Quantities", M::ExtendedQuantities),
    entry(33, "Picking Levels", M::PickLevels),
    entry(34, "Labels", M::Labels),
    entry(35, "Overlay Composition", M::Overlay),
    entry(36, "Playback Runtime Control", M::PlaybackRuntime),
    entry(37, "Probe Widgets", M::ProbeWidgets),
    entry(38, "Surface LIC", M::SurfaceLIC),
    entry(39, "Tensor Glyphs", M::TensorGlyphs),
    entry(40, "GPU Vertex Warp", M::VertexWarp),
    entry(41, "Sprites & Particles", M::Sprites),
    entry(42, "Gaussian Splats", M::GaussianSplats),
    entry(43, "Scene Runtime", M::SceneRuntime),
    entry(44, "Debug Draw", M::DebugDraw),
    entry(45, "Skeletal Animation", M::SkinnedAnimation),
    entry(46, "Decals", M::Decals),
    entry(47, "Lighting Consistency", M::LightingConsistency),
    entry(48, "Scatter Volumes", M::ScatterVolumes),
    entry(49, "Scene Lights", M::SceneLights),
    entry(50, "GPU Wave (compute plugin)", M::GpuWave),
    entry(51, "Async Asset Streaming", M::AsyncUploads),
    entry(52, "Level of Detail", M::Lod),
    entry(53, "Vertex Colours & Painting", M::VertexColours),
    entry(54, "Custom Shading Plugins", M::CustomShading),
    entry(55, "Foreground Composite Pass", M::Foreground),
    entry(56, "Submesh Materials", M::SubmeshMaterials),
    entry(57, "Photometric Lighting", M::PhotometricLighting),
    entry(58, "Physically-Based Surfaces", M::PhysicallyBasedSurfaces),
    entry(59, "Vector Art (SVG)", M::VectorArt),
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
