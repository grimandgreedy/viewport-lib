//! Pick mask for controlling which item types and sub-element levels are
//! included in a pick call.

// The bitflags! macro generates associated constants without doc comments
// for some internal items (e.g. the implicit ALL/NONE sentinels).
#![allow(missing_docs)]

bitflags::bitflags! {
    /// Controls what [`ViewportRenderer::pick`] and [`ViewportRenderer::pick_rect`]
    /// return.
    ///
    /// Set one or more bits to include the corresponding item types and
    /// sub-element levels. The convenience groups (`OBJECT`, `POINT_LIKE`,
    /// `EDGE_LIKE`, `FACE_LIKE`) cover the common cases.
    ///
    /// Bits are grouped by geometric dimension:
    /// - Bit 0: object level (whole item, any type).
    /// - Bits 1-39: point-like sub-elements (0D discrete elements).
    /// - Bits 40-79: edge-like sub-elements (1D segments).
    /// - Bits 80-119: face-like sub-elements (2D patches or curve strips).
    /// - Bits 120-127: reserved for future use.
    ///
    /// Combining bits from different dimensions (e.g. `FACE | VERTEX`) is legal
    /// but the resulting set contains elements that cannot be acted on uniformly.
    /// Use one dimension at a time in normal selection workflows.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PickMask: u128 {
        // Object level -- whole item, any type.
        // Bits 0-0.
        const OBJECT        = 1 << 0;

        // Point-like sub-types (0D discrete elements).
        // Bits 1-39 reserved for point-like types.
        const VERTEX        = 1 << 1;   // mesh vertex
        const CLOUD_POINT   = 1 << 2;   // point cloud point
        const CELL          = 1 << 3;   // volume mesh cell (boundary or interior)
        const VOXEL         = 1 << 4;   // ray-marched volume voxel
        const SPLAT         = 1 << 5;   // gaussian splat
        const INSTANCE      = 1 << 6;   // glyph / tensor glyph / sprite instance
        const POLY_NODE     = 1 << 7;   // polyline node
        // bits 8-39: reserved for future point-like types

        // Edge-like sub-types (1D elements).
        // Bits 40-79 reserved for edge-like types.
        const EDGE          = 1 << 40;  // mesh edge
        const SEGMENT       = 1 << 41;  // polyline / tube / ribbon segment
        // bits 42-79: reserved for future edge-like types

        // Face-like sub-types (2D elements and curve strips).
        // Bits 80-119 reserved for face-like types.
        const FACE          = 1 << 80;  // mesh face
        const STRIP         = 1 << 81;  // one connected curve within a multi-strip item
        // bits 82-119: reserved for future face-like types

        // bits 120-127: reserved for future use

        // Convenience groups.

        /// All point-like sub-elements: mesh vertices, cloud points, volume mesh
        /// cells, volume voxels, gaussian splats, glyph instances, and polyline
        /// nodes.
        const POINT_LIKE    = Self::VERTEX.bits()
                            | Self::CLOUD_POINT.bits()
                            | Self::CELL.bits()
                            | Self::VOXEL.bits()
                            | Self::SPLAT.bits()
                            | Self::INSTANCE.bits()
                            | Self::POLY_NODE.bits();

        /// All edge-like sub-elements: mesh edges and polyline / tube / ribbon
        /// segments.
        const EDGE_LIKE     = Self::EDGE.bits() | Self::SEGMENT.bits();

        /// All face-like sub-elements: mesh faces and connected curve strips.
        const FACE_LIKE     = Self::FACE.bits() | Self::STRIP.bits();
    }
}
