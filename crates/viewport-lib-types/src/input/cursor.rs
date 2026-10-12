//! The shape the mouse pointer takes over a window.

/// The shape a host wants the pointer to take: the ordinary arrow, a hand over a
/// link, a grab hand over something draggable, a resize arrow over an edge.
///
/// Pure data, like the rest of this module: a runner maps it to whatever its
/// windowing layer names the shape. A host that drives an interactive tool over the
/// viewport picks a shape per frame and hands it to the runner's `set_cursor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum CursorShape {
    /// The ordinary arrow.
    #[default]
    Default,
    /// The hand that says a thing can be clicked.
    Pointer,
    /// The I-beam over editable or selectable text.
    Text,
    /// Crosshairs, for picking a point precisely.
    Crosshair,
    /// The four-way arrow over something that can be moved.
    Move,
    /// The barred circle over somewhere a drag cannot land.
    NotAllowed,
    /// The open hand over something that can be picked up.
    Grab,
    /// The closed hand while something is being dragged.
    Grabbing,
    /// Resize left and right.
    ResizeHorizontal,
    /// Resize up and down.
    ResizeVertical,
    /// Resize along the north-east / south-west diagonal.
    ResizeNeSw,
    /// Resize along the north-west / south-east diagonal.
    ResizeNwSe,
    /// Drag a column edge left or right.
    ResizeColumn,
    /// Drag a row edge up or down.
    ResizeRow,
}

impl CursorShape {
    /// Every shape, in declaration order.
    pub const ALL: [CursorShape; 14] = [
        CursorShape::Default,
        CursorShape::Pointer,
        CursorShape::Text,
        CursorShape::Crosshair,
        CursorShape::Move,
        CursorShape::NotAllowed,
        CursorShape::Grab,
        CursorShape::Grabbing,
        CursorShape::ResizeHorizontal,
        CursorShape::ResizeVertical,
        CursorShape::ResizeNeSw,
        CursorShape::ResizeNwSe,
        CursorShape::ResizeColumn,
        CursorShape::ResizeRow,
    ];

    /// The shape's CSS `cursor` keyword, as [`FromStr`] parses it: `"pointer"`,
    /// `"not-allowed"`, `"ew-resize"` and so on. These are the names winit's
    /// `CursorIcon::name` returns and a browser takes, so they are stable across
    /// hosts.
    ///
    /// [`FromStr`]: std::str::FromStr
    pub fn name(self) -> &'static str {
        match self {
            CursorShape::Default => "default",
            CursorShape::Pointer => "pointer",
            CursorShape::Text => "text",
            CursorShape::Crosshair => "crosshair",
            CursorShape::Move => "move",
            CursorShape::NotAllowed => "not-allowed",
            CursorShape::Grab => "grab",
            CursorShape::Grabbing => "grabbing",
            CursorShape::ResizeHorizontal => "ew-resize",
            CursorShape::ResizeVertical => "ns-resize",
            CursorShape::ResizeNeSw => "nesw-resize",
            CursorShape::ResizeNwSe => "nwse-resize",
            CursorShape::ResizeColumn => "col-resize",
            CursorShape::ResizeRow => "row-resize",
        }
    }

    /// The variant's Rust name, which is also how serde writes it.
    fn variant_name(self) -> &'static str {
        match self {
            CursorShape::Default => "Default",
            CursorShape::Pointer => "Pointer",
            CursorShape::Text => "Text",
            CursorShape::Crosshair => "Crosshair",
            CursorShape::Move => "Move",
            CursorShape::NotAllowed => "NotAllowed",
            CursorShape::Grab => "Grab",
            CursorShape::Grabbing => "Grabbing",
            CursorShape::ResizeHorizontal => "ResizeHorizontal",
            CursorShape::ResizeVertical => "ResizeVertical",
            CursorShape::ResizeNeSw => "ResizeNeSw",
            CursorShape::ResizeNwSe => "ResizeNwSe",
            CursorShape::ResizeColumn => "ResizeColumn",
            CursorShape::ResizeRow => "ResizeRow",
        }
    }
}

/// A cursor name that matches no [`CursorShape`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown cursor shape: {0:?}")]
pub struct UnknownCursorShape(pub String);

impl std::str::FromStr for CursorShape {
    type Err = UnknownCursorShape;

    /// Parse a shape by its [`name`](CursorShape::name), ignoring case. The
    /// variant name (`"NotAllowed"`), as serde writes it, also parses.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        CursorShape::ALL
            .iter()
            .copied()
            .find(|c| c.name().eq_ignore_ascii_case(s) || c.variant_name().eq_ignore_ascii_case(s))
            .ok_or_else(|| UnknownCursorShape(s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for shape in CursorShape::ALL {
            assert_eq!(shape.name().parse::<CursorShape>(), Ok(shape));
            assert_eq!(shape.variant_name().parse::<CursorShape>(), Ok(shape));
        }
        assert_eq!(
            "NOT-ALLOWED".parse::<CursorShape>(),
            Ok(CursorShape::NotAllowed)
        );
        assert!("hand".parse::<CursorShape>().is_err());
    }

    #[test]
    fn names_are_distinct() {
        let mut names: Vec<_> = CursorShape::ALL.iter().map(|c| c.name()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), CursorShape::ALL.len());
    }
}
