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
