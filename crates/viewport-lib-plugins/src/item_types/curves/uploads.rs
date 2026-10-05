//! How a renderer uploads, writes and releases streamtubes, tubes and ribbons.

use super::*;

standard_uploads!(
    StreamtubeItem,
    StreamtubeId,
    STREAMTUBE_TYPE_NAME,
    StreamtubePlugin
);
standard_uploads!(TubeItem, TubeId, TUBE_TYPE_NAME, TubePlugin);
standard_uploads!(RibbonItem, RibbonId, RIBBON_TYPE_NAME, RibbonPlugin);
