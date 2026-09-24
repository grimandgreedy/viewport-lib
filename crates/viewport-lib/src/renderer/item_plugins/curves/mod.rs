//! The three curve mesh item types as [`ItemTypePlugin`]s: streamtube, tube
//! and ribbon.
//!
//! All three take strips of world-space points and sweep them into a connected
//! triangle mesh on the CPU, so they share their pick, POLY_NODE pick and
//! outline-mask machinery ([`pipeline`]) and their screen-space CPU pick
//! helpers ([`cpu_pick`]). They stay three plugins with three type names,
//! because consumers submit them on three `SceneFrame` fields; each owns its
//! own pipelines, so pulling one out later does not disturb the others.
//!
//! Streamtube and tube shade through the same pipeline description and
//! therefore compile it once each. Ribbon differs: it is a flat two-sided quad
//! strip, so it keys its pipelines on blend mode, casts shadows, and routes
//! transparent draws through the OIT pass.

mod cpu_pick;
mod draw;
mod pipeline;
mod ribbon;
pub(crate) mod store;
mod streamtube;
mod tube;
pub(crate) mod types;

pub(crate) use ribbon::{RibbonPlugin, TYPE_NAME as RIBBON_TYPE_NAME};
pub(crate) use streamtube::{StreamtubePlugin, TYPE_NAME as STREAMTUBE_TYPE_NAME};
pub(crate) use tube::{TYPE_NAME as TUBE_TYPE_NAME, TubePlugin};

use crate::renderer::{
    RibbonItem, RibbonRefItem, StreamtubeItem, StreamtubeRefItem, TubeItem, TubeRefItem,
};

macro_rules! item_collection {
    ($ty:ty, $name:expr) => {
        impl crate::plugin_api::PluginItem for $ty {
            const TYPE_NAME: &'static str = $name;

            fn settings(&self) -> &crate::scene::material::ItemSettings {
                &self.settings
            }
        }
    };
}

item_collection!(StreamtubeItem, STREAMTUBE_TYPE_NAME);
item_collection!(StreamtubeRefItem, STREAMTUBE_TYPE_NAME);
item_collection!(TubeItem, TUBE_TYPE_NAME);
item_collection!(TubeRefItem, TUBE_TYPE_NAME);
item_collection!(RibbonItem, RIBBON_TYPE_NAME);
item_collection!(RibbonRefItem, RIBBON_TYPE_NAME);
