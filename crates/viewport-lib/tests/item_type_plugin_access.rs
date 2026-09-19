//! A host can reach a registered item-type plugin again.
//!
//! `with_item_type_plugin` takes the plugin by box and the renderer owns it
//! from then on. Without a route back, an item type that stores its own
//! uploaded content could never be uploaded into: the store would exist with
//! nothing able to put anything in it.

mod common;
use common::*;

use viewport_lib::plugin_api::ItemTypePlugin;

/// Stands in for an item type that owns its content rather than storing it
/// through `DeviceResources`.
#[derive(Default)]
struct StoringPlugin {
    uploaded: Vec<String>,
}

impl StoringPlugin {
    /// The plugin's own upload call, reached through `item_type_plugin_mut`.
    fn upload(&mut self, name: &str) -> usize {
        self.uploaded.push(name.to_string());
        self.uploaded.len() - 1
    }
}

impl ItemTypePlugin for StoringPlugin {
    fn type_name(&self) -> &'static str {
        "storing_test"
    }

    fn resident_bytes(&self) -> u64 {
        self.uploaded.len() as u64 * 1024
    }
}

/// A second type, to pin that the lookup is by name and checks the type.
struct OtherPlugin;

impl ItemTypePlugin for OtherPlugin {
    fn type_name(&self) -> &'static str {
        "other_test"
    }
}

#[test]
fn a_registered_plugin_can_be_reached_and_uploaded_into() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.with_item_type_plugin(&device, Box::new(StoringPlugin::default()));
    renderer.with_item_type_plugin(&device, Box::new(OtherPlugin));

    // Upload into content the plugin owns itself.
    let plugin = renderer
        .item_type_plugin_mut::<StoringPlugin>("storing_test")
        .expect("the plugin we just registered");
    let first = plugin.upload("terrain");
    let second = plugin.upload("foliage");
    assert_eq!((first, second), (0, 1));

    // Read it back on the shared borrow.
    let plugin = renderer
        .item_type_plugin::<StoringPlugin>("storing_test")
        .expect("still registered");
    assert_eq!(plugin.uploaded, vec!["terrain", "foliage"]);

    // Content a plugin owns reaches the working-set figure.
    assert_eq!(renderer.resident_bytes().plugin_bytes, 2048);
}

#[test]
fn the_lookup_checks_both_the_name_and_the_type() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.with_item_type_plugin(&device, Box::new(StoringPlugin::default()));

    assert!(
        renderer
            .item_type_plugin::<StoringPlugin>("no_such_type")
            .is_none(),
        "an unregistered name resolves to nothing"
    );
    assert!(
        renderer
            .item_type_plugin::<OtherPlugin>("storing_test")
            .is_none(),
        "a registered name asked for as the wrong type resolves to nothing, not a bad cast"
    );
    assert!(
        renderer
            .item_type_plugin::<StoringPlugin>("storing_test")
            .is_some()
    );
}
