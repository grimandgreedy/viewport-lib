//! An item-type plugin that holds its own uploaded content reports it into the
//! renderer's resident-byte figure.
//!
//! Byte accounting used to enumerate each store by name, so content an item
//! type held itself was invisible to the working-set figure an eviction policy
//! budgets against. `ItemTypePlugin::resident_bytes` closes that.

mod common;
use common::*;

use viewport_lib::plugin_api::ItemTypePlugin;

/// Reports a fixed byte figure, standing in for a plugin that owns a store.
struct AccountingPlugin {
    name: &'static str,
    bytes: u64,
}

impl ItemTypePlugin for AccountingPlugin {
    fn type_name(&self) -> &'static str {
        self.name
    }

    fn resident_bytes(&self) -> u64 {
        self.bytes
    }
}

/// Reports nothing, the default for a type that stores everything through the
/// shared upload calls.
struct SilentPlugin;

impl ItemTypePlugin for SilentPlugin {
    fn type_name(&self) -> &'static str {
        "silent_test"
    }
}

#[test]
fn a_plugins_own_content_lands_in_the_renderers_resident_bytes() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let baseline = renderer.resident_bytes();
    assert_eq!(
        baseline.plugin_bytes, 0,
        "the built-in item types store through DeviceResources, which counts them already"
    );

    renderer.with_item_type_plugin(&device, Box::new(SilentPlugin));
    renderer.with_item_type_plugin(
        &device,
        Box::new(AccountingPlugin {
            name: "accounting_a",
            bytes: 4096,
        }),
    );
    renderer.with_item_type_plugin(
        &device,
        Box::new(AccountingPlugin {
            name: "accounting_b",
            bytes: 1024,
        }),
    );

    let after = renderer.resident_bytes();
    assert_eq!(
        after.plugin_bytes,
        4096 + 1024,
        "every registered plugin's own content is counted"
    );
    assert_eq!(
        after.total(),
        baseline.total() + 4096 + 1024,
        "plugin bytes are part of the evictable working set"
    );

    // Re-registering under an existing name replaces that plugin, so its
    // figure is replaced too rather than being counted twice.
    renderer.with_item_type_plugin(
        &device,
        Box::new(AccountingPlugin {
            name: "accounting_a",
            bytes: 16,
        }),
    );
    assert_eq!(renderer.resident_bytes().plugin_bytes, 16 + 1024);

    // The resources-level call cannot see the registry, so it still reports
    // zero; that is the reason the renderer-level call exists.
    assert_eq!(renderer.resources().resident_bytes().plugin_bytes, 0);
}
