//! The renderer-level upload calls reach the same storage as the
//! `DeviceResources` ones they forward to.
//!
//! `DeviceResources` is a field of `ViewportRenderer` and cannot see the
//! registered item types, so an upload that has to reach a type's own storage
//! can only be reached from the renderer. These are the calls that keep
//! working when storage moves; this pins that they behave identically today,
//! so the move is invisible at the call site.

mod common;
use common::*;

use viewport_lib::renderer::{PointCloudItem, PolylineItem};
use viewport_lib::resources::UploadStatus;

fn polyline() -> PolylineItem {
    let mut item = PolylineItem::default();
    item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
    item.line_width = 2.0;
    item
}

fn point_cloud() -> PointCloudItem {
    let mut item = PointCloudItem::default();
    item.positions = vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]];
    item.point_size = 4.0;
    item
}

#[test]
fn the_renderer_upload_reaches_the_same_store_as_the_resources_one() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Upload one through each route.
    let via_renderer = renderer.upload_polyline(&device, &queue, &polyline());
    let via_resources = renderer
        .resources_mut()
        .upload_polyline(&device, &queue, &polyline());
    assert_ne!(
        via_renderer, via_resources,
        "two uploads must occupy different slots"
    );

    // Either handle resolves through either route, because there is one store.
    assert!(renderer.replace_polyline(&device, &queue, via_resources, &polyline()));
    assert!(
        renderer
            .resources_mut()
            .replace_polyline(&device, &queue, via_renderer, &polyline())
    );

    // And a free through the renderer invalidates it for both.
    assert!(renderer.drop_polyline(via_renderer));
    assert!(
        !renderer.drop_polyline(via_renderer),
        "a freed handle must not resolve twice"
    );
    assert!(
        !renderer
            .resources_mut()
            .replace_polyline(&device, &queue, via_renderer, &polyline()),
        "the resources route must agree the handle is dead"
    );
}

#[test]
fn the_renderer_async_upload_completes_through_the_same_runner() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let job = renderer.begin_upload_point_cloud(&device, &queue, point_cloud());
    // Drive the upload runner until the worker lands the result.
    let mut id = None;
    for _ in 0..200 {
        renderer.resources_mut().process_uploads(&device, &queue);
        match renderer.upload_status(job) {
            UploadStatus::Ready => {
                id = Some(renderer.upload_result_point_cloud(job).expect("ready"));
                break;
            }
            UploadStatus::Failed(e) => panic!("point cloud upload failed: {e:?}"),
            UploadStatus::Pending { .. } => std::thread::sleep(std::time::Duration::from_millis(5)),
            UploadStatus::Unknown => panic!("job id disappeared"),
        }
    }
    let id = id.expect("the queued point-cloud upload should complete in time");

    // The handle the async route produced is live in the shared store.
    assert!(renderer.replace_point_cloud(&device, &queue, id, &point_cloud()));
    assert!(renderer.drop_point_cloud(id));
}

#[test]
fn every_free_route_agrees_on_a_dropped_handle() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let id = renderer
        .resources_mut()
        .upload_point_cloud(&device, &queue, &point_cloud());
    assert!(
        renderer.drop_point_cloud(id),
        "dropped through the renderer"
    );
    assert!(
        !renderer.replace_point_cloud(&device, &queue, id, &point_cloud()),
        "a handle dropped through the renderer is dead for the resources route too"
    );
}
