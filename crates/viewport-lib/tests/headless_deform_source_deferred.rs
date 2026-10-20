//! A buffer-backed deform slot is copied with the frame's own command buffers
//! under the deferred path, not when the frame is encoded.
//!
//! The deferred encoders promise that work the caller submits before the
//! returned batch is seen by it. The test rewrites the slot's source buffer
//! after encoding and before submitting: a copy submitted during encoding reads
//! the old value, one in the batch reads the new one. The deformer's `keep`
//! hook makes the value visible, removing the quad when it reads zero. Skips
//! when no adapter is available or the device cannot run deformers.

use viewport_lib::Colour;
mod common;
use common::*;

use viewport_lib::renderer::{CameraFrame, SceneFrame};
use viewport_lib::resources::{DeformStage, DeformerDesc};
use viewport_lib::wgpu;

const W: u32 = 128;
const H: u32 = 128;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Keeps a vertex while its slot value is above one half. Meshes without slot
/// data are kept.
const KEEP_FROM_SLOT: &str = "\
fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {
    return v;
}

fn keep(v: DeformVertex, ctx: DeformContext) -> f32 {
    if deform_slot_stride(ctx.slot) == 0u {
        return 1.0;
    }
    return deform_read_f32(ctx.slot, v.vertex_index, 0u) - 0.5;
}
";

fn flat(
    renderer: &mut ViewportRenderer,
    mesh: MeshId,
    model: glam::Mat4,
    colour: [f32; 3],
) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = model.to_cols_array_2d();
    item.material = upload_material(
        renderer,
        Material::from_colour(Colour::linear_rgb(colour[0], colour[1], colour[2])),
    );
    item.settings.unlit = true;
    item
}

/// Encode a frame with `render_to_texture_deferred`, run `before_submit`, then
/// submit the batch and read the centre pixel back.
fn render_deferred_centre(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame: &FrameData,
    before_submit: impl FnOnce(),
) -> [u8; 4] {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("deferred_target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let buffers = renderer.render_to_texture_deferred(device, queue, &view, frame);
    before_submit();
    ViewportRenderer::submit_frame(queue, buffers);

    // 128 pixels * 4 bytes is already a multiple of the 256-byte row alignment.
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("deferred_readback"),
        size: (W * H * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 4),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = viewport_lib::gpu::mapped_range(staging.slice(..));
    let i = (((H / 2) * W + W / 2) * 4) as usize;
    [data[i], data[i + 1], data[i + 2], data[i + 3]]
}

#[test]
fn a_deferred_frame_copies_the_source_written_before_its_submit() {
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, FORMAT);
    let plane = viewport_lib::primitives::plane(1.0, 1.0);
    let vertex_count = plane.positions.len();
    let quad = renderer
        .resources_mut()
        .upload_mesh_data(&device, &plane)
        .unwrap();
    let backdrop_mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &plane)
        .unwrap();
    let id = match renderer.resources_mut().register_deformer(
        &device,
        DeformerDesc {
            name: "test_keep_from_slot",
            stage: DeformStage::WorldSpace,
            priority: 0,
            wgsl_body: KEEP_FROM_SLOT.to_string(),
            per_vertex_stride: 4,
        },
    ) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("skipping: device cannot run deformers ({e})");
            return;
        }
    };

    let values = |v: f32| vec![v; vertex_count];
    let source = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("deform_source"),
        size: (vertex_count * 4) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&source, 0, bytemuck::cast_slice(&values(1.0)));
    renderer
        .resources_mut()
        .set_deform_slot_source_buffer(&device, quad, id.slot(), source.clone(), 4)
        .unwrap();

    let backdrop = flat(
        &mut renderer,
        backdrop_mesh,
        glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(4.0),
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.0, 0.0, -0.5),
        ),
        [1.0, 0.0, 0.0],
    );
    let kept = flat(&mut renderer, quad, glam::Mat4::IDENTITY, [1.0, 1.0, 1.0]);
    let mut cam = Camera::default();
    cam.orientation = glam::Quat::IDENTITY;
    cam.center = glam::Vec3::ZERO;
    cam.distance = 5.0;
    cam.aspect = W as f32 / H as f32;
    let mut frame = FrameData::new(
        CameraFrame::new(RenderCamera::from_camera(&cam), [W as f32, H as f32]),
        SceneFrame::from_surface_items(vec![backdrop, kept]),
    );
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // The source holds ones, so the quad is drawn.
    let before = render_deferred_centre(&mut renderer, &device, &queue, &frame, || {});
    assert!(
        before[0] > 150 && before[1] > 150 && before[2] > 150,
        "the quad is drawn while its slot reads one, got {before:?}"
    );

    // Zero the source between encoding and submitting.
    let after = render_deferred_centre(&mut renderer, &device, &queue, &frame, || {
        queue.write_buffer(&source, 0, bytemuck::cast_slice(&values(0.0)));
    });
    assert!(
        after[0] > 150 && after[1] < 80 && after[2] < 80,
        "the frame should copy the source as written before its submit and \
         remove the quad, got {after:?}"
    );
}
