//! Per-viewport target release.
//!
//! A viewport allocates the targets of an effect the first frame that turns it
//! on. With `RendererConfig::target_release_frames` set, a target the frames
//! stop using goes back to its one-texel placeholder after that many frames,
//! and the next frame that needs it allocates it again. With the default
//! `None` nothing is released.

use viewport_lib::resources::build_log;
use viewport_lib::wgpu;
use viewport_lib::{
    Colour, FrameData, Material, PipelineCompilation, RendererConfig, SceneRenderItem,
    ViewportRenderer,
};
use viewport_lib_testkit::fixtures::{probe_frame, probe_quad};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
const SIZE: u32 = 64;
/// The size the build-log test renders at, so allocations from the other
/// tests in this binary cannot be mistaken for its own.
const LOGGED_SIZE: u32 = 72;

struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: ViewportRenderer,
}

impl Rig {
    fn new(release: Option<u32>) -> Option<Self> {
        let (device, queue) = viewport_lib_testkit::headless_device()?;
        let mut config =
            RendererConfig::new(FORMAT).with_pipeline_compilation(PipelineCompilation::Blocking);
        if let Some(frames) = release {
            config = config.with_target_release_frames(frames);
        }
        let mut renderer = ViewportRenderer::with_config(&device, &config);
        viewport_lib_plugins::item_types::install(&mut renderer, &device);
        Some(Self {
            device,
            queue,
            renderer,
        })
    }

    fn render(&mut self, frame: &FrameData, size: u32) {
        self.renderer
            .render_offscreen(&self.device, &self.queue, frame, size, size);
    }

    fn live(&self) -> Vec<&'static str> {
        self.renderer.debug_viewport_live_targets(0)
    }

    fn holds(&self, target: &str) -> bool {
        self.live().contains(&target)
    }

    /// A unit quad as a foreground item.
    fn foreground_item(&mut self) -> SceneRenderItem {
        let mesh = self
            .renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &probe_quad())
            .expect("upload probe quad");
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh;
        item.material = self
            .renderer
            .resources_mut()
            .upload_material(Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8)))
            .expect("material upload");
        item.settings.unlit = true;
        item
    }
}

fn frame(size: u32, bloom: bool) -> FrameData {
    let mut frame = probe_frame(size, [0.2, 0.2, 0.2, 1.0]);
    frame.effects.post_process.bloom.enabled = bloom;
    frame
}

/// Bloom on for one frame, then off: its group is released after three idle
/// frames, and the next frame with bloom on allocates it again.
#[test]
fn an_idle_bloom_group_is_released_and_promoted_again() {
    let Some(mut rig) = Rig::new(Some(3)) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    rig.render(&frame(SIZE, true), SIZE);
    assert!(rig.holds("bloom"), "bloom on: {:?}", rig.live());
    for idle in 1..=2 {
        rig.render(&frame(SIZE, false), SIZE);
        assert!(
            rig.holds("bloom"),
            "bloom released after {idle} idle frames: {:?}",
            rig.live()
        );
    }
    rig.render(&frame(SIZE, false), SIZE);
    assert!(
        !rig.holds("bloom"),
        "bloom still held after three idle frames: {:?}",
        rig.live()
    );
    assert!(
        rig.holds("scene"),
        "the scene group must stay: {:?}",
        rig.live()
    );
    rig.render(&frame(SIZE, false), SIZE);
    assert!(!rig.holds("bloom"));

    rig.render(&frame(SIZE, true), SIZE);
    assert!(
        rig.holds("bloom"),
        "bloom back on must promote at once: {:?}",
        rig.live()
    );
}

/// The release and the promotion show in the target allocations: the release
/// rebuilds bloom's threshold target as a one-texel placeholder, and turning
/// bloom back on allocates it at scene size again.
#[test]
fn bloom_release_shows_in_the_build_log() {
    let Some(mut rig) = Rig::new(Some(3)) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    build_log::enable();
    let full = LOGGED_SIZE as u64 * LOGGED_SIZE as u64 * 8;
    let thresholds = || -> Vec<u64> {
        build_log::drain_textures()
            .into_iter()
            .filter(|(label, _)| label == "bloom_threshold_texture")
            .map(|(_, bytes)| bytes)
            .collect()
    };
    rig.render(&frame(LOGGED_SIZE, true), LOGGED_SIZE);
    assert!(thresholds().contains(&full));
    for _ in 0..2 {
        rig.render(&frame(LOGGED_SIZE, false), LOGGED_SIZE);
    }
    assert!(!thresholds().contains(&full));
    rig.render(&frame(LOGGED_SIZE, false), LOGGED_SIZE);
    let released = thresholds();
    assert!(
        released.contains(&8),
        "the release must rebuild bloom as a placeholder: {released:?}"
    );
    rig.render(&frame(LOGGED_SIZE, true), LOGGED_SIZE);
    assert!(
        thresholds().contains(&full),
        "bloom back on must allocate it at scene size"
    );
}

/// The foreground depth target follows the same rule.
#[test]
fn an_idle_foreground_depth_target_is_released_and_allocated_again() {
    let Some(mut rig) = Rig::new(Some(3)) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let item = rig.foreground_item();
    let mut with_foreground = frame(SIZE, false);
    with_foreground.scene.foreground_items.push(item);
    let without = frame(SIZE, false);

    rig.render(&with_foreground, SIZE);
    assert!(rig.holds("foreground_depth"), "{:?}", rig.live());
    for _ in 0..2 {
        rig.render(&without, SIZE);
        assert!(rig.holds("foreground_depth"), "{:?}", rig.live());
    }
    rig.render(&without, SIZE);
    assert!(
        !rig.holds("foreground_depth"),
        "foreground depth still held after three idle frames: {:?}",
        rig.live()
    );
    rig.render(&with_foreground, SIZE);
    assert!(rig.holds("foreground_depth"), "{:?}", rig.live());
}

/// With the default config nothing a viewport has used is released.
#[test]
fn nothing_is_released_by_default() {
    let Some(mut rig) = Rig::new(None) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let item = rig.foreground_item();
    let mut first = frame(SIZE, true);
    first.scene.foreground_items.push(item);
    rig.render(&first, SIZE);
    let held = rig.live();
    assert!(held.contains(&"bloom") && held.contains(&"foreground_depth"));
    for _ in 0..10 {
        rig.render(&frame(SIZE, false), SIZE);
    }
    assert_eq!(rig.live(), held);
}
