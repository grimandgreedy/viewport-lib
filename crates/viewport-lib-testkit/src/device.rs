//! Headless `wgpu` device construction shared by the harness and the drivers.
//!
//! A single parameterised constructor, [`headless_device_with`], replaces the
//! dozen near-identical copies that used to live in every test binary and
//! driver. A [`DeviceProfile`] describes what a caller needs (adapter power
//! preference, a limits policy, and a feature set split into "must have" and
//! "take if present"), and the constructor turns it into a device or `None`.
//! [`headless_device`] is the zero-argument default the harness uses.

use viewport_lib::ViewportRenderer;
use viewport_lib::wgpu;

/// Which limits a headless device is requested with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limits {
    /// `wgpu`'s built-in default limits.
    WgpuDefault,
    /// `ViewportRenderer::recommended_device_limits(&adapter)`: what a
    /// limits-following consumer requests, and what the renderer asserts it has.
    Recommended,
    /// `wgpu::Limits::downlevel_webgl2_defaults()`, with the adapter's own
    /// texture size and, on wgpu 27, GLES 3.0's 60 inter-stage components: no
    /// storage buffers and no compute, as a browser without WebGPU gives. wgpu enforces requested limits whatever the hardware can
    /// do, so any adapter runs as that device class. The profile also strips
    /// the downlevel flags WebGL2 lacks (see [`WEBGL2_MISSING_DOWNLEVEL`]).
    DownlevelWebGl2,
}

/// The downlevel flags a WebGL2 device does not report, stripped from the
/// adapter's by a [`Limits::DownlevelWebGl2`] profile.
pub const WEBGL2_MISSING_DOWNLEVEL: wgpu::DownlevelFlags = wgpu::DownlevelFlags::COMPUTE_SHADERS
    .union(wgpu::DownlevelFlags::VERTEX_STORAGE)
    .union(wgpu::DownlevelFlags::FRAGMENT_STORAGE)
    .union(wgpu::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE)
    .union(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
    .union(wgpu::DownlevelFlags::CUBE_ARRAY_TEXTURES)
    .union(wgpu::DownlevelFlags::BASE_VERTEX);

/// What kind of headless device a caller needs.
///
/// [`headless_device_with`] reads this to pick an adapter, apply the limits
/// policy, and assemble the requested features. A feature listed in
/// `required_features` that the adapter lacks makes the whole call return `None`
/// so the caller's test skips cleanly; a feature in `optional_features` is
/// requested only when the adapter offers it.
#[derive(Clone, Debug)]
pub struct DeviceProfile {
    /// Debug label for the created device.
    pub label: &'static str,
    /// Adapter power preference.
    pub power_preference: wgpu::PowerPreference,
    /// Limits policy the device is requested with.
    pub limits: Limits,
    /// When set, overrides `max_bind_groups` after the limits policy is applied.
    /// Used to reproduce a consumer that caps bind groups (for example iced).
    pub max_bind_groups: Option<u32>,
    /// Features the adapter must expose; a missing one makes the call return
    /// `None`.
    pub required_features: wgpu::Features,
    /// Features requested only when the adapter offers them.
    pub optional_features: wgpu::Features,
    /// Also request `ViewportRenderer::recommended_device_features(&adapter)`.
    pub recommended_features: bool,
    /// Downlevel flags removed from the adapter's before they reach the
    /// renderer, to model a device class the adapter is not (no cube-array
    /// textures, no indirect execution). The device itself is unchanged; the
    /// renderer reads its capabilities from these flags.
    pub remove_downlevel: wgpu::DownlevelFlags,
}

impl DeviceProfile {
    /// The harness default: a high-performance adapter with `wgpu` default
    /// limits, opportunistically enabling `INDIRECT_FIRST_INSTANCE` (the
    /// GPU-culled indirect draw path), `TIMESTAMP_QUERY` (GPU frame timing),
    /// `DEPTH_CLIP_CONTROL` (depth-clamped shadow cascades) and
    /// `DEPTH32FLOAT_STENCIL8` (the 32-bit float scene depth) when the adapter
    /// has them. This is what [`headless_device`] uses.
    pub fn harness() -> Self {
        Self {
            label: "viewport-lib-testkit",
            power_preference: wgpu::PowerPreference::HighPerformance,
            limits: Limits::WgpuDefault,
            max_bind_groups: None,
            required_features: wgpu::Features::empty(),
            optional_features: wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::DEPTH32FLOAT_STENCIL8,
            recommended_features: false,
            remove_downlevel: wgpu::DownlevelFlags::empty(),
        }
    }

    /// A low-power adapter with recommended limits and no extra features: the
    /// baseline a renderer test uses.
    pub fn low_power(label: &'static str) -> Self {
        Self {
            label,
            power_preference: wgpu::PowerPreference::LowPower,
            limits: Limits::Recommended,
            max_bind_groups: None,
            required_features: wgpu::Features::empty(),
            optional_features: wgpu::Features::empty(),
            recommended_features: false,
            remove_downlevel: wgpu::DownlevelFlags::empty(),
        }
    }

    /// A high-performance adapter with recommended limits and no extra features.
    pub fn high_performance(label: &'static str) -> Self {
        Self {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Self::low_power(label)
        }
    }

    /// Require the given features (return `None` if the adapter lacks any).
    pub fn require(mut self, features: wgpu::Features) -> Self {
        self.required_features |= features;
        self
    }

    /// Request the given features only when the adapter offers them.
    pub fn optional(mut self, features: wgpu::Features) -> Self {
        self.optional_features |= features;
        self
    }

    /// Also request the renderer's recommended device features.
    pub fn with_recommended_features(mut self) -> Self {
        self.recommended_features = true;
        self
    }

    /// Override `max_bind_groups` in the requested limits.
    pub fn max_bind_groups(mut self, max: u32) -> Self {
        self.max_bind_groups = Some(max);
        self
    }

    /// A device on WebGL2's limits and downlevel flags: the class a browser
    /// without WebGPU gives, on any adapter.
    pub fn downlevel_webgl2(label: &'static str) -> Self {
        Self {
            limits: Limits::DownlevelWebGl2,
            ..Self::high_performance(label)
        }
    }

    /// Strip `flags` from the adapter's downlevel flags as the renderer sees
    /// them.
    pub fn without_downlevel(mut self, flags: wgpu::DownlevelFlags) -> Self {
        self.remove_downlevel |= flags;
        self
    }

    /// The downlevel capabilities the renderer should read for a device made
    /// from `adapter` with this profile.
    pub fn downlevel_for(&self, adapter: &wgpu::Adapter) -> wgpu::DownlevelCapabilities {
        let mut caps = adapter.get_downlevel_capabilities();
        let mut remove = self.remove_downlevel;
        if self.limits == Limits::DownlevelWebGl2 {
            remove |= WEBGL2_MISSING_DOWNLEVEL;
        }
        caps.flags.remove(remove);
        caps
    }
}

impl Default for DeviceProfile {
    fn default() -> Self {
        Self::harness()
    }
}

/// Create a headless `wgpu` device and queue for the harness default profile, or
/// `None` when no adapter is available (so tests skip cleanly on machines
/// without a GPU).
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with(&DeviceProfile::harness())
}

/// Create a headless `wgpu` device and queue for `profile`, or `None` when no
/// adapter is available or the adapter lacks a required feature.
pub fn headless_device_with(profile: &DeviceProfile) -> Option<(wgpu::Device, wgpu::Queue)> {
    headless_device_with_info(profile).map(|(device, queue, _)| (device, queue))
}

/// Like [`headless_device_with`], additionally returning the adapter info, so a
/// caller can key machine-dependent artifacts (golden references) on the
/// backend the device actually runs on.
pub fn headless_device_with_info(
    profile: &DeviceProfile,
) -> Option<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo)> {
    headless_device_full(profile).map(|d| (d.device, d.queue, d.info))
}

/// A headless device with everything [`headless_device_full`] learns from the
/// adapter.
pub struct HeadlessDevice {
    /// The device.
    pub device: wgpu::Device,
    /// Its queue.
    pub queue: wgpu::Queue,
    /// Info for the adapter the device came from.
    pub info: wgpu::AdapterInfo,
    /// The adapter's downlevel capabilities after the profile's
    /// [`remove_downlevel`](DeviceProfile::remove_downlevel). Pass them to
    /// `RendererConfig::with_downlevel_capabilities`.
    pub downlevel: wgpu::DownlevelCapabilities,
}

/// Like [`headless_device_with_info`], also returning the downlevel
/// capabilities the renderer should read.
pub fn headless_device_full(profile: &DeviceProfile) -> Option<HeadlessDevice> {
    // From the environment, so `WGPU_BACKEND=gl` puts a run on the GL backend.
    let instance = wgpu::instance_from_env();
    let adapter = pollster::block_on(
        instance.request_adapter(&wgpu::headless_adapter_options(profile.power_preference)),
    )
    .ok()?;

    if !adapter.features().contains(profile.required_features) {
        return None;
    }

    let mut required_features = profile.required_features;
    required_features |= profile.optional_features & adapter.features();
    if profile.recommended_features {
        required_features |= ViewportRenderer::recommended_device_features(&adapter);
    }

    let mut required_limits = match profile.limits {
        Limits::WgpuDefault => wgpu::Limits::default(),
        Limits::Recommended => ViewportRenderer::recommended_device_limits(&adapter),
        Limits::DownlevelWebGl2 => {
            #[allow(unused_mut)]
            let mut limits =
                wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
            // wgpu 27 counts inter-stage components, and its WebGL2 default of 31
            // is a desktop GL 3.1 driver's. WebGL2 is GLES 3.0, which guarantees
            // 60, and wgpu's GL backend reports the browser's own figure. 29 and
            // 30 count variables, where the WebGL2 default (15) is GLES 3.0's.
            #[cfg(feature = "wgpu27")]
            {
                limits.max_inter_stage_shader_components = 60;
            }
            limits
        }
    };
    if let Some(max) = profile.max_bind_groups {
        required_limits.max_bind_groups = max;
    }

    let info = adapter.get_info();
    let downlevel = profile.downlevel_for(&adapter);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(profile.label),
        required_features,
        required_limits,
        ..Default::default()
    }))
    .ok()?;
    Some(HeadlessDevice {
        device,
        queue,
        info,
        downlevel,
    })
}
