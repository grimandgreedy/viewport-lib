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
}

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
}

impl DeviceProfile {
    /// The harness default: a high-performance adapter with `wgpu` default
    /// limits, opportunistically enabling `INDIRECT_FIRST_INSTANCE` (the
    /// GPU-culled indirect draw path) and `TIMESTAMP_QUERY` (GPU frame timing)
    /// when the adapter has them. This is what [`headless_device`] uses.
    pub fn harness() -> Self {
        Self {
            label: "viewport-lib-testkit",
            power_preference: wgpu::PowerPreference::HighPerformance,
            limits: Limits::WgpuDefault,
            max_bind_groups: None,
            required_features: wgpu::Features::empty(),
            optional_features: wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::TIMESTAMP_QUERY,
            recommended_features: false,
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
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: profile.power_preference,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
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
    };
    if let Some(max) = profile.max_bind_groups {
        required_limits.max_bind_groups = max;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(profile.label),
        required_features,
        required_limits,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}
