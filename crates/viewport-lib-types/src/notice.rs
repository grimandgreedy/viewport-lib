//! Notices: conditions the renderer reports while it keeps drawing.
//!
//! A notice is something a host should hear about but that does not fail a
//! call: items a path cannot draw, a stale handle in the frame, a texture bound
//! into a slot that wants the other colour space. The renderer coalesces
//! repeats of the same condition into one [`Notice`] with a count and a frame
//! range, and hands them over with `ViewportRenderer::take_notices`.

use std::fmt;

/// How serious a [`Notice`] is. Ordered: `Info < Warning < Error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Severity {
    /// Worth knowing, nothing is wrong with the image.
    Info,
    /// Something the host asked for was not drawn, or not drawn as asked.
    Warning,
    /// The frame is wrong in a way the host has to fix.
    Error,
}

/// Which part of the renderer a [`Notice`] comes from.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NoticeCategory {
    /// GPU memory and what is resident in it.
    Residency,
    /// Background uploads and the upload-job queue.
    UploadQueue,
    /// The device: limits, features, loss.
    Device,
    /// A resource bound into a slot: textures, buffers, render targets.
    ResourceSlot,
    /// A submitted item, or a plugin's items.
    Item,
    /// A render path or pipeline that cannot do what the frame asks.
    Pipeline,
}

/// What a [`Notice`] is about, with the ids and sizes that identify it.
///
/// `Display` gives the message the renderer logs.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum NoticeDetail {
    /// A post-effect producer fills a composite slot whose built-in effect is
    /// also enabled; the producer's output wins.
    PostEffectSlotTaken {
        /// The slot, as its `PostEffectSlot` variant name.
        slot: &'static str,
        /// The producer's type name.
        producer: String,
    },
    /// The frame carries foreground items, which a host-owned render pass
    /// (`paint` / `paint_viewport`) cannot draw.
    ForegroundUnsupportedOnPath,
    /// A plugin's items were skipped because the LDR pipeline is active and the
    /// plugin does not draw on it.
    ItemsSkippedOnLdr {
        /// The item type name the plugin registered.
        plugin: String,
    },
    /// Transparent volume meshes were skipped because the LDR pipeline has no
    /// order-independent transparency pass.
    VolumeTransparencyOnLdr,
    /// A plugin that encodes after the scene on the LDR path was skipped
    /// because the frame was drawn into a host-owned render pass
    /// (`paint_viewport`), which cannot be ended mid-frame.
    EncodeSkippedOnPath {
        /// The item type name the plugin registered.
        plugin: String,
    },
    /// A runtime's GPU plugins have a `post_paint` hook, but this frame left
    /// no scene colour and depth they can read, so the hook was not called.
    PostPaintSkipped {
        /// Why the frame has no readable targets, for example
        /// `"the LDR frame was drawn into the host's render pass"`.
        reason: &'static str,
    },
    /// The frame asked for `ExposureMode::Automatic` on the LDR pipeline, which
    /// has no HDR target to meter, so it was drawn at the passthrough exposure.
    AutoExposureOnLdr,
    /// A cull submission needs more compaction scratch than the device binds as
    /// one storage buffer, so draws go out in cull-arrival order.
    CullPlanOverCapacity {
        /// Batches in the submission.
        batches: u32,
        /// Instances in the submission.
        instances: u32,
    },
    /// A CPU pick ran while the CPU pick cache is disabled, so it found nothing.
    PickCacheDisabled,
    /// A texture was bound into a slot that wants the other colour space.
    TextureSlotMismatch {
        /// Raw id of the texture.
        texture: u64,
        /// The slot field, as a consumer writes it.
        slot: &'static str,
        /// The space the texture was uploaded in.
        found: &'static str,
        /// The `TextureData` constructor that produces what the slot needs.
        constructor: &'static str,
    },
    /// An item named a handle whose resource has been freed, so it was skipped.
    StaleHandle {
        /// What the handle names, for example `"mesh"`.
        kind: &'static str,
        /// Slot index of the handle.
        index: u32,
        /// Where it was used, for example `"foreground item"`.
        context: &'static str,
    },
    /// More per-instance custom data was submitted than the buffer holds; the
    /// excess instances read the zero block.
    CustomDataOverflow {
        /// Capacity of the buffer, in blocks.
        capacity: u32,
    },
    /// A derived mesh buffer would exceed the device's `max_buffer_size`, so the
    /// view that needs it was skipped.
    BufferOverDeviceLimit {
        /// Slot index of the mesh.
        mesh: u32,
        /// Which buffer, for example `"edge index"`.
        buffer: &'static str,
        /// The size the buffer would have had.
        bytes: u64,
    },
    /// `ShadowSettings::atlas_resolution` was outside what the renderer
    /// allocates on this device (256 to 8192, the device's largest 2D texture,
    /// an even number), so the cascade atlas was made at `used` instead.
    ShadowAtlasClamped {
        /// The resolution the settings asked for.
        requested: u32,
        /// The atlas side allocated.
        used: u32,
    },
    /// More clip objects clip geometry than the clip uniforms hold; the ones
    /// past the limit do not clip and get no section cap.
    ClipObjectsOverCap {
        /// Enabled clipping planes in the frame. The first 6 apply.
        planes: u32,
        /// Enabled clipping boxes, spheres and cylinders in the frame. The
        /// first 4 apply.
        volumes: u32,
    },
    /// `RendererConfig::scene_depth_format` asked for a format the device
    /// does not support, so the scene depth uses `used` instead.
    SceneDepthFormatFallback {
        /// The format the config asked for, as its wgpu debug name.
        requested: String,
        /// The format the renderer uses, as its wgpu debug name.
        used: String,
        /// The device feature the requested format needs.
        feature: &'static str,
    },
    /// The GPU device was lost.
    DeviceLost {
        /// The reason the backend gave.
        reason: String,
    },
    /// Two overlay layers in one frame name some of the same items without one
    /// containing the other, so the later one was ignored and its items drew
    /// as if it were not there.
    OverlayLayersOverlap {
        /// Index in `OverlayFrame::layers` of the layer that was ignored.
        ignored: u32,
        /// Index of the earlier layer it overlaps, which was kept.
        kept: u32,
    },
    /// Overlay layers were nested deeper than the renderer composites; the
    /// layers past the limit drew flat, their style applied to each member.
    OverlayLayersTooDeep {
        /// How deep the renderer nests layers.
        limit: u32,
        /// Layers this frame that drew flat.
        flattened: u32,
    },
    /// The frame's overlays read the destination (a destination-reading blend
    /// or a backdrop blur), but it was drawn into a host-owned render pass,
    /// which cannot be ended to copy the destination; those blends drew as
    /// source-over and the backdrops were not drawn.
    OverlayDestinationReadsOnPath,
    /// A notice reported by a plugin.
    Plugin {
        /// The key the plugin reported it under; repeats of one key coalesce.
        key: &'static str,
        /// The plugin's message.
        message: String,
    },
}

impl fmt::Display for NoticeDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PostEffectSlotTaken { slot, producer } => write!(
                f,
                "post-effect producer '{producer}' overrides the enabled built-in {slot} slot; \
                 switch the built-in off when replacing it"
            ),
            Self::ForegroundUnsupportedOnPath => f.write_str(
                "foreground_items are not drawn by paint()/paint_viewport(): the foreground pass \
                 needs its own cleared depth attachment, which a host-owned render pass cannot \
                 provide. Use the HDR path (post-processing enabled) or an owned-encoder render \
                 path.",
            ),
            Self::ItemsSkippedOnLdr { plugin } => write!(
                f,
                "'{plugin}' items are not drawn on the LDR pipeline (PipelineMode::Direct): the \
                 plugin does not opt into it via ItemTypePlugin::draws_ldr. Opt in (with an \
                 LDR-format pipeline) or set effects.display.mode = PipelineMode::Hdr to render \
                 them."
            ),
            Self::EncodeSkippedOnPath { plugin } => write!(
                f,
                "'{plugin}' passes after the scene were skipped: paint_viewport() draws into a \
                 host-owned render pass, which cannot be ended for them. Use pass().prepare and \
                 pass().paint for one viewport, an owned-encoder path, or the HDR path."
            ),
            Self::VolumeTransparencyOnLdr => f.write_str(
                "transparent volume meshes are not drawn on the LDR pipeline \
                 (PipelineMode::Direct): they require the OIT pass, which only exists in the HDR \
                 pipeline. Set effects.display.mode = PipelineMode::Hdr.",
            ),
            Self::PostPaintSkipped { reason } => write!(
                f,
                "GPU plugin post_paint was not called: {reason}. It runs on HDR frames and on \
                 LDR frames drawn into a target with TEXTURE_BINDING."
            ),
            Self::AutoExposureOnLdr => f.write_str(
                "ExposureMode::Automatic has no meter on the LDR pipeline \
                 (PipelineMode::Direct, or a host-owned render pass): the frame was drawn at the \
                 passthrough exposure. Use a fixed exposure (Manual or PhysicalCamera) there, or \
                 set effects.display.mode = PipelineMode::Hdr.",
            ),
            Self::CullPlanOverCapacity { batches, instances } => write!(
                f,
                "cull submission of {batches} batches and {instances} instances needs more \
                 compaction scratch than max_storage_buffer_binding_size; draws submit in \
                 cull-arrival order, so frames are not bit-reproducible"
            ),
            Self::PickCacheDisabled => f.write_str(
                "renderer.pick()/pick_rect() was called but the CPU pick cache is disabled; \
                 enable it with ViewportRenderer::set_cpu_pick_cache(true)",
            ),
            Self::TextureSlotMismatch {
                texture,
                slot,
                found,
                constructor,
            } => write!(
                f,
                "{slot} was given a texture uploaded as {found}; build it with {constructor} \
                 (texture {texture})"
            ),
            Self::StaleHandle {
                kind,
                index,
                context,
            } => write!(
                f,
                "{context} names a freed {kind} (index {index}); the item was skipped"
            ),
            Self::CustomDataOverflow { capacity } => write!(
                f,
                "per-instance custom-data buffer overflowed its {capacity} blocks; excess \
                 instances fell back to the zero block"
            ),
            Self::BufferOverDeviceLimit {
                mesh,
                buffer,
                bytes,
            } => write!(
                f,
                "{buffer} buffer for mesh {mesh} would be {bytes} bytes, over the device's \
                 max_buffer_size; the view that needs it was skipped"
            ),
            Self::ShadowAtlasClamped { requested, used } => write!(
                f,
                "shadow atlas_resolution {requested} is outside what this device takes \
                 (256 to 8192, within its max_texture_dimension_2d, even); the atlas is {used} \
                 square"
            ),
            Self::ClipObjectsOverCap { planes, volumes } => write!(
                f,
                "the frame enables {planes} clip planes and {volumes} clip volumes; only the \
                 first 6 planes and 4 volumes clip, and the rest are ignored"
            ),
            Self::SceneDepthFormatFallback {
                requested,
                used,
                feature,
            } => write!(
                f,
                "scene depth format {requested} needs Features::{feature}, which the device was \
                 not created with; the scene depth is {used}. Request the feature (it is in \
                 ViewportRenderer::recommended_device_features) to use {requested}."
            ),
            Self::DeviceLost { reason } => write!(f, "GPU device lost: {reason}"),
            Self::OverlayLayersOverlap { ignored, kept } => write!(
                f,
                "overlay layer {ignored} names some of layer {kept}'s items without containing \
                 them or lying inside them, so layer {ignored} was ignored. Nest layers so that \
                 each one's ranges either hold another's or miss it."
            ),
            Self::OverlayLayersTooDeep { limit, flattened } => write!(
                f,
                "{flattened} overlay layers sit more than {limit} deep and drew flat: their \
                 opacity, blend and colour filter were applied to each member, and their blur, \
                 shadow and backdrop were not drawn."
            ),
            Self::OverlayDestinationReadsOnPath => f.write_str(
                "overlays that read the destination were drawn into a host-owned render pass \
                 (paint()/paint_viewport()), which cannot be ended to copy it: \
                 destination-reading blends drew as source-over and backdrops were not drawn. \
                 Use an owned-encoder render path.",
            ),
            Self::Plugin { key, message } => write!(f, "{key}: {message}"),
        }
    }
}

/// One reported condition, with every repeat of it coalesced in.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Notice {
    /// How serious it is.
    pub severity: Severity,
    /// Which part of the renderer reported it.
    pub category: NoticeCategory,
    /// What it is about.
    pub detail: NoticeDetail,
    /// The renderer frame it was first reported on.
    pub first_frame: u64,
    /// The renderer frame it was last reported on.
    pub last_frame: u64,
    /// How many times it was reported, including the first.
    pub count: u64,
}

impl Notice {
    /// A notice reported once, on `frame`.
    pub fn new(
        severity: Severity,
        category: NoticeCategory,
        detail: NoticeDetail,
        frame: u64,
    ) -> Self {
        Self {
            severity,
            category,
            detail,
            first_frame: frame,
            last_frame: frame,
            count: 1,
        }
    }
}

impl fmt::Display for Notice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.detail, f)
    }
}

/// The notices waiting to be taken, by severity, and how many were dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NoticeCounts {
    /// Waiting notices of severity [`Severity::Info`].
    pub info: u32,
    /// Waiting notices of severity [`Severity::Warning`].
    pub warning: u32,
    /// Waiting notices of severity [`Severity::Error`].
    pub error: u32,
    /// Notices pushed out of the full buffer since the last take.
    pub dropped: u32,
}

impl NoticeCounts {
    /// Waiting notices of every severity.
    pub fn total(&self) -> u32 {
        self.info + self.warning + self.error
    }
}
