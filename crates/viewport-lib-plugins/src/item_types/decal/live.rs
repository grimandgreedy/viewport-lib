//! Decals that live across frames: a lifetime, a fade-out, and UV animation.
//!
//! [`DecalItem`] is submitted fresh every frame. An impact mark that should
//! fade after ten seconds, or a decal that scrolls, needs something to hold
//! its age between frames, which is what [`LiveDecals`] does. The application
//! owns one, advances it each frame, and submits what it collects.

use super::types::{DecalAnimation, DecalItem};

/// Opaque handle returned by [`LiveDecals::add`] and its siblings.
///
/// Pass to [`LiveDecals::remove`] to delete the decal before it expires.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecalHandle(u64);

/// One decal held by a [`LiveDecals`], with optional lifetime and animation.
pub struct LiveDecal {
    id: u64,
    /// The base decal parameters. `uv_offset` and `uv_scale` are recomputed
    /// from `animation` each frame; all other fields are used as-is.
    pub item: DecalItem,
    /// Optional UV animation. See [`DecalAnimation`].
    pub animation: Option<DecalAnimation>,
    /// Total lifetime in seconds. `None` = permanent.
    pub lifetime: Option<f32>,
    /// How many seconds the fade-out lasts at the end of the lifetime.
    /// Must be <= `lifetime`. Default: 20% of lifetime when 0.0.
    pub fade_duration: f32,
    /// Elapsed time in seconds since the decal was added.
    pub age: f32,
}

/// A set of decals that persist across frames.
///
/// Call [`update`](Self::update) once per frame with the frame delta-time to
/// advance ages and drop expired decals, then [`collect`](Self::collect) to
/// get this frame's [`DecalItem`]s:
///
/// ```ignore
/// decals.update(dt);
/// frame.scene.items_mut::<DecalItem>().extend(decals.collect());
/// ```
#[derive(Default)]
pub struct LiveDecals {
    decals: Vec<LiveDecal>,
    next_id: u64,
}

impl LiveDecals {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a permanent decal. Returns a handle for later removal.
    pub fn add(&mut self, item: DecalItem) -> DecalHandle {
        self.push(item, None, None, 0.0)
    }

    /// Add a decal that fades and is removed after `lifetime` seconds.
    pub fn add_with_lifetime(
        &mut self,
        item: DecalItem,
        lifetime: f32,
        fade_duration: f32,
    ) -> DecalHandle {
        self.push(item, Some(lifetime), None, fade_duration)
    }

    /// Add a decal with an animation and an optional lifetime.
    pub fn add_animated(
        &mut self,
        item: DecalItem,
        animation: DecalAnimation,
        lifetime: Option<f32>,
    ) -> DecalHandle {
        self.push(item, lifetime, Some(animation), 0.0)
    }

    /// Remove a decal by handle. No-op if the handle is no longer valid.
    pub fn remove(&mut self, handle: DecalHandle) {
        self.decals.retain(|d| d.id != handle.0);
    }

    /// The decals currently held, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &LiveDecal> {
        self.decals.iter()
    }

    /// How many decals are held.
    pub fn len(&self) -> usize {
        self.decals.len()
    }

    /// `true` when no decal is held.
    pub fn is_empty(&self) -> bool {
        self.decals.is_empty()
    }

    /// Advance every decal by `dt` seconds and drop the expired ones.
    ///
    /// Call once per frame before [`collect`](Self::collect).
    pub fn update(&mut self, dt: f32) {
        for ld in &mut self.decals {
            ld.age += dt;
        }
        self.decals
            .retain(|ld| ld.lifetime.is_none_or(|lt| ld.age < lt));
    }

    /// Build this frame's [`DecalItem`] list.
    ///
    /// Applies lifetime fading (alpha ramps to 0 in the last 20% of life, or
    /// over `fade_duration` when set) and computes UV offset and scale for
    /// animated decals.
    pub fn collect(&self) -> Vec<DecalItem> {
        self.decals
            .iter()
            .map(|ld| {
                let mut item = ld.item.clone();

                // Fade out at the end of lifetime.
                if let Some(lt) = ld.lifetime {
                    let fade = if ld.fade_duration > 0.0 {
                        ld.fade_duration.min(lt)
                    } else {
                        lt * 0.2
                    };
                    let time_left = lt - ld.age;
                    if time_left < fade {
                        item.alpha *= (time_left / fade).clamp(0.0, 1.0);
                    }
                }

                if let Some(anim) = &ld.animation {
                    match anim {
                        DecalAnimation::UvScroll { vx, vy } => {
                            // Accumulate offset from base, wrapping in [0, 1].
                            item.uv_offset[0] =
                                (ld.item.uv_offset[0] + vx * ld.age).rem_euclid(1.0);
                            item.uv_offset[1] =
                                (ld.item.uv_offset[1] + vy * ld.age).rem_euclid(1.0);
                        }
                        DecalAnimation::SpriteSheet { cols, rows, fps } => {
                            let total = cols * rows;
                            let frame = ((ld.age * fps) as u32).rem_euclid(total.max(1));
                            let col = frame % cols;
                            let row = frame / cols;
                            item.uv_scale = [1.0 / *cols as f32, 1.0 / *rows as f32];
                            item.uv_offset = [col as f32 / *cols as f32, row as f32 / *rows as f32];
                        }
                    }
                }

                item
            })
            .collect()
    }

    fn push(
        &mut self,
        item: DecalItem,
        lifetime: Option<f32>,
        animation: Option<DecalAnimation>,
        fade_duration: f32,
    ) -> DecalHandle {
        let id = self.next_id;
        self.next_id += 1;
        self.decals.push(LiveDecal {
            id,
            item,
            animation,
            lifetime,
            fade_duration,
            age: 0.0,
        });
        DecalHandle(id)
    }
}
