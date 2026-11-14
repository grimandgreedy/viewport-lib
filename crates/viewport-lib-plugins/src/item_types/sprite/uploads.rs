//! How a renderer uploads, writes and releases sprites.

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

use super::*;
use crate::item_types::{cast_bytes, host, plugin_mut};

/// Ranged writes into a stored sprite batch, one implementation per channel.
///
/// The positions are their own vertex stream and the rest of a sprite is one
/// interleaved record, so a particle feed that only moves its sprites writes a
/// quarter of the bytes a full update would.
macro_rules! sprite_writes {
    ($marker:ty, channel = $channel:expr, encode = $encode:expr) => {
        impl viewport_lib::plugin_api::Writes<$marker> for ViewportRenderer {
            fn write_range(
                &mut self,
                _channel: $marker,
                queue: &gpu::Queue,
                id: SpriteSetId,
                first_element: u32,
                data: &[<$marker as viewport_lib::plugin_api::Channel>::Input],
            ) -> viewport_lib::error::ViewportResult<()> {
                let bytes = $encode(data);
                plugin_mut::<SpritePlugin>(self, TYPE_NAME)?.write_set_channel(
                    queue,
                    id,
                    $channel,
                    first_element,
                    &bytes,
                )
            }

            fn reserve(
                &mut self,
                _channel: $marker,
                device: &gpu::Device,
                queue: &gpu::Queue,
                id: SpriteSetId,
                capacity: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                plugin_mut::<SpritePlugin>(self, TYPE_NAME)?
                    .reserve_set(device, queue, id, capacity)
            }

            fn set_len(
                &mut self,
                _channel: $marker,
                id: SpriteSetId,
                len: u32,
            ) -> viewport_lib::error::ViewportResult<()> {
                plugin_mut::<SpritePlugin>(self, TYPE_NAME)?.set_set_len(id, len)
            }

            fn extent(
                &self,
                _channel: $marker,
                id: SpriteSetId,
            ) -> Option<viewport_lib::plugin_api::Extent> {
                self.item_type_plugin::<SpritePlugin>(TYPE_NAME)?
                    .set_extent(id, $channel)
            }
        }
    };
}

sprite_writes!(
    channels::Positions,
    channel = store::SpriteChannel::Positions,
    encode = cast_bytes
);
sprite_writes!(
    channels::Sprites,
    channel = store::SpriteChannel::Sprites,
    encode = encode_sprites
);
sprite_writes!(
    channels::SortKeys,
    channel = store::SpriteChannel::SortKeys,
    encode = cast_bytes
);
sprite_writes!(
    channels::Effects,
    channel = store::SpriteChannel::Effects,
    encode = effects::encode_effects
);

/// Sprites keep two stores behind one item struct: a batch drawn as one set,
/// and an instance set drawn per transform. `Uploads` carries one `Id` per
/// implementation, so it covers the plain set and the instance set keeps calls
/// of its own on [`SpriteInstanceUploads`].
impl viewport_lib::plugin_api::Uploads<SpriteItem> for ViewportRenderer {
    type Id = SpriteSetId;

    fn upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<SpriteSetId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME)?;
        Ok(host.plugin.upload_set(device, queue, host.resources, item))
    }

    fn begin_upload(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: SpriteItem,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME)?;
        Ok(host.plugin.begin_upload(
            &host.jobs,
            device,
            queue,
            host.resources,
            item,
            store::Retain::Set,
        ))
    }

    fn replace(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: SpriteSetId,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        let host = host::<SpritePlugin>(self, TYPE_NAME)?;
        host.plugin
            .replace_set(device, queue, host.resources, id, item)
    }
}

impl viewport_lib::plugin_api::Handles<SpriteSetId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<SpriteSetId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME)?;
        host.plugin.take_set_result(&host.jobs, job)
    }

    fn release(&mut self, id: SpriteSetId) -> bool {
        plugin_mut::<SpritePlugin>(self, TYPE_NAME).is_ok_and(|p| p.drop_set(id))
    }
}

/// The sprite instance-set store, which shares `SpriteItem` with the plain set
/// store and so cannot share its `Uploads` implementation. Its handle type is
/// its own, so taking a finished job and releasing a set go through
/// [`Handles`](viewport_lib::plugin_api::Handles) like every other store.
pub trait SpriteInstanceUploads {
    /// Upload an instance set, returning a handle valid until
    /// [`release`](viewport_lib::plugin_api::Handles::release).
    fn upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<SpriteInstanceSetId>;

    /// Start an off-thread upload of an instance set.
    fn begin_upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: SpriteItem,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId>;

    /// Replace the sprites behind an instance-set handle, keeping the handle.
    fn replace_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: SpriteInstanceSetId,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<()>;
}

impl SpriteInstanceUploads for ViewportRenderer {
    fn upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<SpriteInstanceSetId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME)?;
        Ok(host
            .plugin
            .upload_instance_set(device, queue, host.resources, item))
    }

    fn begin_upload_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        item: SpriteItem,
    ) -> viewport_lib::error::ViewportResult<viewport_lib::resources::JobId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME)?;
        Ok(host.plugin.begin_upload(
            &host.jobs,
            device,
            queue,
            host.resources,
            item,
            store::Retain::InstanceSet,
        ))
    }

    fn replace_sprite_instance_set(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        id: SpriteInstanceSetId,
        item: &SpriteItem,
    ) -> viewport_lib::error::ViewportResult<()> {
        let host = host::<SpritePlugin>(self, TYPE_NAME)?;
        host.plugin
            .replace_instance_set(device, queue, host.resources, id, item)
    }
}

impl viewport_lib::plugin_api::Handles<SpriteInstanceSetId> for ViewportRenderer {
    fn upload_result(
        &mut self,
        job: viewport_lib::resources::JobId,
    ) -> viewport_lib::error::ViewportResult<SpriteInstanceSetId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME)?;
        host.plugin.take_instance_set_result(&host.jobs, job)
    }

    fn release(&mut self, id: SpriteInstanceSetId) -> bool {
        plugin_mut::<SpritePlugin>(self, TYPE_NAME).is_ok_and(|p| p.drop_instance_set(id))
    }
}

/// Custom sprite materials: a [`MaterialPlugin`](viewport_lib::MaterialPlugin)
/// whose body draws a sprite's fragment rather than a mesh surface.
///
/// The body follows the particle fragment-hook contract, so one body draws
/// both. It defines
///
/// ```text
/// fn shade_sprite(f: SpriteFragment) -> vec4<f32>
/// ```
///
/// plus any helpers, and returns the sprite's base colour as straight-alpha
/// linear RGBA in place of `tint * texel`. The flash, outline and dissolve
/// apply after it, and lighting too for a lit batch. `SpriteFragment` starts
/// with the fields of the particle hook's `ParticleFragment`:
///
/// ```text
/// struct SpriteFragment {
///     uv:          vec2<f32>, // atlas UV
///     texel:       vec4<f32>, // the texture at uv, palette-swapped if the batch swaps
///     colour:      vec4<f32>, // the sprite's tint
///     age:         f32,       // always 0 for a sprite
///     age_sec:     f32,       // always 0 for a sprite
///     world:       vec3<f32>,
///     time:        f32,       // seconds, from the frame time
///     local_uv:    vec2<f32>, // 0..1 across the quad
///     uv_rect:     vec4<f32>, // the sprite's atlas rect
///     flash:       f32,       // this sprite's flash, dissolve and palette row
///     dissolve:    f32,
///     palette_row: u32,
///     uv_ddx:      vec2<f32>, // derivatives of uv, for textureSampleGrad
///     uv_ddy:      vec2<f32>,
///     user:        u32,       // the sprite's SpriteItem::user_values entry
/// };
/// ```
///
/// A body may instead or as well define
///
/// ```text
/// fn displace_sprite(v: SpriteVertex) -> vec3<f32>
/// ```
///
/// which returns where one corner of the quad goes, in world space:
///
/// ```text
/// struct SpriteVertex {
///     corner:      u32,       // 0 bottom left, 1 bottom right, 2 top left, 3 top right
///     local_uv:    vec2<f32>, // 0..1 across the quad
///     world:       vec3<f32>, // the corner in world space
///     anchor:      vec3<f32>, // the sprite's position
///     time:        f32,
///     base:        vec3<f32>, // midpoint of the quad's bottom edge
///     pivot:       vec2<f32>, // where anchor sits across the quad
///     size:        vec2<f32>, // the quad in world units
///     right:       vec3<f32>, // the quad's unit axes, after rotation
///     up:          vec3<f32>,
///     texel:       vec2<f32>, // world size of one texel as the draw mode draws it
///     user:        u32,       // the sprite's user value
///     tile:        u32,       // tiles only: the tile shown this frame
///     tile_placed: u32,       // tiles only: the tile or animation slot as placed
/// };
/// ```
///
/// Positions are relative to the renderer's world origin. The pick and the
/// outline follow the moved quad.
///
/// The bindings are the particle hook's too: up to four textures as
/// `htex0..htex3` (the plugin's `texture_count`, each item binding its own in
/// [`SpriteMaterial::textures`](super::SpriteMaterial)) read through
/// `sprite_samp` (the plugin's `sampler()`), and parameters in
/// `hook_params.lanes`, lane 0 reserved with the time in `.w` and parameter
/// `i` of the plugin's `param_layout` in lane `1 + i`. Declared parameters can
/// also be read by name as `params.<name>`; each item sets its values in
/// [`SpriteMaterial::params`](super::SpriteMaterial). At most seven
/// parameters. Every top-level identifier in the body is prefixed with the
/// plugin's name, and the body may read `camera`. It runs after
/// per-fragment discards, so sample with `textureSampleGrad` or
/// `textureSampleLevel`. `needs_back_hemisphere` and
/// `reads_vertex_attribute` do not apply to sprites and are ignored.
///
/// A batch selects a material through
/// [`SpriteEffects::material`](super::SpriteEffects::material).
pub trait SpriteMaterials {
    /// Register `plugin` as a sprite material, validating its body against
    /// the sprite and lit sprite shaders.
    ///
    /// # Errors
    ///
    /// [`ViewportError::ShadeNameTaken`](viewport_lib::error::ViewportError::ShadeNameTaken)
    /// when a sprite material of that name exists;
    /// [`ViewportError::ShadeShaderInvalid`](viewport_lib::error::ViewportError::ShadeShaderInvalid)
    /// when the name or a parameter is not a WGSL identifier, the body
    /// defines no `shade_sprite`, too many parameters or textures are
    /// declared, or a composed module fails validation.
    fn register_sprite_material(
        &mut self,
        device: &gpu::Device,
        plugin: &dyn viewport_lib::MaterialPlugin,
    ) -> viewport_lib::error::ViewportResult<SpriteMaterialId>;

    /// The sprite material registered as `name`.
    fn sprite_material_id(&self, name: &str) -> Option<SpriteMaterialId>;
}

impl SpriteMaterials for ViewportRenderer {
    fn register_sprite_material(
        &mut self,
        device: &gpu::Device,
        plugin: &dyn viewport_lib::MaterialPlugin,
    ) -> viewport_lib::error::ViewportResult<SpriteMaterialId> {
        let host = host::<SpritePlugin>(self, TYPE_NAME)?;
        host.plugin
            .register_material(device, host.resources, plugin)
    }

    fn sprite_material_id(&self, name: &str) -> Option<SpriteMaterialId> {
        self.item_type_plugin::<SpritePlugin>(TYPE_NAME)?
            .material_id(name)
    }
}

/// Registering sprite and tile deformers: WGSL bodies that move the corners
/// of other people's sprites and tiles, the 2D counterpart of the mesh
/// deformer registry.
///
/// A deformer's body defines
///
/// ```text
/// fn deform_sprite(v: SpriteVertex, slot: u32) -> vec3<f32>
/// ```
///
/// and returns where the corner goes, in world space. `SpriteVertex` is the
/// vertex hook's contract (see [`SpriteMaterials`]). `slot` is the slot of the
/// mesh deformer named by [`SpriteDeformerDesc::params`], whose parameters
/// the body reads as `deform_header.slot_params[slot * 4u + i]`: a plugin that
/// writes its mesh deformer's parameters once a frame through a
/// `DeformSlotHandle` drives its sprites and tiles with the same write.
///
/// Every registered deformer is composed into every sprite and tilemap
/// shader, so registering one rebuilds those pipelines once. It runs only on
/// the batches whose [`SpriteItem::deformers`] and the layers whose
/// `TilemapItem::deformers` hold it, in priority order and before a
/// material's `displace_sprite`, in every pass that draws the quad. A batch
/// or layer that holds none pays one branch per corner.
pub trait SpriteDeformers {
    /// Register a sprite and tile deformer, validating its body against every
    /// sprite and tilemap shader with the deformers already registered.
    ///
    /// # Errors
    ///
    /// [`ViewportError::ShadeNameTaken`](viewport_lib::error::ViewportError::ShadeNameTaken)
    /// when a sprite deformer of that name exists;
    /// [`ViewportError::ShadeShaderInvalid`](viewport_lib::error::ViewportError::ShadeShaderInvalid)
    /// when the name is not a WGSL identifier, the body defines no
    /// `deform_sprite`, [`MAX_SPRITE_DEFORMERS`] are already registered, or a
    /// composed module fails validation.
    fn register_sprite_deformer(
        &mut self,
        device: &gpu::Device,
        desc: &SpriteDeformerDesc,
    ) -> viewport_lib::error::ViewportResult<SpriteDeformerId>;

    /// The sprite deformer registered as `name`.
    fn sprite_deformer_id(&self, name: &str) -> Option<SpriteDeformerId>;
}

impl SpriteDeformers for ViewportRenderer {
    fn register_sprite_deformer(
        &mut self,
        device: &gpu::Device,
        desc: &SpriteDeformerDesc,
    ) -> viewport_lib::error::ViewportResult<SpriteDeformerId> {
        use crate::item_types::tilemap::{self, TilemapPlugin};
        let existing = plugin_mut::<SpritePlugin>(self, TYPE_NAME)?.deformers();
        let entry = deform::entry(existing, desc)?;
        let tiles = self
            .item_type_plugin::<TilemapPlugin>(tilemap::TYPE_NAME)
            .is_some();
        if tiles {
            plugin_mut::<TilemapPlugin>(self, tilemap::TYPE_NAME)?
                .check_deformer(device, &entry)?;
        }
        let id = plugin_mut::<SpritePlugin>(self, TYPE_NAME)?.register_deformer(device, desc)?;
        if tiles {
            plugin_mut::<TilemapPlugin>(self, tilemap::TYPE_NAME)?.add_deformer(entry);
        }
        Ok(id)
    }

    fn sprite_deformer_id(&self, name: &str) -> Option<SpriteDeformerId> {
        self.item_type_plugin::<SpritePlugin>(TYPE_NAME)?
            .deformers()
            .iter()
            .find(|e| e.name == name)
            .map(deform::id_of)
    }
}
