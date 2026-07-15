//! The rendering context: GPU device, window surface, and resource registries
//! (textures, materials, fonts, targets). Game-agnostic — the logical size is
//! chosen by the caller, and no game resources are assumed.

use std::sync::Arc;

use wgpu::{
    Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor, PowerPreference, Queue,
    RequestAdapterOptions, Sampler, Surface, SurfaceConfiguration, TextureUsages, TextureView,
};
use winit::window::Window;

use crate::frame::FrameScratch;
use crate::material::{Material, MaterialDesc, TextureSource, create_fallback_texture};
use crate::sprite::{Filter, GpuTexture, SpriteBatch, create_gpu_texture};
use crate::target::{SCENE_FORMAT, SceneTarget};
use crate::text::TextRenderer;
use crate::{FontId, MaterialId, TargetId, TextStyle, TextureId, UniformValue, Vk2dError};

/// Which GPU backend the context should request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Request Vulkan specifically (the crate's primary target).
    Vulkan,
    /// Let wgpu pick any available backend (Vulkan/Metal/DX12/GL).
    Auto,
}

impl Backend {
    fn backends(self) -> Backends {
        match self {
            Backend::Vulkan => Backends::VULKAN,
            Backend::Auto => Backends::all(),
        }
    }
}

/// Configuration for [`Context::new`].
#[derive(Debug, Clone, Copy)]
pub struct ContextConfig {
    /// The logical resolution the world is authored at. The scene is drawn at
    /// this size and upscaled to the window; the app decides it (there is no
    /// baked-in default game resolution).
    pub logical_size: (u32, u32),
    /// Which GPU backend to request.
    pub prefer_backend: Backend,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            logical_size: (1280, 720),
            prefer_backend: Backend::Vulkan,
        }
    }
}

/// Text metrics from [`Context::measure_text_ext`]: the same `(width, height)`
/// as [`Context::measure_text`], plus `offset_y` — the baseline's distance
/// from the line's top, in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextMetricsExt {
    /// The pen-advance width of the measured text, in logical pixels.
    pub width: f32,
    /// The baked line height at the requested size, in logical pixels.
    pub height: f32,
    /// Distance from the line's top down to its baseline, in logical pixels —
    /// the same offset [`crate::Frame::text`] adds internally when drawing.
    pub offset_y: f32,
}

/// Owns the GPU device/queue, the window surface, and (as later tasks add them)
/// the texture/material/font/target registries. Construct one per window with
/// [`Context::new`].
pub struct Context {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    surface: Surface<'static>,
    pub(crate) config: SurfaceConfiguration,
    /// App-chosen logical resolution (the scene target size).
    pub(crate) logical_size: (u32, u32),
    adapter_info: String,
    /// Offscreen render targets, indexed by `TargetId`.
    pub(crate) targets: Vec<SceneTarget>,
    /// Textures uploaded by the app, indexed by `TextureId`.
    pub(crate) textures: Vec<GpuTexture>,
    /// Shared sprite pipeline + per-frame buffers. Built lazily on first
    /// texture load (it targets the scene format).
    pub(crate) sprites: SpriteBatch,
    /// Materials (WGSL shaders), indexed by `MaterialId`.
    pub(crate) materials: Vec<Material>,
    /// Vector-primitive batch (rects/lines/circles/triangles).
    pub(crate) shapes: crate::shapes::ShapeBatch,
    /// Text renderers, indexed by `FontId`. Each renderer lazily bakes and
    /// caches one glyph atlas per distinct requested pixel size — see
    /// `TextRenderer`'s own doc comment in `text.rs`.
    pub(crate) fonts: Vec<TextRenderer>,
    /// Shared 1x1 white texture substituted into any declared material
    /// texture slot that hasn't been bound yet. Built lazily on first use by
    /// [`Context::fallback_texture`] (most apps never touch texture
    /// materials, so most contexts never pay for it).
    fallback_texture: Option<(TextureView, Sampler)>,
    /// Bind groups for sourcing a render target as a sprite ([`crate::Frame::target_sprite`]),
    /// indexed parallel to `targets` and built lazily the first time each
    /// target is drawn that way — mirrors how a `GpuTexture`'s bind group is
    /// built once at load rather than rebuilt per draw. Targets never resize
    /// in this crate, so a lazily-built entry stays valid for the target's
    /// whole lifetime.
    pub(crate) target_sprite_bind_groups: Vec<Option<wgpu::BindGroup>>,
    /// CPU command/run storage recycled by successive immediate-mode frames.
    pub(crate) frame_scratch: FrameScratch,
}

impl Context {
    /// Create a context for `window`. Requests the configured backend, a
    /// high-performance adapter, and a device; configures the surface. Returns
    /// a [`Vk2dError`] on any failure — never panics.
    pub fn new(window: Arc<Window>, cfg: ContextConfig) -> Result<Self, Vk2dError> {
        pollster::block_on(Self::new_async(window, cfg))
    }

    async fn new_async(window: Arc<Window>, cfg: ContextConfig) -> Result<Self, Vk2dError> {
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Err(Vk2dError::SurfaceCreation(
                "window has a zero-sized drawable area".to_string(),
            ));
        }

        let mut descriptor = InstanceDescriptor::new_without_display_handle();
        descriptor.backends = cfg.prefer_backend.backends();
        let instance = Instance::new(descriptor);

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| Vk2dError::SurfaceCreation(e.to_string()))?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|e| Vk2dError::AdapterUnavailable(e.to_string()))?;

        let info = adapter.get_info();
        let adapter_info = format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type);

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .await
            .map_err(|e| Vk2dError::DeviceRequest(e.to_string()))?;

        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .ok_or_else(|| {
                Vk2dError::SurfaceCreation("adapter does not support this surface".to_string())
            })?;
        surface.configure(&device, &config);

        // Sprites and shapes render into the scene target, so they target the
        // scene format (not the surface format).
        let sprites = SpriteBatch::new(&device, SCENE_FORMAT);
        let shapes = crate::shapes::ShapeBatch::new(&device, SCENE_FORMAT);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            logical_size: cfg.logical_size,
            adapter_info,
            targets: Vec::new(),
            textures: Vec::new(),
            sprites,
            materials: Vec::new(),
            shapes,
            fonts: Vec::new(),
            fallback_texture: None,
            target_sprite_bind_groups: Vec::new(),
            frame_scratch: FrameScratch::default(),
        })
    }

    /// The shared 1x1 white fallback view/sampler, created on first use.
    fn fallback_texture(&mut self) -> (&TextureView, &Sampler) {
        let (view, sampler) = self
            .fallback_texture
            .get_or_insert_with(|| create_fallback_texture(&self.device, &self.queue));
        (view, sampler)
    }

    /// Bake a TTF (decoded bytes) into a glyph atlas at `atlas_px` and register
    /// it. `Err(ShaderCompile)` if the font is unparsable — never a panic.
    pub fn load_font(&mut self, ttf_bytes: &[u8], atlas_px: f32) -> Result<FontId, Vk2dError> {
        let renderer = TextRenderer::new(
            &self.device,
            &self.queue,
            SCENE_FORMAT,
            ttf_bytes,
            atlas_px,
            4096,
        )?;
        let id = FontId(self.fonts.len() as u32);
        self.fonts.push(renderer);
        Ok(id)
    }

    /// Measure `text` in `font` at `style.size`: `(width, height)` in logical
    /// pixels. Returns `(0, 0)` for an unknown font.
    pub fn measure_text(&self, font: FontId, text: &str, style: TextStyle) -> (f32, f32) {
        self.fonts
            .get(font.0 as usize)
            .map(|f| f.measure(&self.device, &self.queue, text, style.size))
            .unwrap_or((0.0, 0.0))
    }

    /// Measure `text` in `font` at `style.size`, plus `offset_y`: the distance
    /// from the line's top (the `pos.y` a caller would pass to
    /// [`crate::Frame::text`]) down to its baseline, in logical pixels. Callers
    /// that lay out text against a baseline (aligning mixed-size runs, drawing
    /// an underline, or vertically centering multi-line labels) need this in
    /// addition to the raw `(width, height)` [`Context::measure_text`] gives.
    /// `offset_y` is computed by the exact same scale expression the draw path
    /// ([`crate::Frame::text`]) uses, so it always matches what actually
    /// renders. Returns all-zero metrics for an unknown font.
    pub fn measure_text_ext(&self, font: FontId, text: &str, style: TextStyle) -> TextMetricsExt {
        let Some(renderer) = self.fonts.get(font.0 as usize) else {
            return TextMetricsExt {
                width: 0.0,
                height: 0.0,
                offset_y: 0.0,
            };
        };
        let (width, height) = renderer.measure(&self.device, &self.queue, text, style.size);
        TextMetricsExt {
            width,
            height,
            offset_y: renderer.baseline_offset(&self.device, &self.queue, style.size),
        }
    }

    /// Raw wgpu device — an escape hatch for the optional `egui` integration,
    /// which must build its own pipelines. Feature-gated so the wgpu type never
    /// leaks into the default public API. Not a stable surface for general use.
    #[cfg(feature = "egui")]
    #[doc(hidden)]
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Raw wgpu queue — see [`Context::wgpu_device`].
    #[cfg(feature = "egui")]
    #[doc(hidden)]
    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The swapchain texture format — the `egui` overlay paints to it.
    #[cfg(feature = "egui")]
    #[doc(hidden)]
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Compile a WGSL material and register it. The WGSL is compiled to SPIR-V
    /// now (at load), so a WGSL syntax/validation error is returned here — not
    /// at first draw. The render pipelines themselves are built lazily on first
    /// use (fullscreen on first [`crate::Frame::material_fullscreen`], sprite on
    /// first [`crate::Frame::material_sprite`]); this lets a material intended
    /// only for `material_sprite` — whose `fs_main` takes the sprite vertex
    /// stage's uv/tint inputs, which the bufferless fullscreen `vs_main` cannot
    /// supply — register without a pipeline-stage-match failure it would never
    /// hit in practice. Materials render into the scene target (scene format).
    /// Any declared [`MaterialDesc::textures`] slot starts bound to the shared
    /// 1x1 white fallback until [`Context::set_material_texture`] binds a real
    /// texture.
    pub fn load_material(&mut self, desc: MaterialDesc) -> Result<MaterialId, Vk2dError> {
        self.fallback_texture();
        let fallback = self
            .fallback_texture
            .as_ref()
            .map(|(v, s)| (v, s))
            .expect("fallback_texture() just initialized this");
        let material = Material::new(&self.device, &desc, SCENE_FORMAT, fallback)?;
        let id = MaterialId(self.materials.len() as u32);
        self.materials.push(material);
        Ok(id)
    }

    /// Push a named uniform to a material. Unknown material ids or names are
    /// ignored (the no-panic contract).
    pub fn set_material_uniform(&self, material: MaterialId, name: &str, value: UniformValue) {
        if let Some(mat) = self.materials.get(material.0 as usize) {
            mat.set_uniform(&self.queue, name, value);
        }
    }

    /// Bind a loaded texture to a declared texture slot (by name) of
    /// `material`. Resolves `texture` to its GPU view/sampler and rebuilds the
    /// material's bind group. Unknown material id, unknown slot name, or
    /// unknown texture id is a no-op (the no-panic contract).
    pub(crate) fn set_material_texture(
        &mut self,
        material: MaterialId,
        name: &str,
        texture: TextureId,
    ) {
        let Some(gpu) = self.textures.get(texture.0 as usize) else {
            return;
        };
        // Clone the view/sampler: wgpu's `TextureView`/`Sampler` are cheap
        // reference-counted handles, so this does not duplicate GPU memory.
        let view = gpu.view.clone();
        let sampler = gpu.sampler.clone();
        let fallback = self
            .fallback_texture
            .get_or_insert_with(|| create_fallback_texture(&self.device, &self.queue));
        let fallback = (&fallback.0, &fallback.1);
        if let Some(mat) = self.materials.get_mut(material.0 as usize) {
            mat.set_texture(
                &self.device,
                name,
                TextureSource::Texture(texture),
                view,
                sampler,
                fallback,
            );
        }
    }

    /// Bind a render target's finished color texture to a declared texture
    /// slot (by name) of `material` — the multi-pass hookup (bloom ping-pong,
    /// scene-reading post-process). Mirrors [`Context::set_material_texture`]
    /// but sources the view/sampler from `targets[target]` instead of the
    /// texture registry. Unknown material id, unknown slot name, or unknown
    /// target id is a no-op (the no-panic contract).
    pub(crate) fn set_material_target(
        &mut self,
        material: MaterialId,
        name: &str,
        target: TargetId,
    ) {
        let Some(scene) = self.targets.get(target.0 as usize) else {
            return;
        };
        // Clone the view/sampler: cheap reference-counted handles (see
        // `set_material_texture`), not a GPU memory duplication.
        let view = scene.color_view().clone();
        let sampler = scene.color_sampler().clone();
        let fallback = self
            .fallback_texture
            .get_or_insert_with(|| create_fallback_texture(&self.device, &self.queue));
        let fallback = (&fallback.0, &fallback.1);
        if let Some(mat) = self.materials.get_mut(material.0 as usize) {
            mat.set_texture(
                &self.device,
                name,
                TextureSource::Target(target),
                view,
                sampler,
                fallback,
            );
        }
    }

    /// Upload decoded RGBA bytes (`width`x`height`, 4 bytes/texel, row-major) as
    /// a texture and return its handle. The app decodes the image; vk2d never
    /// loads files.
    pub fn load_texture_rgba(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        filter: Filter,
    ) -> TextureId {
        let texture = create_gpu_texture(
            &self.device,
            &self.queue,
            &self.sprites.bind_group_layout,
            bytes,
            width,
            height,
            filter,
        );
        let id = TextureId(self.textures.len() as u32);
        self.textures.push(texture);
        id
    }

    /// Create a `width`x`height` offscreen render target. Draw into it with
    /// [`Context::begin_target_frame`], then sample it as a material input with
    /// [`crate::Frame::bind_material_target`] or present it via the swapchain.
    ///
    /// The default scene target (index 0) is reserved first, so a target you
    /// create always has index ≥ 1 and can never alias the scene the swapchain
    /// path renders into — creating a target before your first frame is safe.
    pub fn create_target(&mut self, width: u32, height: u32) -> TargetId {
        // Reserve targets[0] for the scene before appending an app target, so an
        // app-created target never lands at index 0 (which begin_frame renders
        // into and blits — sampling it would be a self-usage conflict).
        self.ensure_scene();
        self.push_target(width, height)
    }

    /// Append a target and return its id, without reserving the scene target.
    /// The raw builder shared by `create_target` and `ensure_scene`.
    pub(crate) fn push_target(&mut self, width: u32, height: u32) -> TargetId {
        let target = SceneTarget::new(&self.device, width, height, self.config.format);
        let id = TargetId(self.targets.len() as u32);
        self.targets.push(target);
        // Kept parallel to `targets`: `target_sprite`'s bind-group cache is
        // built lazily per index on first use, not here.
        self.target_sprite_bind_groups.push(None);
        id
    }

    /// The cached bind group for sourcing `targets[index]` as a sprite
    /// ([`crate::Frame::target_sprite`]), building it on first use. Reused on
    /// later calls the same way a `GpuTexture`'s bind group is built once at
    /// load — targets never resize in this crate, so the cached entry stays
    /// valid for the target's whole lifetime. `index` must already be a valid
    /// `targets` index; callers (`Frame::target_sprite`) check that first.
    pub(crate) fn target_sprite_bind_group(&mut self, index: usize) -> &wgpu::BindGroup {
        if self.target_sprite_bind_groups[index].is_none() {
            let scene = &self.targets[index];
            let view = scene.color_view();
            let sampler = scene.color_sampler();
            let bind_group = self
                .sprites
                .build_source_bind_group(&self.device, view, sampler);
            self.target_sprite_bind_groups[index] = Some(bind_group);
        }
        self.target_sprite_bind_groups[index]
            .as_ref()
            .expect("just inserted above")
    }

    /// Prepare one [`crate::Frame::material_sprite`] run: ensure `material`'s
    /// sprite-shaded pipeline is built (lazily, on first use) and build a fresh
    /// bind group binding `texture` into the material's slot 0. Returns the
    /// owned bind group, or `None` if the material or texture id is unknown or
    /// the material declares no texture slot (all no-ops the draw skips).
    ///
    /// Called from `Frame::render_scene`'s pre-warm phase — BEFORE the scene
    /// pass borrows `ctx` immutably — because both the pipeline build and the
    /// fallback-texture lazy-init need `&mut self`. The returned bind group is a
    /// standalone owned resource (it borrows the device/layout only at
    /// creation), so it can be held and bound during the immutable pass.
    pub(crate) fn prepare_material_sprite(
        &mut self,
        material: MaterialId,
        texture: TextureId,
    ) -> Option<wgpu::BindGroup> {
        // The material must exist and declare a texture slot for the sprite.
        let mat = self.materials.get(material.0 as usize)?;
        if !mat.has_texture_slot() {
            return None;
        }
        let gpu = self.textures.get(texture.0 as usize)?;
        // Cheap ref-counted clones (see `set_material_texture`): no GPU-memory
        // duplication.
        let view = gpu.view.clone();
        let sampler = gpu.sampler.clone();
        // Build the sprite-shaded pipeline (lazy, cached on the material).
        self.materials
            .get_mut(material.0 as usize)?
            .sprite_pipeline(&self.device)?;
        // Ensure the shared fallback exists, then build this run's bind group.
        let fallback = self
            .fallback_texture
            .get_or_insert_with(|| create_fallback_texture(&self.device, &self.queue));
        let fallback = (&fallback.0, &fallback.1);
        let mat = self.materials.get(material.0 as usize)?;
        Some(mat.build_sprite_bind_group(&self.device, &view, &sampler, fallback))
    }

    /// A human-readable description of the selected adapter (name, backend,
    /// device type) — useful for logging and diagnostics.
    pub fn adapter_info(&self) -> &str {
        &self.adapter_info
    }

    /// The current swapchain size in physical pixels.
    pub fn surface_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Read one target's texel at `(x, y)` back as `[r, g, b, a]` bytes — a
    /// doc-hidden test hook so integration tests can assert what a finished
    /// target actually contains (or what a `target_sprite` composite produced
    /// when the readback target is the composite destination). Not a stable
    /// API and not part of the drawing path; returns `None` for an unknown
    /// target or out-of-range coordinate. Blocks on the copy + map.
    #[doc(hidden)]
    pub fn read_target_pixel(&self, target: TargetId, x: u32, y: u32) -> Option<[u8; 4]> {
        let scene = self.targets.get(target.0 as usize)?;
        let (w, h) = scene.size();
        if x >= w || y >= h {
            return None;
        }
        // wgpu requires bytes_per_row to be a multiple of 256; copy the target
        // into a row-padded staging buffer and index the requested texel out of
        // it. A single probe copies the whole texture rather than a sub-region
        // to keep the copy's offset/extent alignment trivially valid.
        let unpadded_bpr = 4 * w;
        let padded_bpr = unpadded_bpr.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vk2d.readback"),
            size: (padded_bpr * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vk2d.readback.encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: scene.color_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let data = slice.get_mapped_range();
        let row = (y * padded_bpr) as usize;
        let px = row + (x * 4) as usize;
        let pixel = [data[px], data[px + 1], data[px + 2], data[px + 3]];
        drop(data);
        buffer.unmap();
        Some(pixel)
    }

    /// Reconfigure the surface after the window resized. A zero dimension is
    /// ignored (minimized window).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.config.usage = TextureUsages::RENDER_ATTACHMENT;
        self.surface.configure(&self.device, &self.config);
    }

    /// Acquire the next swapchain texture, or `Err(SurfaceLost)` if the surface
    /// needs reconfiguration. Used internally by frame begin/present.
    pub(crate) fn acquire(&self) -> Result<wgpu::SurfaceTexture, Vk2dError> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
            _ => Err(Vk2dError::SurfaceLost),
        }
    }
}
