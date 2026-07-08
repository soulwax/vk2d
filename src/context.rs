//! The rendering context: GPU device, window surface, and resource registries
//! (textures, materials, fonts, targets). Game-agnostic — the logical size is
//! chosen by the caller, and no game resources are assumed.

use std::sync::Arc;

use wgpu::{
    Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor, PowerPreference, Queue,
    RequestAdapterOptions, Sampler, Surface, SurfaceConfiguration, TextureUsages, TextureView,
};
use winit::window::Window;

use crate::material::{Material, MaterialDesc, create_fallback_texture};
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
    /// Text renderers, indexed by `FontId` (one baked atlas each).
    pub(crate) fonts: Vec<TextRenderer>,
    /// Shared 1x1 white texture substituted into any declared material
    /// texture slot that hasn't been bound yet. Built lazily on first use by
    /// [`Context::fallback_texture`] (most apps never touch texture
    /// materials, so most contexts never pay for it).
    fallback_texture: Option<(TextureView, Sampler)>,
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
            .map(|f| f.measure(text, style.size))
            .unwrap_or((0.0, 0.0))
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
    /// now (at load), so a shader error is returned here — not at first draw.
    /// Materials render into the scene target (scene format). Any declared
    /// [`MaterialDesc::textures`] slot starts bound to the shared 1x1 white
    /// fallback until [`Context::set_material_texture`] binds a real texture.
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
            mat.set_texture(&self.device, name, view, sampler, fallback);
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

    /// Create a `width`x`height` offscreen render target. Draws into it via the
    /// frame API; present it to the swapchain with a Nearest upscale.
    pub fn create_target(&mut self, width: u32, height: u32) -> TargetId {
        let target = SceneTarget::new(&self.device, width, height, self.config.format);
        let id = TargetId(self.targets.len() as u32);
        self.targets.push(target);
        id
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
