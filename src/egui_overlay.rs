//! Optional egui overlay (feature `egui`). Wires egui's context + winit input
//! bridge + wgpu paint renderer, and paints a caller-supplied UI onto the
//! swapchain over the presented scene. Game-agnostic: the caller passes a UI
//! closure; the overlay knows nothing about what it draws.

use std::sync::Arc;

use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use egui_winit::State;
use wgpu::{
    CommandEncoderDescriptor, Device, LoadOp, Operations, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TextureFormat, TextureView,
};
use winit::event::WindowEvent;
use winit::window::Window;

/// An egui overlay bound to one window/surface. Feed it window events, then
/// `paint` a UI closure after presenting the scene.
pub struct EguiOverlay {
    context: egui::Context,
    input: State,
    renderer: Renderer,
}

impl EguiOverlay {
    /// Build an overlay for `window`, painting to `surface_format`.
    pub fn new(device: &Device, window: &Window, surface_format: TextureFormat) -> Self {
        let context = egui::Context::default();
        let input = State::new(
            context.clone(),
            context.viewport_id(),
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        let renderer = Renderer::new(
            device,
            surface_format,
            RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: true,
                predictable_texture_filtering: false,
            },
        );
        Self {
            context,
            input,
            renderer,
        }
    }

    /// Feed a window event to egui. Returns true if egui consumed it.
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.input.on_window_event(window, event).consumed
    }

    /// Run `build` to construct the UI, then paint it onto `surface_view`
    /// (loading over existing content). Records its own command buffer and
    /// submits it — call after the scene's `present`.
    pub fn paint(
        &mut self,
        device: &Device,
        queue: &Queue,
        window: &Arc<Window>,
        surface_view: &TextureView,
        surface_size: [u32; 2],
        mut build: impl FnMut(&egui::Context),
    ) {
        let raw_input = self.input.take_egui_input(window);
        // egui 0.35's `run_ui` hands a top-level `Ui`; expose its context so the
        // caller builds windows/panels the same way they would elsewhere.
        let full_output = self.context.run_ui(raw_input, |ui| build(ui.ctx()));
        self.input
            .handle_platform_output(window, full_output.platform_output);

        let paint_jobs = self
            .context
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = ScreenDescriptor {
            size_in_pixels: surface_size,
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("vk2d.egui.encoder"),
        });
        self.renderer
            .update_buffers(device, queue, &mut encoder, &paint_jobs, &screen);
        {
            let pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("vk2d.egui.pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            self.renderer.render(&mut pass, &paint_jobs, &screen);
        }
        queue.submit(std::iter::once(encoder.finish()));

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
