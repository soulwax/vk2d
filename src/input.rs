//! Optional winit input collector (feature `winit-input`). Turns winit window
//! events into a simple per-frame `InputState` snapshot (keys held/pressed,
//! mouse position/buttons) so an app can poll input in the macroquad style
//! without hand-tracking every event. A convenience, not part of the renderer.

use std::collections::HashSet;

use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{Key, NamedKey};

/// A per-frame input snapshot. Call [`InputState::feed`] for each window event,
/// then query it during the frame, then [`InputState::end_frame`] to clear the
/// edge-triggered "pressed this frame" sets.
#[derive(Default)]
pub struct InputState {
    keys_down: HashSet<String>,
    keys_pressed: HashSet<String>,
    mouse: (f32, f32),
    buttons_down: HashSet<u8>,
    buttons_pressed: HashSet<u8>,
}

fn key_name(key: &Key) -> Option<String> {
    match key {
        Key::Named(n) => Some(format!("{n:?}")),
        Key::Character(s) => Some(s.to_string()),
        _ => None,
    }
}

fn button_id(b: MouseButton) -> u8 {
    match b {
        MouseButton::Left => 0,
        MouseButton::Right => 1,
        MouseButton::Middle => 2,
        _ => 3,
    }
}

impl InputState {
    /// A fresh input state with nothing pressed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a winit window event.
    pub fn feed(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(name) = key_name(&event.logical_key) {
                    match event.state {
                        ElementState::Pressed => {
                            if !self.keys_down.contains(&name) {
                                self.keys_pressed.insert(name.clone());
                            }
                            self.keys_down.insert(name);
                        }
                        ElementState::Released => {
                            self.keys_down.remove(&name);
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse = (position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let id = button_id(*button);
                match state {
                    ElementState::Pressed => {
                        if !self.buttons_down.contains(&id) {
                            self.buttons_pressed.insert(id);
                        }
                        self.buttons_down.insert(id);
                    }
                    ElementState::Released => {
                        self.buttons_down.remove(&id);
                    }
                }
            }
            _ => {}
        }
    }

    /// Clear the edge-triggered sets. Call once at the end of each frame.
    pub fn end_frame(&mut self) {
        self.keys_pressed.clear();
        self.buttons_pressed.clear();
    }

    /// Whether a named key is currently held. Names match winit's `NamedKey`
    /// debug form (e.g. "Escape", "F1") or the character (e.g. "a").
    pub fn is_key_down(&self, name: &str) -> bool {
        self.keys_down.contains(name)
    }

    /// Whether a named key went down this frame (edge).
    pub fn is_key_pressed(&self, name: &str) -> bool {
        self.keys_pressed.contains(name)
    }

    /// Convenience: was a `NamedKey` pressed this frame?
    pub fn is_named_pressed(&self, key: NamedKey) -> bool {
        self.keys_pressed.contains(&format!("{key:?}"))
    }

    /// Cursor position in physical pixels.
    pub fn mouse_position(&self) -> (f32, f32) {
        self.mouse
    }

    /// Whether mouse button `id` (0=left, 1=right, 2=middle) is held.
    pub fn is_mouse_down(&self, id: u8) -> bool {
        self.buttons_down.contains(&id)
    }

    /// Whether mouse button `id` went down this frame (edge).
    pub fn is_mouse_pressed(&self, id: u8) -> bool {
        self.buttons_pressed.contains(&id)
    }
}
