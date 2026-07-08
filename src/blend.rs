//! Blend modes for materials and draws. Game-agnostic: which effect uses which
//! mode is the application's decision, so no shader-id defaults live here.

use wgpu::{BlendComponent, BlendFactor, BlendOperation, BlendState};

/// How a draw composites onto the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Standard source-over-destination alpha compositing:
    /// `Src·SrcA + Dst·(1−SrcA)`.
    #[default]
    Alpha,
    /// Alpha-weighted additive: `Src·SrcA + Dst·1`. Light accumulates without
    /// darkening what is beneath it — the glow mode.
    Additive,
    /// No blending; the source replaces the destination.
    Opaque,
}

impl BlendMode {
    /// The `wgpu::BlendState` for this mode, or `None` for `Opaque` (which maps
    /// to a `None` blend on the color target).
    // Consumed by the material/sprite pipelines (later tasks); until then only
    // the unit tests exercise it.
    #[allow(dead_code)]
    pub(crate) fn blend_state(self) -> Option<BlendState> {
        match self {
            BlendMode::Opaque => None,
            BlendMode::Alpha => Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
            }),
            BlendMode::Additive => Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_has_no_blend_and_others_do() {
        assert!(BlendMode::Opaque.blend_state().is_none());
        assert!(BlendMode::Alpha.blend_state().is_some());
        assert!(BlendMode::Additive.blend_state().is_some());
    }

    #[test]
    fn blend_states_use_expected_factors() {
        let alpha = BlendMode::Alpha.blend_state().unwrap();
        assert_eq!(alpha.color.src_factor, BlendFactor::SrcAlpha);
        assert_eq!(alpha.color.dst_factor, BlendFactor::OneMinusSrcAlpha);

        let additive = BlendMode::Additive.blend_state().unwrap();
        assert_eq!(additive.color.src_factor, BlendFactor::SrcAlpha);
        assert_eq!(additive.color.dst_factor, BlendFactor::One);
    }
}
