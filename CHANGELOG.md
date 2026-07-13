# Changelog

All notable changes to vk2d are documented in this file.

## [v0.1.1] - 2026-07-13

Release type: PATCH

- Reduced steady-state CPU and allocator pressure in draw-heavy scenes: frame
  commands, sprite instances, sprite upload bytes, shape upload bytes, and
  staged draw ranges now retain their high-water capacity between frames.
- Removed per-update uniform heap allocations and moved circle trigonometry to
  a one-time shared lookup table, making repeated VFX primitives cheaper.
- Made repeated material texture/target bindings idempotent, preventing normal
  immediate-mode usage from rebuilding identical wgpu bind groups every frame.
