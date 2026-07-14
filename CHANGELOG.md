# Changelog

All notable changes to vk2d are documented in this file.

## [Unreleased]

Release type: PATCH

- Added a doc-hidden `Context::read_target_pixel` GPU read-back helper and gave
  scene targets `COPY_SRC` usage, so integration tests can assert what a
  finished render target actually contains. No change to the drawing path.
- Added a live-GPU regression test (`tests/target_view_readback.rs`, `#[ignore]`d
  from the default gate) and a diagnostic example (`examples/target_view_readback.rs`)
  that pin `Frame::target_sprite`'s read-back as view-agnostic: a target drawn
  under a non-identity (Y-up) `View2` composites back faithfully, including the
  supersampled downscale path. Verified `target_sprite` does not cause the
  black-readback a consumer reported (root cause was an empty source target on
  the consumer side, not vk2d).

## [v0.1.1] - 2026-07-13

Release type: PATCH

- Reduced steady-state CPU and allocator pressure in draw-heavy scenes: frame
  commands, sprite instances, sprite upload bytes, shape upload bytes, and
  staged draw ranges now retain their high-water capacity between frames.
- Removed per-update uniform heap allocations and moved circle trigonometry to
  a one-time shared lookup table, making repeated VFX primitives cheaper.
- Made repeated material texture/target bindings idempotent, preventing normal
  immediate-mode usage from rebuilding identical wgpu bind groups every frame.
