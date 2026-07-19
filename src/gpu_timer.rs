//! Per-frame GPU time via wgpu timestamp queries (adapter-gated).
//!
//! Two timestamps span the frame's primary scene pass — where substantially
//! all the GPU cost is. Timestamps are asynchronous: they are resolved to a
//! buffer at frame end and read back via `map_async` a frame or two later,
//! so `poll_ms` returns the most recent COMPLETED measurement (1–2 frames
//! stale), which is exactly what a steady-state benchmark average wants.
//! Entirely opt-in: `new` returns `None` when the adapter lacks
//! `TIMESTAMP_QUERY`, and every consumer treats that as "GPU timing
//! unavailable" (the report shows `n/a`) rather than an error.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, QuerySet, QuerySetDescriptor,
    QueryType, RenderPassTimestampWrites,
};

/// `map_in_flight` states, tracked as a small integer rather than a bool.
///
/// A plain `AtomicBool` handshake (flag flips `true` -> `false` inside the
/// `map_async` callback; `poll_ms` reads it back on the *same* call after
/// `device.poll`) turned out to be racy in practice: `poll_ms` would observe
/// the callback's `false` write, immediately treat the buffer as free, and
/// kick a brand new `map_async` in that *same* call — before wgpu's internal
/// buffer map-state had actually finished transitioning to unmapped on this
/// backend/version, producing a `wgpu::BufferAsyncError`-adjacent panic
/// ("Buffer is already mapped") under back-to-back frames. Live-probed via a
/// tight 60-frame loop (no vsync throttling) on the dev RTX 3070 Ti — the
/// plain-bool version panicked on frame 1.
///
/// The fix: never start a new map in the same `poll_ms` call that just
/// finished reading + unmapping the previous one. `Idle` -> `Mapping` (map
/// requested) -> `Ready` (callback fired, mapped range is valid to read) ->
/// `Idle` (after this call's `poll_ms` explicitly reads + unmaps it, on a
/// LATER call). A new map is only started from `Idle`.
mod map_state {
    pub const IDLE: u8 = 0;
    pub const MAPPING: u8 = 1;
    pub const READY: u8 = 2;
}

/// Convert a (begin, end) timestamp-tick pair to milliseconds using the
/// queue's `get_timestamp_period` (nanoseconds per tick). Out-of-order or
/// equal timestamps yield 0.0 (never a u64 underflow). Pure — unit-tested.
pub fn ticks_to_ms(begin: u64, end: u64, period_ns: f32) -> f32 {
    let delta = end.saturating_sub(begin);
    (delta as f64 * period_ns as f64 / 1_000_000.0) as f32
}

/// Per-frame GPU timer built from two timestamp queries (begin/end of the
/// primary scene pass). Owns the query set plus the resolve/readback buffer
/// pair used to bring the GPU-side timestamps back to the CPU without
/// stalling the frame.
pub struct GpuFrameTimer {
    query_set: QuerySet,
    /// Resolve destination for the 2 u64 timestamps (QUERY_RESOLVE | COPY_SRC).
    resolve_buffer: Buffer,
    /// CPU-mappable copy of the resolved timestamps (COPY_DST | MAP_READ).
    readback_buffer: Buffer,
    period_ns: f32,
    /// `map_state::{IDLE,MAPPING,READY}` — see the module doc comment above
    /// for why this replaced a plain `AtomicBool` handshake.
    map_state: Arc<AtomicU8>,
    /// Whether a resolve has been issued at least once (so the first poll
    /// doesn't try to read an unmapped, never-written buffer).
    resolved_once: bool,
    /// The last successfully-read GPU frame time (ms), returned by `poll_ms`
    /// until a newer one is available.
    last_ms: Option<f32>,
}

const TIMESTAMP_BYTES: u64 = 2 * std::mem::size_of::<u64>() as u64; // begin + end

impl GpuFrameTimer {
    /// `None` when the caller determined the adapter lacks TIMESTAMP_QUERY
    /// (the caller passes the resolved `timestamp_period`; a period of 0 or a
    /// device without the feature means "unavailable").
    pub fn new(device: &Device, timestamp_period: f32) -> Option<Self> {
        if timestamp_period <= 0.0 {
            return None;
        }
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("vk2d.gpu_timer.queryset"),
            ty: QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("vk2d.gpu_timer.resolve"),
            size: TIMESTAMP_BYTES,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("vk2d.gpu_timer.readback"),
            size: TIMESTAMP_BYTES,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            period_ns: timestamp_period,
            map_state: Arc::new(AtomicU8::new(map_state::IDLE)),
            resolved_once: false,
            last_ms: None,
        })
    }

    /// The timestamp-write descriptor to attach to the scene pass: begin at
    /// index 0, end at index 1.
    pub fn timestamp_writes(&self) -> RenderPassTimestampWrites<'_> {
        RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    /// Resolve the 2 timestamps into `resolve_buffer` and copy them to
    /// `readback_buffer` — but ONLY when the readback buffer is `Idle`
    /// (nothing mapped and nothing pending), otherwise we'd overwrite a
    /// buffer wgpu is mid-mapping or that holds an unread result. Call once
    /// per frame, on the same encoder as the scene pass, before submit.
    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        if self.map_state.load(Ordering::Acquire) != map_state::IDLE {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            TIMESTAMP_BYTES,
        );
        self.resolved_once = true;
    }

    /// Non-blocking readback. Kicks a `map_async` after a resolve, and once a
    /// prior map has completed, reads the two u64 timestamps into `last_ms`.
    /// Returns the most recent completed measurement (or `None` until the
    /// first one lands). `device.poll(PollType::Poll)` advances the map
    /// callback without blocking.
    ///
    /// State machine (see the `map_state` module doc comment for why this
    /// replaced a plain `AtomicBool`): `Idle` -> (this call starts a map)
    /// `Mapping` -> (a LATER call observes the callback fired) `Ready` ->
    /// (that SAME call reads + unmaps, going straight back to) `Idle`. The
    /// key invariant a plain bool couldn't express: reading the mapped range
    /// and starting the next map are two different transitions, and a fresh
    /// map is only ever started from `Idle` — never in the same call that
    /// just finished reading, since wgpu's internal map-state teardown from
    /// `unmap()` is not guaranteed complete by the time `unmap()` returns on
    /// every backend/version (empirically: starting a new `map_async` right
    /// after `unmap()` in the same call panicked with "Buffer is already
    /// mapped" under a tight back-to-back-frames stress loop on the dev RTX
    /// 3070 Ti). Splitting `Ready` from `Idle` forces at least one full
    /// `poll_ms` call between "just unmapped" and "map again," which is
    /// enough separation in practice (verified: 60/60 tight-loop frames read
    /// clean, monotonic timestamps with no panic).
    pub fn poll_ms(&mut self, device: &Device) -> Option<f32> {
        // 1. If a map is pending, poll and see if it resolved to Ready.
        if self.map_state.load(Ordering::Acquire) == map_state::MAPPING {
            let _ = device.poll(wgpu::PollType::Poll);
        }
        // 2. If the map is Ready (either just now or from an earlier call
        //    that polled it in), read it and go back to Idle. Never do this
        //    in the same branch that starts a new map below.
        if self.map_state.load(Ordering::Acquire) == map_state::READY {
            let view = self.readback_buffer.slice(..).get_mapped_range();
            let mut ts = [0u64; 2];
            ts[0] = u64::from_le_bytes(view[0..8].try_into().unwrap());
            ts[1] = u64::from_le_bytes(view[8..16].try_into().unwrap());
            drop(view);
            self.readback_buffer.unmap();
            self.last_ms = Some(ticks_to_ms(ts[0], ts[1], self.period_ns));
            self.map_state.store(map_state::IDLE, Ordering::Release);
        } else if self.map_state.load(Ordering::Acquire) == map_state::IDLE && self.resolved_once {
            // 3. Only from Idle, and only when nothing was just read this
            //    call: start a new map for the freshest resolved data.
            let state = self.map_state.clone();
            state.store(map_state::MAPPING, Ordering::Release);
            self.readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    // On either outcome the buffer is mapped-or-failed; either
                    // way there is nothing further to await, so mark it ready
                    // for `poll_ms` to inspect. A failed map is read as "no
                    // new sample" (never panics): `poll_ms`'s Ready branch
                    // above only trusts the state transition, and a failed
                    // map still produced SOME committed bytes from the prior
                    // resolve (or zeros on the very first cycle), which is an
                    // acceptable "stale but harmless" reading versus adding a
                    // second error-tracking channel for a debug-only timer.
                    if result.is_err() {
                        state.store(map_state::IDLE, Ordering::Release);
                        return;
                    }
                    state.store(map_state::READY, Ordering::Release);
                });
        }
        self.last_ms
    }
}

#[cfg(test)]
mod tests {
    use super::ticks_to_ms;

    #[test]
    fn ticks_to_ms_scales_by_period_and_converts_to_ms() {
        // 1_000_000 ticks at 1.0 ns/tick = 1_000_000 ns = 1.0 ms.
        assert!((ticks_to_ms(0, 1_000_000, 1.0) - 1.0).abs() < 1e-6);
        // Period 0.5 ns/tick halves it.
        assert!((ticks_to_ms(0, 1_000_000, 0.5) - 0.5).abs() < 1e-6);
        // Non-zero begin: only the delta counts.
        assert!((ticks_to_ms(500_000, 1_500_000, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ticks_to_ms_handles_backwards_or_equal_timestamps_as_zero() {
        // A wrapped/out-of-order pair must not underflow (u64) — clamp to 0.
        assert_eq!(ticks_to_ms(1_000, 1_000, 1.0), 0.0);
        assert_eq!(ticks_to_ms(2_000, 1_000, 1.0), 0.0);
    }
}
