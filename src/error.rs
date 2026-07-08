//! Public error type. Setup and load functions return `Result<_, Vk2dError>`;
//! per-frame draw calls are infallible (bad handles are skipped).

/// Errors that setup and resource-loading operations can return.
#[derive(Debug)]
#[non_exhaustive]
pub enum Vk2dError {
    /// No GPU adapter matched the requested backend/surface.
    AdapterUnavailable(String),
    /// The window surface could not be created.
    SurfaceCreation(String),
    /// The GPU device/queue request failed.
    DeviceRequest(String),
    /// A WGSL shader failed to parse, validate, or emit SPIR-V. `message`
    /// carries the formatted compiler diagnostic (with source location where
    /// naga provides one).
    ShaderCompile {
        /// The formatted compiler diagnostic (with source location where naga
        /// provides one).
        message: String,
    },
    /// The surface was lost and could not be reconfigured.
    SurfaceLost,
    /// [`crate::Context::begin_target_frame`] was called with a [`crate::TargetId`]
    /// that has no corresponding offscreen target (never created via
    /// [`crate::Context::create_target`]).
    UnknownTarget,
}

impl std::fmt::Display for Vk2dError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AdapterUnavailable(m) => write!(f, "no adapter: {m}"),
            Self::SurfaceCreation(m) => write!(f, "surface creation failed: {m}"),
            Self::DeviceRequest(m) => write!(f, "device request failed: {m}"),
            Self::ShaderCompile { message } => write!(f, "shader compile error: {message}"),
            Self::SurfaceLost => write!(f, "surface lost"),
            Self::UnknownTarget => write!(f, "unknown render target id"),
        }
    }
}

impl std::error::Error for Vk2dError {}
