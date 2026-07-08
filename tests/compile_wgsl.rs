//! Headless tests for the WGSL -> SPIR-V compile path (no GPU needed).

// A minimal valid fullscreen-triangle shader compiles; garbage does not.
const GOOD: &str = r#"
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
"#;

#[test]
fn good_wgsl_compiles_to_spirv() {
    let spv = vk2d::testing::compile_wgsl_to_spirv(GOOD).expect("valid WGSL compiles");
    assert!(!spv.is_empty());
    // SPIR-V magic number is the first word of the binary.
    assert_eq!(spv[0], 0x0723_0203);
}

#[test]
fn bad_wgsl_returns_compile_error() {
    let err = vk2d::testing::compile_wgsl_to_spirv("this is not wgsl").unwrap_err();
    assert!(matches!(err, vk2d::Vk2dError::ShaderCompile { .. }));
}
