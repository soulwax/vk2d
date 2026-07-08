//! A material that samples one declared texture compiles + renders headlessly.
//! Uses a hidden test window? No — vk2d needs a surface. Instead assert the
//! WGSL-with-texture-bindings compiles via the same naga path (render wiring is
//! covered by the gallery, Task A4). This locks the binding-layout contract.

const TEX_MATERIAL: &str = r#"
struct U { tint: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t0: texture_2d<f32>;
@group(0) @binding(2) var s0: sampler;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
}
@fragment fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureSample(t0, s0, p.xy * 0.0) * u.tint;
}
"#;

#[test]
fn texture_material_wgsl_compiles() {
    assert!(vk2d::testing::compile_wgsl_to_spirv(TEX_MATERIAL).is_ok());
}
