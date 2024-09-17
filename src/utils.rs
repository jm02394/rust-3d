use std::f32::consts::PI;

pub type RGBA = [u8; 4];

pub const BLACK: RGBA = [0, 0, 0, 255];
pub const WHITE: RGBA = [255, 255, 255, 255];
pub const RED: RGBA = [255, 0, 0, 255];
pub const GREEN: RGBA = [0, 255, 0, 255];
pub const BLUE: RGBA = [0, 0, 255, 255];
pub const PURPLE: RGBA = [255, 0, 255, 255];
pub const YELLOW: RGBA = [255, 255, 0, 255];
pub const TEAL: RGBA = [0, 255, 255, 255];

pub fn mult_rgba(inp: RGBA, f: f32) -> RGBA {
    [
        (inp[0] as f32 * f) as u8,
        (inp[1] as f32 * f) as u8,
        (inp[2] as f32 * f) as u8,
        inp[3],
    ]
}

pub fn radians(inp: f32) -> f32 {
    PI / 180. * inp
}

pub fn interpolate(i0: f32, d0: f32, i1: f32, d1: f32) -> Vec<f32> {
    if i0 == i1 {
        return Vec::from([d0]);
    }

    let mut out: Vec<f32> = Vec::new();

    let a = (d1 - d0) as f32 / (i1 - i0) as f32;
    let mut d = d0 as f32;
    for _ in i0 as i32..=i1 as i32 {
        out.push(d);
        d += a;
    }

    out
}

pub fn interpolate_i(i0: i32, d0: f32, i1: i32, d1: f32) -> Vec<f32> {
    interpolate(i0 as f32, d0, i1 as f32, d1)
}

pub fn cat(a: &[f32], b: &[f32]) -> Vec<f32> {
    [a, b].concat()
}

pub fn min(n1: f32, n2: f32) -> f32 {
    if n1 < n2 {
        n1
    } else {
        n2
    }
}

pub fn max(n1: f32, n2: f32) -> f32 {
    if n1 < n2 {
        n2
    } else {
        n1
    }
}
