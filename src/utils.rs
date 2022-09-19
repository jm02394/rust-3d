pub fn interpolate(i0: isize, d0: f32, i1: isize, d1: f32) -> Vec<f32> {
    if i0 == i1 {
        return Vec::from([d0]);
    }

    let mut out: Vec<f32> = Vec::new();

    let a = (d1 - d0) as f32 / (i1 - i0) as f32;
    let mut d = d0 as f32;
    for _ in i0 as u32..i1 as u32 {
        out.push(d);
        d += a;
    }
    
    out
}

pub fn cat(a: &[f32], b: &[f32]) -> Vec<f32> {
    [a, b].concat()
}

pub fn min(n1: f32, n2: f32) -> f32 {
    if n1 < n2 { n1 }
    else { n2 }
}

pub fn max(n1: f32, n2: f32) -> f32 {
    if n1 < n2 { n2 }
    else { n1 }
}