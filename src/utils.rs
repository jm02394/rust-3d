pub fn interpolate(n1: f32, n2: f32, steps: f32) -> Vec<f32> {
    let step_size = (n2 - n1) / steps;
    let mut out: Vec<f32> = Vec::new();

    for i in 0..steps as isize {
        out.push(step_size * i as f32);
    }

    out
}

pub fn min(n1: f32, n2: f32) -> f32 {
    if n1 < n2 { n1 }
    else { n2 }
}

pub fn max(n1: f32, n2: f32) -> f32 {
    if n1 < n2 { n2 }
    else { n1 }
}