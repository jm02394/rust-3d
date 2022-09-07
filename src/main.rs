use macroquad::prelude::*;

#[derive(PartialEq)]
struct Vec2 {
    x: f32,
    y: f32,
}

#[derive(PartialEq)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn project(&self, c: Vec3, r: Vec3, e: Vec3) -> Vec2  {
        let xp = self.x - c.x;
        let yp = -(self.y - c.y);
        let zp = self.z - c.z;

        let (dx, dy, dz) = match r {
            Vec3 { x: 0.0, y: 0.0, z: 0.0 } => (xp, yp, zp),
            _ => (
                r.y.cos() * (r.z.sin() * yp + r.z.cos() * xp) - r.y.sin() * zp,
                r.x.sin() * (r.y.cos() * zp + r.y.sin() * (r.z.sin() * yp + r.z.cos()  * xp)) + r.x.cos() * (r.z.cos() * yp - r.z.sin() * xp),
                r.x.cos() * (r.y.cos() * zp + r.y.sin() * (r.z.sin() * yp + r.z.cos()  * xp)) - r.x.sin() * (r.z.cos() * yp - r.z.sin() * xp),
            )
        };
        
        let bx = e.z / dz * dx + e.x;
        let by = e.z / dz * dy + e.y;
        
        Vec2 { x: (screen_width() + bx).floor(), y: (screen_height() + by).floor() } // Point { x: (width / 2 + bx).floor(), y: (height / 2 + by).floor() }
    }
}

#[macroquad::main("BasicShapes")]
async fn main() {
    loop {
        clear_background(WHITE);

        draw_line(40.0, 40.0, 100.0, 200.0, 15.0, BLUE);

        next_frame().await
    }
}