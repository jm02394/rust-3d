#![deny(clippy::all)]
#![forbid(unsafe_code)]
#![allow(dead_code)]

use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use core::f32::{INFINITY, NEG_INFINITY};

use std::mem;
use std::time::Instant;

use rand::Rng;

use bresenham::Bresenham;

mod utils;
use utils::*;

const WIDTH: u32 = 300;
const HEIGHT: u32 = 300;
const ASPECT_RATIO: u32 = HEIGHT / WIDTH;

const MOVE_SCALE: f32 = 0.05;
const ROT_SCALE: f32 = 0.1;

const AXIS_LEN: i32 = 10;

const ORIGIN: Vec3 = Vec3 {
    x: 0.,
    y: 0.,
    z: 0.,
};

fn get_cube() -> Prim {
    Prim {
        tris: vec![
            //Bottom
            Tri::new(
                Vec3::new_i(1, 0, 0),
                Vec3::new_i(0, 0, 0),
                Vec3::new_i(0, 0, 1),
                PURPLE,
            ),
            Tri::new(
                Vec3::new_i(1, 0, 0),
                Vec3::new_i(1, 0, 1),
                Vec3::new_i(0, 0, 1),
                PURPLE,
            ),
            //Front
            Tri::new(
                Vec3::new_i(0, 0, 0),
                Vec3::new_i(0, 1, 0),
                Vec3::new_i(1, 1, 0),
                RED,
            ),
            Tri::new(
                Vec3::new_i(0, 0, 0),
                Vec3::new_i(1, 0, 0),
                Vec3::new_i(1, 1, 0),
                RED,
            ),
            //Left
            Tri::new(
                Vec3::new_i(0, 1, 0),
                Vec3::new_i(0, 0, 0),
                Vec3::new_i(0, 0, 1),
                GREEN,
            ),
            Tri::new(
                Vec3::new_i(0, 1, 0),
                Vec3::new_i(0, 1, 1),
                Vec3::new_i(0, 0, 1),
                GREEN,
            ),
            //Back
            Tri::new(
                Vec3::new_i(1, 0, 1),
                Vec3::new_i(0, 0, 1),
                Vec3::new_i(0, 1, 1),
                TEAL,
            ),
            Tri::new(
                Vec3::new_i(1, 0, 1),
                Vec3::new_i(1, 1, 1),
                Vec3::new_i(0, 1, 1),
                TEAL,
            ),
            //Right
            Tri::new(
                Vec3::new_i(1, 0, 0),
                Vec3::new_i(1, 0, 1),
                Vec3::new_i(1, 1, 1),
                BLUE,
            ),
            Tri::new(
                Vec3::new_i(1, 0, 0),
                Vec3::new_i(1, 1, 0),
                Vec3::new_i(1, 1, 1),
                BLUE,
            ),
            //Top
            Tri::new(
                Vec3::new_i(1, 1, 0),
                Vec3::new_i(0, 1, 0),
                Vec3::new_i(0, 1, 1),
                YELLOW,
            ),
            Tri::new(
                Vec3::new_i(1, 1, 0),
                Vec3::new_i(1, 1, 1),
                Vec3::new_i(0, 1, 1),
                YELLOW,
            ),
        ],
    }
}

#[derive(Clone, Copy)]
struct CZ {
    c: RGBA,
    z: f32,
}

impl CZ {
    fn new(c: RGBA, z: f32) -> Self {
        Self { c, z }
    }
}

struct ZBuffer {
    b: Box<[Box<[CZ]>]>,
}

impl ZBuffer {
    fn new() -> Self {
        Self {
            b: vec![
                vec![
                    CZ {
                        c: [255, 255, 255, 255],
                        z: INFINITY
                    };
                    HEIGHT as usize
                ]
                .into_boxed_slice();
                WIDTH as usize
            ]
            .into_boxed_slice(),
        }
    }

    fn force_set(&mut self, x: i32, y: i32, cz: CZ) {
        if !(0 > x || x >= WIDTH as i32 || 0 > y || y >= HEIGHT as i32) {
            self.b[x as usize][y as usize] = cz;
        }
    }

    fn try_set(&mut self, x: i32, y: i32, cz: CZ) {
        if let Some(z_buf_result) = self.get(x, y) {
            if cz.z < z_buf_result.z {
                self.force_set(x, y, cz);
            }
        }
    }

    fn get(&self, x: i32, y: i32) -> Option<CZ> {
        if !(0 > x || x >= WIDTH as i32 || 0 > y || y >= HEIGHT as i32) {
            Some(self.b[x as usize][y as usize])
        } else {
            None
        }
    }
}

struct Canvas<'a> {
    cam: &'a Camera,
    zbuffer: ZBuffer,
}

impl<'a> Canvas<'a> {
    fn new(cam: &'a Camera, zbuffer: ZBuffer) -> Self {
        Self { cam, zbuffer }
    }

    /*fn draw_px(&mut self, x: i32, y: i32, col: RGBA) {
        if 0 > x || x >= WIDTH as i32 || 0 > y || y >= HEIGHT as i32 {
            return
        }
        let i = (x * 4 + y * WIDTH as i32 * 4) as usize;
        self.frame[i..i + 4].copy_from_slice(&col);
    }*/

    fn draw_line(&mut self, a: Vec2, b: Vec2, cz: CZ) {
        for (x, y) in Bresenham::new((a.x as isize, a.y as isize), (b.x as isize, b.y as isize)) {
            //self.draw_px(x, y, col);
            self.zbuffer.try_set(x as i32, y as i32, cz);
        }
    }

    fn render_tri(&mut self, t: Tri) {
        let ((a, mut az), (b, mut bz), (c, mut cz)) = (
            &mut self.cam.projected(&t.a),
            &mut self.cam.projected(&t.b),
            &mut self.cam.projected(&t.c),
        );

        if !(a.is_inbounds() || b.is_inbounds() || c.is_inbounds()) {
            return;
        }

        //let (mut az, mut bz, mut cz) = (t.a.z, t.b.z, t.c.z);

        if b.y < a.y {
            mem::swap(b, a);
            mem::swap(&mut bz, &mut az);
        };
        if c.y < a.y {
            mem::swap(c, a);
            mem::swap(&mut cz, &mut az);
        };
        if c.y < b.y {
            mem::swap(c, b);
            mem::swap(&mut cz, &mut bz);
        };

        let mut xab = interpolate_i(a.y, a.x as f32, b.y, b.x as f32);
        let mut zab = interpolate_i(a.y, az, b.y, bz);

        let xbc = interpolate_i(b.y, b.x as f32, c.y, c.x as f32);
        let zbc = interpolate_i(b.y, bz, c.y, cz);

        let xac = interpolate_i(a.y, a.x as f32, c.y, c.x as f32);
        let zac = interpolate_i(a.y, az, c.y, cz);

        xab.pop();
        let xabc = xab.into_iter().chain(xbc.into_iter()).collect::<Vec<_>>();

        zab.pop();
        let zabc = zab.into_iter().chain(zbc.into_iter()).collect::<Vec<_>>();

        let (xl, xr): (Vec<f32>, Vec<f32>);
        let (zl, zr): (Vec<f32>, Vec<f32>);
        let m = xabc.len() / 2;

        if xac.len() == 0 || xabc.len() == 0 {
            return;
        }

        if xac[m] < xabc[m] {
            xl = xac;
            xr = xabc;
            zl = zac;
            zr = zabc;
        } else {
            xl = xabc;
            xr = xac;
            zl = zabc;
            zr = zac;
        }

        for y in a.y..=c.y {
            let sub = (y - a.y) as usize;

            if sub >= zl.len() || sub >= zr.len() {
                continue;
            }

            let xlp = xl[sub];
            let xrp = xr[sub];

            //println!("{}, {}, {}, {}", xlp, zl[sub], xrp, zr[sub]);
            let zint = interpolate(xlp, zl[sub], xrp, zr[sub]);
            //println!("{:?}", zint);
            //std::process::exit(0);

            //let zint = |x: i32| ((zr[sub] - zl[sub]) / (xrp - xlp)) * x as f32 + zl[sub];

            /*if xrp < xlp {
                std::process::exit(0);
            }*/

            for x in xlp as i32..xrp as i32 {
                /*if !(0 < x && x < WIDTH as i32) {
                    continue;
                }*/

                if Vec2::new(x, y).is_inbounds() {
                    self.zbuffer
                        .try_set(x, y, CZ::new(t.col, zint[x as usize - xlp as usize]));
                }
            }
        }

        /*self.zbuffer.set(a.x, a.y, CZ::new(BLACK, NEG_INFINITY));
        self.zbuffer.set(b.x, b.y, CZ::new(BLACK, NEG_INFINITY));
        self.zbuffer.set(c.x, c.y, CZ::new(BLACK, NEG_INFINITY));*/
    }

    fn render_prim(&mut self, i: &Prim) {
        for t in i.tris.iter() {
            self.render_tri(*t);
        }
    }

    fn render_line(&mut self, i: &Line) {
        let ((sp, sz), (ep, ez)) = (self.cam.projected(&i.start), self.cam.projected(&i.end));
        /*if !(sp.is_inbounds() && ep.is_inbounds()) {
            return;
        }*/

        let points = Bresenham::new(
            (sp.x as isize, sp.y as isize),
            (ep.x as isize, ep.y as isize),
        )
        .into_iter()
        .collect::<Vec<(isize, isize)>>();
        let interp = interpolate_i(0, sz, points.len() as i32, ez);

        for (e, (x, y)) in points.into_iter().enumerate() {
            self.zbuffer
                .try_set(x as i32, y as i32, CZ::new(i.col, interp[e]))
        }
    }

    fn render_point(&mut self, i: &Vec3) {
        let (p, z) = self.cam.projected(i);

        for dx in -1..=1 {
            for dy in -1..=1 {
                self.zbuffer.try_set(p.x + dx, p.y + dy, CZ::new(BLACK, z));
            }
        }
    }

    /*fn render_prim_cols(&mut self, i: &Prim, cols: Vec<RGBA>) {
        for (ti, t) in i.tris.iter().enumerate() {
            let (ap, bp, cp) = (
                &mut t.a.project(&self.cam),
                &mut t.b.project(&self.cam),
                &mut t.c.project(&self.cam),
            );
            self.render_tri(
                &mut Vec3::new(ap.x as f32, ap.y as f32, t.a.z),
                &mut Vec3::new(bp.x as f32, bp.y as f32, t.b.z),
                &mut Vec3::new(cp.x as f32, cp.y as f32, t.c.z),
                cols[ti],
            )
        }
    }*/
}

struct Camera {
    pos: Vec3,
    rot: Vec3,
    proj: Vec3,
    sc: f32, //scale
    focal_length: f32,
    view_angle: f32,
}

impl Camera {
    fn new(pos: Vec3, rot: Vec3, proj: Vec3, sc: f32) -> Self {
        Self {
            pos,
            rot,
            proj,
            sc,
            focal_length: 1.,
            view_angle: radians(20.),
        }
    }

    fn translate_mut(&mut self, x: f32, y: f32, z: f32) {
        self.pos = self.pos.translated(x, y, z);
    }

    fn projected(&self, vec: &Vec3) -> (Vec2, f32) {
        // (projected Vec2, z-depth)
        let c = &self.pos;
        let r = &self.rot;

        let view_vec = vec
            .translated_vec(&self.pos.inv())
            .rotated_around(&ORIGIN, &self.rot);

        /*let line_x = |z: f32| vec.x / vec.z * z;
        let line_y = |z: f32| vec.y / vec.z * z;*/

        let out = Vec2::new(
            WIDTH as i32 / 2 + (view_vec.x / view_vec.z * self.focal_length * 100.).round() as i32,
            HEIGHT as i32 / 2 - (view_vec.y / view_vec.z * self.focal_length * 100.).round() as i32,
        );

        (out, c.distance(vec))
    }

    fn projected_old(&self, vec: &Vec3) -> (Vec2, f32) {
        // (projected Vec2, z-depth)
        let c = &ORIGIN; //let c = &self.pos;
        let r = &ORIGIN; //let r = &self.rot;
        let e = &self.proj;

        let xp = vec.x - c.x;
        let yp = -(vec.y - c.y);
        let zp = vec.z - c.z;

        let (dx, dy, dz) = if r == &Vec3::new_i(0, 0, 0) {
            (xp, yp, zp)
        } else {
            (
                r.y.cos() * (r.z.sin() * yp + r.z.cos() * xp) - r.y.sin() * zp,
                r.x.sin() * (r.y.cos() * zp + r.y.sin() * (r.z.sin() * yp + r.z.cos() * xp))
                    + r.x.cos() * (r.z.cos() * yp - r.z.sin() * xp),
                r.x.cos() * (r.y.cos() * zp + r.y.sin() * (r.z.sin() * yp + r.z.cos() * xp))
                    - r.x.sin() * (r.z.cos() * yp - r.z.sin() * xp),
            )
        };

        // reverse translation, rotation, and yeah (not working)
        //let r_pos = vec.translated_vec(&self.pos.inv());
        //let r_rot = r_pos.rotated_around(&ORIGIN, &self.rot.inv());

        let sx = WIDTH as f32;
        let sy = HEIGHT as f32;

        //let dz = c.distance(vec);
        //let dz = zp.abs();

        //let bx = (dx * sx) / dz * self.focal;
        //let by = (dy * sy) / dz * self.focal;

        let bx = e.z / dz * dx * self.focal_length;
        let by = e.z / dz * dy * self.focal_length;

        let out = Vec2::new(
            (WIDTH as f32 / 2. + bx * self.sc).floor() as i32,
            (HEIGHT as f32 / 2. + by * self.sc).floor() as i32,
        );

        (out, c.distance(vec))
    }
}

#[derive(PartialEq, Clone, Copy)]
struct Vec2 {
    x: i32,
    y: i32,
}

impl Vec2 {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn is_inbounds(&self) -> bool {
        0 <= self.x && self.x < WIDTH as i32 && 0 <= self.y && self.y < HEIGHT as i32
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new_i(x: i32, y: i32, z: i32) -> Self {
        Self {
            x: x as f32,
            y: y as f32,
            z: z as f32,
        }
    }

    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn distance(&self, other: &Vec3) -> f32 {
        ((other.x - self.x).powf(2.) + (other.y - self.y).powf(2.) + (other.z - self.z).powf(2.))
            .sqrt()
    }

    fn inv(&self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }

    fn diff(&self, other: &Vec3) -> Self {
        Self::new(other.x - self.x, other.y - self.y, other.z - self.z)
    }

    fn translated(&self, x: f32, y: f32, z: f32) -> Self {
        Self::new(self.x + x, self.y + y, self.z + z)
    }

    fn translated_vec(&self, other: &Vec3) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn rotated_single_axis(o1: f32, o2: f32, s1: f32, s2: f32, rot: f32) -> (f32, f32) {
        let d1 = s1 - o1;
        let d2 = s2 - o2;

        let d = (d1.powi(2) + d2.powi(2)).sqrt();

        let div = if d1 == 0. {
            //let div = if (1000. * dx).floor() / 1000. == 0. {
            if s1 > o1 {
                radians(90.)
            } else {
                radians(-90.)
            }
        } else {
            (d2 / d1).atan()
        };
        let r = div + rot + {
            if s1 > o1 {
                radians(180.)
            } else {
                0.
            }
        };

        (d * r.cos(), d * r.sin())
    }

    fn rotated_around(&self, origin: &Vec3, rot: &Vec3) -> Self {
        if rot == &Vec3::new_i(0, 0, 0) {
            return *self;
        }

        let (rx_y, rx_z) = Self::rotated_single_axis(origin.y, origin.z, self.y, self.z, rot.x);
        let (ry_x, ry_z) = Self::rotated_single_axis(origin.x, origin.z, self.x, self.z, rot.y);
        let (rz_x, rz_y) = Self::rotated_single_axis(origin.x, origin.y, self.x, self.y, rot.z);

        Self::new(
            -(self.x + ry_x + rz_x),
            -(self.y + rx_y + rz_y),
            -(self.z + rx_z + ry_z),
        )
    }

    fn rotated_around_old(&self, origin: &Vec3, rot: &Vec3) -> Self {
        //let d = self.distance(origin);

        let dx = self.x - origin.x;
        //let dy = self.y - origin.y;
        let dz = self.z - origin.z;

        let dxz = (dx.powi(2) + dz.powi(2)).sqrt();

        //println!("{}", dx);
        //std::process::exit(0);

        let div = if dx == 0. {
            //let div = if (1000. * dx).floor() / 1000. == 0. {
            if self.x > origin.x {
                radians(90.)
            } else {
                radians(-90.)
            }
        } else {
            (dz / dx).atan()
        };
        let ry = div + rot.y + {
            if self.x > origin.x {
                radians(180.)
            } else {
                0.
            }
        };

        //self.x += origin.x - d * ry.cos();
        //self.z += origin.z - d * ry.sin();

        let ox = origin.x - dxz * ry.cos();
        let oy = self.y;
        let oz = origin.z - dxz * ry.sin();

        /*println!(
            "({}, {}, {}) -> ({}, {}, {})",
            self.x,
            self.y,
            self.z,
            ox.round(),
            oy.round(),
            oz.round(),
        );*/

        Self::new(ox, oy, oz)
    }
}

#[derive(PartialEq, Clone, Copy)]
struct Tri {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    col: RGBA,
}

impl Tri {
    fn new(a: Vec3, b: Vec3, c: Vec3, col: RGBA) -> Self {
        Self { a, b, c, col }
    }

    fn translated(&self, x: f32, y: f32, z: f32) -> Self {
        Self {
            a: self.a.translated(x, y, z),
            b: self.b.translated(x, y, z),
            c: self.c.translated(x, y, z),
            col: self.col,
        }
    }

    fn translated_vec(&self, vec: &Vec3) -> Self {
        Self {
            a: self.a.translated_vec(vec),
            b: self.b.translated_vec(vec),
            c: self.c.translated_vec(vec),
            col: self.col,
        }
    }

    fn rotated_around(&self, origin: &Vec3, rot: &Vec3) -> Self {
        Self::new(
            self.a.rotated_around(origin, rot),
            self.b.rotated_around(origin, rot),
            self.c.rotated_around(origin, rot),
            self.col,
        )
    }
}

struct Prim {
    tris: Vec<Tri>,
}

impl Prim {
    fn new(tris: Vec<Tri>) -> Self {
        Self { tris }
    }

    fn translated(&self, x: f32, y: f32, z: f32) -> Self {
        Prim::new(self.tris.iter().map(|t| t.translated(x, y, z)).collect())
    }

    fn translated_vec(&self, vec: &Vec3) -> Self {
        Prim::new(self.tris.iter().map(|t| t.translated_vec(vec)).collect())
    }

    fn rotated_around(&self, origin: &Vec3, rot: &Vec3) -> Self {
        let mut out: Vec<Tri> = Vec::new();

        for t in &self.tris {
            out.push(t.rotated_around(origin, rot));
        }

        Self::new(out)
    }
}

struct World {
    c: f32,
    cols: [RGBA; 16],
    cam: Camera,
}

struct Line {
    start: Vec3,
    end: Vec3,
    col: RGBA,
}

impl Line {
    fn new(start: Vec3, end: Vec3, col: RGBA) -> Self {
        Self { start, end, col }
    }
}

impl World {
    fn new() -> Self {
        let mut colarray = [[0, 0, 0, 255]; 16];
        for i in 0..colarray.len() {
            colarray[i] = [
                rand::thread_rng().gen_range(0..=255),
                rand::thread_rng().gen_range(0..=255),
                rand::thread_rng().gen_range(0..=255),
                255,
            ];
        }

        Self {
            c: 0.,
            cols: colarray,
            cam: Camera::new(
                Vec3::new(0., 0., -10.),                          //Vec3::new(0., 1.5, -2.),
                Vec3::new(radians(0.), radians(0.), radians(0.)), //Vec3::new(radians(-30.), radians(30.), 0.),
                Vec3::new_i(0, 0, 200),
                1.,
            ),
        }
    }

    fn update(&mut self) {
        self.c += 1.;
        /*if self.c == 360. {
            self.c = 0.
        }*/

        //self.cam.translate_mut(0.003, 0.002, 0.); //y: 0.002
    }

    fn draw(&self, frame: &mut [u8]) {
        let mut r = Canvas::new(&self.cam, ZBuffer::new());

        /*let trans_fac = self.cam.pos.inv();
        let rot_fac = self.cam.rot;

        println!("{:?}, {:?}", trans_fac, rot_fac);*/

        let rot_fac = Vec3::new(
            radians(self.c),
            radians(self.c * 0.8),
            0., //radians(self.c * 0.6),
        );

        r.render_prim(&get_cube().rotated_around(&ORIGIN, &rot_fac));
        r.render_prim(
            &get_cube()
                .translated(1., 1., 0.)
                .rotated_around(&ORIGIN, &rot_fac),
        );
        r.render_prim(
            &get_cube()
                .translated(2., 2., 0.)
                .rotated_around(&ORIGIN, &rot_fac),
        );
        r.render_prim(
            &get_cube()
                .translated(3., 3., 0.)
                .rotated_around(&ORIGIN, &rot_fac),
        );
        r.render_prim(
            &get_cube()
                .translated(4., 4., 0.)
                .rotated_around(&ORIGIN, &rot_fac),
        );

        /*let (correct_origin, _oz) = self.cam.projected_old(&ORIGIN);
        r.zbuffer.force_set(
            correct_origin.x,
            correct_origin.y,
            CZ::new(BLACK, NEG_INFINITY),
        );*/

        /*r.render_line(&Line::new(
            Vec3::new_i(-AXIS_LEN, 0, 0),
            Vec3::new_i(AXIS_LEN, 0, 0),
            GREEN,
        ));
        r.render_line(&Line::new(
            Vec3::new_i(0, -AXIS_LEN, 0),
            Vec3::new_i(0, AXIS_LEN, 0),
            RED,
        ));
        r.render_line(&Line::new(
            Vec3::new_i(0, 0, -AXIS_LEN),
            Vec3::new_i(0, 0, AXIS_LEN),
            BLUE,
        ));*/
        /*r.render_tri(Tri::new(
            Vec3::new(-100., -10., 0.),
            Vec3::new(-100., -10., 100.),
            Vec3::new(100., -10., 100.),
            RED,
        ));*/
        //r.render_tri(&Tri::new(Vec3::new(-200., -250., 0.3), Vec3::new(200., 50., 0.1), Vec3::new(20., 250., 1.0)), [0, 255, 0, 255]);
        //r.render_tri(&Tri::new(Vec3::new(0., 0., 0.3), Vec3::new(1., 0., 0.1), Vec3::new(0., 1., 1.0)), [0, 255, 0, 255]);

        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            /*let rgba = [255, 255, 255, 255];

            pixel.copy_from_slice(&rgba);*/

            let x = i % WIDTH as usize;
            let y = i / WIDTH as usize;

            if let Some(z_buffer_result) = r.zbuffer.get(x as i32, y as i32) {
                pixel.copy_from_slice(&z_buffer_result.c);
            }

            //r.draw_px(x as i32, y as i32, r.zbuffer.get(x, y));
        }
    }
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("hi")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };
    window.set_maximized(true);

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };
    let mut world = World::new();

    let mut last_frame_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        // Draw the current frame
        if let Event::RedrawRequested(_) = event {
            println!("FPS: {}", 1000 / last_frame_time.elapsed().as_millis());
            last_frame_time = Instant::now();

            world.draw(pixels.get_frame());

            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        let mut move_mod = 1.;

        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }
            if input.key_held(VirtualKeyCode::LShift) {
                move_mod = 2.;
            }
            if input.key_held(VirtualKeyCode::Down) {
                world.cam.rot.x += radians(-10.) * ROT_SCALE;
            }
            if input.key_held(VirtualKeyCode::Up) {
                world.cam.rot.x += radians(10.) * ROT_SCALE;
            }
            if input.key_held(VirtualKeyCode::Left) {
                world.cam.rot.y += radians(-10.) * ROT_SCALE;
            }
            if input.key_held(VirtualKeyCode::Right) {
                world.cam.rot.y += radians(10.) * ROT_SCALE;
            }
            if input.key_held(VirtualKeyCode::W) {
                world.cam.pos.x +=
                    world.cam.rot.y.sin() * world.cam.rot.x.cos() * move_mod * MOVE_SCALE; //world.cam.pos.z += 0.5;
                world.cam.pos.y += world.cam.rot.x.sin() * move_mod * MOVE_SCALE;
                world.cam.pos.z +=
                    world.cam.rot.y.cos() * world.cam.rot.x.cos() * move_mod * MOVE_SCALE;
            }
            if input.key_held(VirtualKeyCode::S) {
                world.cam.pos.x -=
                    world.cam.rot.y.sin() * world.cam.rot.x.cos() * move_mod * MOVE_SCALE; //world.cam.pos.z += -0.5;
                world.cam.pos.y -= world.cam.rot.x.sin() * move_mod * MOVE_SCALE;
                world.cam.pos.z -=
                    world.cam.rot.y.cos() * world.cam.rot.x.cos() * move_mod * MOVE_SCALE;
            }
            if input.key_held(VirtualKeyCode::A) {
                world.cam.pos.x -= world.cam.rot.y.cos() * move_mod * MOVE_SCALE; //world.cam.pos.x += -0.5;
                world.cam.pos.z += world.cam.rot.y.sin() * move_mod * MOVE_SCALE;
            }
            if input.key_held(VirtualKeyCode::D) {
                world.cam.pos.x += world.cam.rot.y.cos() * move_mod * MOVE_SCALE; //world.cam.pos.x += 0.5;
                world.cam.pos.z -= world.cam.rot.y.sin() * move_mod * MOVE_SCALE;
            }
            if input.key_held(VirtualKeyCode::E) {
                world.cam.pos.y += MOVE_SCALE;
            }
            if input.key_held(VirtualKeyCode::Q) {
                world.cam.pos.y += -MOVE_SCALE;
            }
            if input.key_held(VirtualKeyCode::C) {
                world.cam.rot.z += radians(10.) * ROT_SCALE;
            }
            if input.key_held(VirtualKeyCode::Z) {
                world.cam.rot.z += radians(-10.) * ROT_SCALE;
            }
            if input.key_held(VirtualKeyCode::Minus) {
                world.cam.focal_length -= 0.005;
            }
            if input.key_held(VirtualKeyCode::Equals) {
                world.cam.focal_length += 0.005;
            }

            // Rotation clipping
            if world.cam.rot.x > radians(90.) {
                world.cam.rot.x = radians(90.);
            }
            if world.cam.rot.x < radians(-90.) {
                world.cam.rot.x = radians(-90.);
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                pixels.resize_surface(size.width, size.height);
            }

            // Update internal state and request a redraw
            world.update();
            window.request_redraw();
        }
    });
}
