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

const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

const MOVE_SCALE: f32 = 0.05;
const ROT_SCALE: f32 = 0.1;

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

    fn set(&mut self, x: i32, y: i32, cz: CZ) {
        if !(0 > x || x >= WIDTH as i32 || 0 > y || y >= HEIGHT as i32) {
            self.b[x as usize][y as usize] = cz;
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

    fn draw_line(&mut self, a: Vec2, b: Vec2, col: RGBA) {
        for (x, y) in Bresenham::new((a.x as isize, a.y as isize), (b.x as isize, b.y as isize)) {
            //self.draw_px(x, y, col);
            self.zbuffer.set(x as i32, y as i32, CZ::new(col, 0.));
        }
    }

    fn render_tri(&mut self, t: Tri) {
        let ((a, mut az), (b, mut bz), (c, mut cz)) = (
            &mut t.a.project(&self.cam),
            &mut t.b.project(&self.cam),
            &mut t.c.project(&self.cam),
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
            //let zint = interpolate(xlp, zl[sub], xrp, zr[sub]);
            //println!("{:?}", zint);
            //std::process::exit(0);

            let zint = |x: i32| ((zr[sub] - zl[sub]) / (xrp - xlp)) * x as f32 + zl[sub];

            for x in 0..(xrp - xlp) as i32 {
                if let Some(z_buffer_result) = self.zbuffer.get(x, y) {
                    if z_buffer_result.z > zint(x) {
                        self.zbuffer.set(xlp as i32 + x, y, CZ::new(t.col, zint(x)));
                    }
                }
            }
        }

        self.zbuffer.set(a.x, a.y, CZ::new(BLACK, NEG_INFINITY));
        self.zbuffer.set(b.x, b.y, CZ::new(BLACK, NEG_INFINITY));
        self.zbuffer.set(c.x, c.y, CZ::new(BLACK, NEG_INFINITY));
    }

    fn render_prim(&mut self, i: &Prim) {
        for t in i.tris.iter() {
            self.render_tri(*t);
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
}

impl Camera {
    fn new(pos: Vec3, rot: Vec3, proj: Vec3, sc: f32) -> Self {
        Self { pos, rot, proj, sc }
    }

    fn translate_mut(&mut self, x: f32, y: f32, z: f32) {
        self.pos = self.pos.translate(x, y, z);
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
        0 < self.x && self.x < WIDTH as i32 && 0 < self.y && self.y < HEIGHT as i32
    }
}

#[derive(PartialEq, Clone, Copy)]
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

    fn distance(&self, inp: Vec3) -> f32 {
        ((inp.x - self.x).powf(2.) + (inp.y - self.y).powf(2.) + (inp.z - self.z).powf(2.)).sqrt()
    }

    fn translate(&self, x: f32, y: f32, z: f32) -> Self {
        Self {
            x: self.x + x,
            y: self.y + y,
            z: self.z + z,
        }
    }

    fn project(&self, cam: &Camera) -> (Vec2, f32) {
        // (projected Vec2, z-depth)
        let c = &cam.pos;
        let r = &cam.rot;
        let e = &cam.proj;

        let xp = self.x - c.x;
        let yp = -(self.y - c.y);
        let zp = self.z - c.z;

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

        let bx = e.z / dz * dx + e.x;
        let by = e.z / dz * dy + e.y;

        let d =
            ((c.x - self.x).powf(2.) + (c.y - self.y).powf(2.) + (c.z - self.z).powf(2.)).sqrt();

        (
            Vec2 {
                x: (WIDTH as f32 / 2. + bx * cam.sc).floor() as i32,
                y: (HEIGHT as f32 / 2. + by * cam.sc).floor() as i32,
            },
            d,
        )
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

    fn translate(&self, x: f32, y: f32, z: f32) -> Self {
        Self {
            a: self.a.translate(x, y, z),
            b: self.b.translate(x, y, z),
            c: self.c.translate(x, y, z),
            col: self.col,
        }
    }
}

struct Prim {
    tris: Vec<Tri>,
}

impl Prim {
    fn new(tris: Vec<Tri>) -> Self {
        Self { tris }
    }

    fn translate(&self, x: f32, y: f32, z: f32) -> Self {
        Prim::new(self.tris.iter().map(|t| t.translate(x, y, z)).collect())
    }
}

struct World {
    c: f32,
    cols: [RGBA; 16],
    cam: Camera,
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
                Vec3::new(0., 0., -2.),                           //Vec3::new(0., 1.5, -2.),
                Vec3::new(radians(0.), radians(0.), radians(0.)), //Vec3::new(radians(-30.), radians(30.), 0.),
                Vec3::new_i(0, 0, 200),
                1.,
            ),
        }
    }

    fn update(&mut self) {
        self.c += 1.;

        //self.cam.translate_mut(0.003, 0.002, 0.); //y: 0.002
    }

    fn draw(&self, frame: &mut [u8]) {
        let mut r = Canvas::new(&self.cam, ZBuffer::new());

        r.render_prim(&get_cube());
        r.render_prim(&get_cube().translate(1., 1., 0.));
        /*r.render_tri(Tri::new(
            Vec3::new(1., 1., 0.),
            Vec3::new(2., 1., 0.),
            Vec3::new(1., 2., 0.),
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

    //let mut last_frame_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        // Draw the current frame
        if let Event::RedrawRequested(_) = event {
            //println!("FPS: {:?}", 1000 / last_frame_time.elapsed().as_millis());
            //last_frame_time = Instant::now();

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
                world.cam.pos.x += world.cam.rot.y.sin() * MOVE_SCALE * move_mod; //world.cam.pos.z += 0.5;
                world.cam.pos.y += world.cam.rot.x.sin() * MOVE_SCALE * move_mod;
                world.cam.pos.z += world.cam.rot.y.cos() * MOVE_SCALE * move_mod;
            }
            if input.key_held(VirtualKeyCode::S) {
                world.cam.pos.x -= world.cam.rot.y.sin() * MOVE_SCALE * move_mod; //world.cam.pos.z += -0.5;
                world.cam.pos.y -= world.cam.rot.x.sin() * MOVE_SCALE * move_mod;
                world.cam.pos.z -= world.cam.rot.y.cos() * MOVE_SCALE * move_mod;
            }
            if input.key_held(VirtualKeyCode::A) {
                world.cam.pos.x -= world.cam.rot.y.cos() * MOVE_SCALE * move_mod; //world.cam.pos.x += -0.5;
                world.cam.pos.z += world.cam.rot.y.sin() * MOVE_SCALE * move_mod;
            }
            if input.key_held(VirtualKeyCode::D) {
                world.cam.pos.x += world.cam.rot.y.cos() * MOVE_SCALE * move_mod; //world.cam.pos.x += 0.5;
                world.cam.pos.z -= world.cam.rot.y.sin() * MOVE_SCALE * move_mod;
            }
            if input.key_held(VirtualKeyCode::E) {
                world.cam.pos.y += MOVE_SCALE;
            }
            if input.key_held(VirtualKeyCode::Q) {
                world.cam.pos.y += -MOVE_SCALE;
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
