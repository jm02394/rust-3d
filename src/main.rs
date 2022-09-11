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

use std::mem;

extern crate bresenham;
use bresenham::Bresenham;

const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

fn get_cube() -> Prim {
    Prim { tris: vec![
        //Bottom
        Tri::new(
            Vec3::new_i(1, 0, 0), Vec3::new_i(0, 0, 0), Vec3::new_i(0, 0, 1)
        ),
        Tri::new(
            Vec3::new_i(1, 0, 0), Vec3::new_i(1, 0, 1), Vec3::new_i(0, 0, 1)
        ),
        //Front
        Tri::new(
            Vec3::new_i(0, 0, 0), Vec3::new_i(0, 1, 0), Vec3::new_i(1, 1, 0)
        ),
        Tri::new(
            Vec3::new_i(0, 0, 0), Vec3::new_i(1, 0, 0), Vec3::new_i(1, 1, 0)
        ),
        //Left
        Tri::new(
            Vec3::new_i(0, 1, 0), Vec3::new_i(0, 0, 0), Vec3::new_i(0, 0, 1)
        ),
        Tri::new(
            Vec3::new_i(0, 1, 0), Vec3::new_i(0, 1, 1), Vec3::new_i(0, 0, 1)
        ),
        //Back
        Tri::new(
            Vec3::new_i(1, 0, 1), Vec3::new_i(0, 0, 1), Vec3::new_i(0, 1, 1)
        ),
        Tri::new(
            Vec3::new_i(1, 0, 1), Vec3::new_i(1, 1, 1), Vec3::new_i(0, 1, 1)
        ),
        //Right
        Tri::new(
            Vec3::new_i(1, 0, 0), Vec3::new_i(1, 0, 1), Vec3::new_i(1, 1, 1)
        ),
        Tri::new(
            Vec3::new_i(1, 0, 0), Vec3::new_i(1, 1, 0), Vec3::new_i(1, 1, 1)
        ),
        //Top
        Tri::new(
            Vec3::new_i(1, 1, 0), Vec3::new_i(0, 1, 0), Vec3::new_i(0, 1, 1)
        ),
        Tri::new(
            Vec3::new_i(1, 1, 0), Vec3::new_i(1, 1, 1), Vec3::new_i(0, 1, 1)
        )
    ] }
}

struct Canvas<'a> {
    cam: &'a Camera,
    frame: &'a mut [u8],
}

impl<'a> Canvas<'a> {
    fn new(cam: &'a Camera, frame: &'a mut [u8]) -> Self {
        Self { cam, frame }
    }

    fn draw_px(&mut self, x: isize, y: isize, col: RGBA) {
        if 0 > x || x >= WIDTH as isize || 0 > y || y >= HEIGHT as isize {
            return
        }
        let i = (x * 4 + y * WIDTH as isize * 4) as usize;
        self.frame[i..i + 4].copy_from_slice(&col);
    }

    fn draw_line(&mut self, a: Vec2, b: Vec2, col: RGBA) {
        for (x, y) in Bresenham::new((a.x as isize, a.y as isize), (b.x as isize, b.y as isize)) {
            self.draw_px(x, y, col);
        }
    }

    /*fn draw_line(&mut self, mut a: Vec2, mut b: Vec2, col: RGBA) {
        let steep = (a.x - b.x).abs() < (a.y - b.y).abs();

        if steep { 
            mem::swap(&mut a.x, &mut a.y);
            mem::swap(&mut b.x, &mut b.y);
        }

        if a.x > b.x { mem::swap(&mut a, &mut b) }

        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let err_inc = dy.abs() * 2;
        let mut err = 0;

        let mut y = a.y;

        for x in a.x..=b.x {
            if steep { self.draw_px(y, x, col) }
            else { self.draw_px(x, y, col) }

            err += err_inc;

            if err > dx {
                y += if b.y > a.y { 1 } else { -1 };
                err -= dx * 2;
            }
        }
    }*/

    fn render_tri(&mut self, i: &Tri, col: RGBA) {
        let (a, b, c) = (&mut i.a.project(&self.cam), &mut i.b.project(&self.cam), &mut i.c.project(&self.cam));

        if b.y < a.y { mem::swap(b, a) };
        if c.y < a.y { mem::swap(c, a) };
        if c.y < b.y { mem::swap(c, b) };

        let th = (c.y - a.y) as f32;

        if th != 0. {
            for y in a.y..=b.y {
                let sh = (b.y - a.y) as f32;

                if sh != 0. {
                    let alpha = (y - a.y) as f32 / th;
                    let beta = (y - a.y) as f32 / sh;

                    let lx = a.x as f32 + ((c.x - a.x) as f32 * alpha);
                    let rx = a.x as f32 + ((b.x - a.x) as f32 * beta);

                    self.draw_line(
                        Vec2::new(lx as isize, y), 
                        Vec2::new(rx as isize, y),
                        col
                    );
                }
            }

            for y in b.y..=c.y {
                let sh = (c.y - b.y) as f32;

                if sh != 0. {
                    let alpha = (y - a.y) as f32 / th;
                    let beta = (y - b.y) as f32 / sh;

                    let lx = a.x as f32 + ((c.x - a.x) as f32 * alpha);
                    let rx = b.x as f32 + ((c.x - b.x) as f32 * beta);

                    self.draw_line(
                        Vec2::new(lx as isize, y), 
                        Vec2::new(rx as isize, y),
                        col
                    );
                }
            }
        }

        self.draw_line(*a, *b, [0, 0, 0, 255]);
        self.draw_line(*b, *c, [0, 0, 0, 255]);
        self.draw_line(*a, *c, [0, 0, 0, 255]);
    }

    fn render_prim(&mut self, i: &Prim) {
        for t in i.tris.iter() {
            self.render_tri(t, [255, 0, 0, 255]);
        }
    }
}

struct Camera {
    pos: Vec3,
    rot: Vec3,
    proj: Vec3,
    sc: f32,
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
    x: isize,
    y: isize,
}

impl Vec2 {
    fn new(x: isize, y: isize) -> Self {
        Self { x, y }
    }
}

#[derive(PartialEq)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new_i(x: i32, y: i32, z: i32) -> Self {
        Self { x: x as f32, y: y as f32, z: z as f32 }
    }

    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn translate(&self, x: f32, y: f32, z: f32) -> Self {
        Self { x: self.x + x, y: self.y + y, z: self.z + z }
    }

    fn project(&self, cam: &Camera) -> Vec2 {
        let c = &cam.pos;
        let r = &cam.rot;
        let e = &cam.proj;

        let xp = self.x - c.x;
        let yp = -(self.y - c.y);
        let zp = self.z - c.z;
        
        let (dx, dy, dz) = if r == &Vec3::new_i(0, 0, 0) { (xp, yp, zp) } else {(
            r.y.cos() * (r.z.sin() * yp + r.z.cos() * xp) - r.y.sin() * zp,
            r.x.sin() * (r.y.cos() * zp + r.y.sin() * (r.z.sin() * yp + r.z.cos()  * xp)) + r.x.cos() * (r.z.cos() * yp - r.z.sin() * xp),
            r.x.cos() * (r.y.cos() * zp + r.y.sin() * (r.z.sin() * yp + r.z.cos()  * xp)) - r.x.sin() * (r.z.cos() * yp - r.z.sin() * xp),
        )};

        let bx = e.z / dz * dx + e.x;
        let by = e.z / dz * dy + e.y;
        
        Vec2 { x: (WIDTH as f32 / 2. + bx * cam.sc).floor() as isize, y: (HEIGHT as f32 / 2. + by * cam.sc).floor() as isize }
    }
}

type RGBA = [u8; 4];

struct Tri {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    color: RGBA,
}

impl Tri {
    fn new(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self { a, b, c, color: [0, 0, 0, 255] }
    }

    fn translate(&self, x: f32, y: f32, z: f32) -> Self {
        Self { a: self.a.translate(x, y, z), b: self.b.translate(x, y, z), c: self.c.translate(x, y, z), color: self.color }
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
    tris: Vec<Tri>,
    c: f32,
    cam: Camera,
}

impl World {
    fn new() -> Self {
        Self {
            tris: vec![Tri { 
                a: Vec3::new_i(0, 0, 0),
                b: Vec3::new_i(1, 0, 0),
                c: Vec3::new_i(1, 1, 0),
                color: [0, 0, 0, 255],
            }],
            c: 0.,
            cam: Camera::new(Vec3::new_i(0, 0, -2), Vec3::new_i(0, 0, 0), Vec3::new_i(0, 0, 200), 1.),
        }
    }

    fn update(&mut self) {
        self.c += 1.;

        self.cam.translate_mut(0.001, 0.002, 0.);
    }

    fn draw(&self, frame: &mut [u8]) {
        for (_i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let rgba = [255, 255, 255, 255];

            pixel.copy_from_slice(&rgba);
        }
        
        let mut r = Canvas::new(&self.cam, frame);

        r.render_prim(&get_cube());
        r.render_prim(&get_cube().translate(1., 1., 0.));
    }
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Hello Pixels")
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

    event_loop.run(move |event, _, control_flow| {
        // Draw the current frame
        if let Event::RedrawRequested(_) = event {
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

        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
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