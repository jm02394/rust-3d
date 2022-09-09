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

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 1000;

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

struct Renderer<'a> {
    cam: Camera,
    frame: &'a mut [u8],
}

impl<'a> Renderer<'a> {
    fn new(cam: Camera, frame: &'a mut [u8]) -> Self {
        Self { cam, frame }
    }

    fn px(&mut self, x: usize, y: usize) {
        if x >= WIDTH as usize || y >= HEIGHT as usize {
            return
        }
        let i = x * 4 + y * WIDTH as usize * 4;
        self.frame[i..i + 4].copy_from_slice(&[0, 0, 0, 255]);
    }

    fn draw_line(&mut self, p1: Vec2, p2: Vec2) {
        // Brezhnev
        let dx = (p2.x as i32 - p1.x as i32).abs();
        let sx: i32 = if p1.x < p2.x { 1 } else { -1 };
        let dy = -(p2.y as i32 - p1.y as i32).abs();
        let sy: i32 = if p1.y < p2.y { 1 } else { -1 };
        let mut err = dx + dy;

        let (mut x0, mut y0) = (p1.x as i32, p1.y as i32);

        loop {
            self.px(x0 as usize, y0 as usize);
            if x0 == p2.x as i32 && y0 == p2.y as i32 { break };
            let e2 = 2 * err;

            if e2 >= dy {
                if x0 == p2.x as i32 { break };
                err = err + dy;
                x0 = x0 + sx;
            }
            if e2 <= dy {
                if y0 == p2.x as i32 { break };
                err = err + dx;
                y0 = y0 + sy;
            }
        }

        // shittier
        /*let mut dx = p2.x as i32 - p1.x as i32;
        let mut dy = p2.y as i32 - p1.y as i32;

        if dx == 0 || dy == 0 {
            return
        }

        let step: i32;

        if dx.abs() >= dy.abs() {
            step = dx.abs();
        } else {
            step = dy.abs();
        }

        dx = dx / step;
        dy = dy / step;

        let (mut x, mut y) = (p1.x as i32, p1.y as i32);
        let mut i = 1;

        while i <= step {
            px(x as usize, y as usize, frame);

            x += dx;
            y += dy;
            i += 1;
        }*/
    }

    fn draw_tri(&mut self, i: &Tri) {
        self.draw_line(i.a.project(&self.cam), i.b.project(&self.cam));
        self.draw_line(i.b.project(&self.cam), i.c.project(&self.cam));
        self.draw_line(i.a.project(&self.cam), i.c.project(&self.cam));
    }

    fn draw_prim(&mut self, i: &Prim) {
        for t in i.tris.iter() {
            self.draw_tri(t);
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
}

#[derive(PartialEq, Clone, Copy)]
struct Vec2 {
    x: usize,
    y: usize,
}

impl Vec2 {
    fn new(x: i32, y: i32) -> Self {
        Self { x: x as usize, y: y as usize }
    }

    /*fn line(&self, p2: Vec2, frame: &mut [u8]) {
        let slope = (p2.y - self.y) / (p2.x - self.x);

        if (p2.y - self.y).abs() > (p2.x - self.x).abs() {
            if self.x < p2.x {
                for x in 0..(self.x - p2.x) as u16 {
                    let y = slope * (x as f32 - p2.x) + p2.y;
                    px(x as usize, y as usize, frame);
                }
            } else {
                for x in 0..(p2.x - self.x) as u16 {
                    let y = slope * (x as f32 - self.x) + self.y;
                    px(x as usize, y as usize, frame);
                }
            }
        } else {
            if self.y < p2.y {
                for y in 0..(self.x - p2.x) as u16 {
                    let x = slope * (y as f32 - p2.x) + p2.y;
                    px(x as usize, y as usize, frame);
                }
            } else {
                for y in 0..(p2.x - self.x) as u16 {
                    let x = slope * (y as f32 - self.x) + self.y;
                    px(x as usize, y as usize, frame);
                }
            }
        }
    }*/
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
        
        Vec2 { x: (WIDTH as f32 / 2. + bx * cam.sc).floor() as usize, y: (HEIGHT as f32 / 2. + by * cam.sc).floor() as usize }
    }
}

type RGBA = [i32; 4];

struct Tri2 {
    a: Vec2,
    b: Vec2,
    c: Vec2,
    color: RGBA,
}

impl Tri2 {
    
}

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
}

struct Prim {
    tris: Vec<Tri>,
}
struct World {
    tris: Vec<Tri>,
    c: f32,
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

impl World {
    fn new() -> Self {
        Self {
            tris: vec![Tri { 
                a: Vec3 { x: 0., y: 0., z: 0. },
                b: Vec3 { x: 1., y: 0., z: 0. },
                c: Vec3 { x: 1., y: 1., z: 0. },
                color: [0, 0, 0, 255],
            }],
            c: 0.,
        }
    }

    fn update(&mut self) {
        self.c += 1.;
    }

    fn draw(&self, frame: &mut [u8]) {
        for (_i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let rgba = [255, 255, 255, 255];

            pixel.copy_from_slice(&rgba);
        }
        
        Renderer::new(Camera::new(Vec3::new(self.c * 0.001, self.c * 0.002, -2.), Vec3::new_i(0, 0, 0), Vec3::new_i(0, 0, 200), 5.), frame).draw_prim(&get_cube());
        //draw_line(Vec2::new(WIDTH as i32, HEIGHT as i32), Vec2::new(0, 0), frame);
    }
}