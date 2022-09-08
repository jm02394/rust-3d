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

const WIDTH: u32 = 200;
const HEIGHT: u32 = 200;

fn px(x: usize, y: usize, frame: &mut [u8]) {
    let i = x * 4 + y * WIDTH as usize * 4;
    frame[i..i + 4].copy_from_slice(&[0, 0, 0, 255]);
}

fn line(p1: Vec2, p2: Vec2, frame: &mut [u8]) {
    let dx = (p2.x as i16 - p1.x as i16).abs();
    let sx: i16 = if p1.x < p2.x { 1 } else { -1 };
    let dy = -(p2.y as i16 - p1.y as i16).abs();
    let sy: i16 = if p1.y < p2.y { 1 } else { -1 };
    let mut err = dx + dy;

    let (mut x0, mut y0) = (p1.x as i16, p1.y as i16);

    loop {
        px(x0 as usize, y0 as usize, frame);
        if x0 == p2.x as i16 && y0 == p2.y as i16 { break };
        let e2 = 2 * err;

        if e2 >= dy {
            if x0 == p2.x as i16 { break };
            err = err + dy;
            x0 = x0 + sx;
        }
        if e2 <= dy {
            if y0 == p2.x as i16 { break };
            err = err + dx;
            y0 = y0 + sy;
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
struct Vec2 {
    x: usize,
    y: usize,
}

impl Vec2 {
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
    fn project(&self, c: Vec3, r: Vec3, e: Vec3) -> Vec2 {
        let xp = self.x - c.x;
        let yp = -(self.y - c.y);
        let zp = self.z - c.z;

        let (dx, dy, dz) = match r {
            Vec3 { x: 0., y: 0., z: 0. } => (xp, yp, zp),
            _ => (
                r.y.cos() * (r.z.sin() * yp + r.z.cos() * xp) - r.y.sin() * zp,
                r.x.sin() * (r.y.cos() * zp + r.y.sin() * (r.z.sin() * yp + r.z.cos()  * xp)) + r.x.cos() * (r.z.cos() * yp - r.z.sin() * xp),
                r.x.cos() * (r.y.cos() * zp + r.y.sin() * (r.z.sin() * yp + r.z.cos()  * xp)) - r.x.sin() * (r.z.cos() * yp - r.z.sin() * xp),
            )
        };
        
        let bx = e.z / dz * dx + e.x;
        let by = e.z / dz * dy + e.y;
        
        Vec2 { x: (WIDTH as f32 / 2. + bx).floor() as usize, y: (HEIGHT as f32 / 2. + by).floor() as usize }
    }

    fn process(&self) -> Vec2 {
        self.project(Vec3 { x: 0., y: 0., z: -2.}, Vec3 { x: 0., y: 0., z: 0.}, Vec3 { x: 0., y: 0., z: 1000.})
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

struct Prim {
    tris: Vec<Tri>,
}

/// Representation of the application state. In this example, a box will bounce around the screen.
struct World {
    tris: Vec<Tri>,
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
    /// Create a new `World` instance that can draw a moving box.
    fn new() -> Self {
        Self {
            tris: vec![Tri { 
                a: Vec3 { x: 0., y: 0., z: 0. },
                b: Vec3 { x: 1., y: 0., z: 0. },
                c: Vec3 { x: 1., y: 1., z: 0. },
                color: [0, 0, 0, 255],
            }],
        }
    }

    /// Update the `World` internal state; bounce the box around the screen.
    fn update(&mut self) {
        // fuck yourself
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            /*let x = (i % WIDTH as usize) as i16;
            let y = (i / WIDTH as usize) as i16;

            let pos = (Vec3 { x: 0., y: 0., z: 0. }).process();

            Vec2 { x: 0., y: 0. }.line(Vec2 { x: 10., y: 30. }, frame);

            let rgba = if self.line1.contains(&Vec2 { x: x as f32, y: y as f32 }) {
                [0, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            };*/

            let rgba = [255, 255, 255, 255];

            pixel.copy_from_slice(&rgba);
        }
        
        //Vec2 { x: 0., y: 0. }.line(Vec2 { x: 10., y: 30. }, frame);
        line(Vec2 { x: (WIDTH/2) as usize, y: (HEIGHT/2) as usize }, Vec2 { x: 10, y: 30 }, frame);
    }
}