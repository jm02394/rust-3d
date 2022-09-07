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

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

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
        
        Vec2 { x: (WIDTH as f32 / 2. + bx).floor(), y: (HEIGHT as f32 / 2. + by).floor() }
    }

    fn process(&self) -> Vec2 {
        self.project(Vec3 { x: 0., y: 0., z: -2.}, Vec3 { x: 0., y: 0., z: 0.}, Vec3 { x: 0., y: 0., z: 1000.})
    }
}

struct Tri {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    color: [i32; 4],
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
            }]
        }
    }

    /// Update the `World` internal state; bounce the box around the screen.
    fn update(&mut self) {
        todo!();
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let x = (i % WIDTH as usize) as i16;
            let y = (i / WIDTH as usize) as i16;

            let pos = (Vec3 { x: 0., y: 0., z: 0. }).process();

            let rgba = if x as f32 == pos.x && y as f32 == pos.y {
                [0, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            };

            pixel.copy_from_slice(&rgba);
        }
    }
}