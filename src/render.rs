//! This module handles everything that has to do with the window. That includes opening a window,
//! parsing events and rendering. See shader.comp for the physics simulation algorithm.
use std::num::NonZeroU64;
use {
    crate::{Globals, Particle},
    cgmath::{prelude::*, Matrix4, PerspectiveFov, Point3, Quaternion, Rad, Vector3},
    std::{collections::HashSet, f32::consts::PI, time::Instant},
    winit::{
        event,
        event_loop::{ControlFlow, EventLoop},
    },
};
use wgpu::util::DeviceExt as _;

const TICKS_PER_FRAME: u32 = 3; // Number of simulation steps per redraw
const PARTICLES_PER_GROUP: u32 = 256; // REMEMBER TO CHANGE SHADER.COMP

fn build_matrix(pos: Point3<f32>, dir: Vector3<f32>, aspect: f32) -> [[f32; 4]; 4] {
    { 
        Matrix4::from(PerspectiveFov {
            fovy: Rad(PI / 2.0),
            aspect,
            near: 1E8,
            far: 1E14,
        }) * Matrix4::look_to_rh(pos, dir, Vector3::new(0.0, 1.0, 0.0))
    }.into()
}

pub async fn run(mut globals: Globals, particles: Vec<Particle>) {
    // How many bytes do the particles need
    let particles_size = (particles.len() * std::mem::size_of::<Particle>()) as u64;

    let work_group_count = ((particles.len() as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

    let event_loop = EventLoop::new();
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    #[cfg(not(feature = "gl"))]
    let (window, mut size, surface) = {
        let window = winit::window::Window::new(&event_loop).unwrap();

        let size = window.inner_size();
        let surface = unsafe { instance.create_surface(&window) };

        (window, size, surface)
    };

    #[cfg(feature = "gl")]
    let (window, mut size, surface) = {
        let wb = winit::WindowBuilder::new();
        let cb = wgpu::glutin::ContextBuilder::new().with_vsync(true);
        let context = cb.build_windowed(wb, &event_loop).unwrap();

        let size = context
            .window()
            .get_inner_size()
            .unwrap()
            .to_physical(context.window().get_hidpi_factor());

        let (context, window) = unsafe { context.make_current().unwrap().split() };

        let surface = wgpu::Surface::create(&window);

        (window, size, surface)
    };

    // Try to grab mouse
    let _ = window.set_cursor_grab(true);

    window.set_cursor_visible(false);
    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(
        window.primary_monitor(),
    )));

    // Pick a GPU
    let adapter = instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,

        },
    ).await.unwrap();

    println!("{:?}", adapter.get_info());

    // Request access to that GPU
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
            limits: wgpu::Limits::default(),
            label: None,
        },
        None, // Trace path
    ).await.unwrap();

    // Load compute shader for the simulation
    let cs = include_bytes!("shader.comp.spv");
    let cs_data = wgpu::util::make_spirv(cs);
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: cs_data,
    });

    // Load vertex shader to set calculate perspective, size and position of particles
    let vs = include_bytes!("shader.vert.spv");
    let vs_data = wgpu::util::make_spirv(vs);
    let vs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Vertex Shader"),
        source: vs_data,
    });

    // Load fragment shader
    let fs = include_bytes!("shader.frag.spv");
    let fs_data = wgpu::util::make_spirv(fs);
    let fs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Fragment Shader"),
        source: fs_data,
    });

    let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Globals Buffer"),
        contents: bytemuck::cast_slice(&[globals]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create buffer for the previous state of the particles
    let old_buffer = device.create_buffer(&wgpu::BufferDescriptor {  //READ ONLY
        label: Some("Previous State of the Particles Buffer"),
        size: particles_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,                   // or false?
    });

    let current_buffer_initializer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Particles Buffer"),
        contents: bytemuck::cast_slice(&particles),
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    // Create buffer for the current state of the particles
    let current_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Current State of the Particles Buffer"),
        mapped_at_creation: false,
        size: particles_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    // Create swap chain to render images to
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };
    surface.configure(&device, &config);

    // Texture to keep track of which particle is in front (for the camera)
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Front Particle Texture"),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    });
    let mut depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Describe the buffers that will be available to the GPU
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Avaialable Buffers"),
        entries: &[
            // Globals
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Uniform, 
                    has_dynamic_offset: false, 
                    min_binding_size: None 
                },
                count: None,
            },
            // Old Particle data
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                    has_dynamic_offset: false, 
                    min_binding_size: None 
                },
                count: None,
            },
            // Current Particle data
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                    has_dynamic_offset: false, 
                    min_binding_size: None 
                },
                count: None,
            },
        ],
    });

    // Create the resources described by the bind_group_layout
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        label: Some("Resources described by the bind_group_layout"),
        entries: &[
            // Globals
            wgpu::BindGroupEntry {  
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &globals_buffer,
                    offset: 0,
                    size: NonZeroU64::new(std::mem::size_of::<Globals>() as u64)
                }),
            },
            // Old Particle data
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &old_buffer,
                    offset: 0,
                    size: NonZeroU64::new(particles_size)
                }),
            },
            // Current Particle data
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &current_buffer,
                    offset: 0,
                    size: NonZeroU64::new(particles_size)
                }),
            },
        ],
    });

    // Combine all bind_group_layouts
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bind_group_layouts"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute pipeline"),
        layout: Some(&pipeline_layout),
        module:  &cs_module,
        entry_point: "main",
    });

    // Create render pipeline
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState { 
            module: &vs_module, 
            entry_point: "main", 
            buffers: &[] 
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState::IGNORE,
                back: wgpu::StencilFaceState::IGNORE,
                read_mask: 0,
                write_mask: 0,
            },
            bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: "main",
            targets: &[
                wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }
            ],
        }),
        multiview: None,
    });

    // Where is the camera looking at?
    let mut camera_dir = -Vector3::from(globals.camera_pos);
    camera_dir = camera_dir.normalize();
    globals.matrix = build_matrix(
        globals.camera_pos.into(),
        camera_dir,
        size.width as f32 / size.height as f32,
    );

    // Speed of the camera
    let mut fly_speed = 1E10;

    // Which keys are currently held down?
    let mut pressed_keys = HashSet::new();

    // Vector that points to the right of the camera
    let mut right = camera_dir.cross(Vector3::new(0.0, 1.0, 0.0)).normalize();

    // Time of the last tick
    let mut last_tick = Instant::now();

    // Initial setup
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { 
            label: None 
        });

        // Initialize current particle buffer
        encoder.copy_buffer_to_buffer(
            &current_buffer_initializer,
            0,
            &current_buffer,
            0,
            particles_size,
        );

        queue.submit([encoder.finish()]);
    }

    // Start main loop
    event_loop.run(move |event, _, control_flow| {
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            // Move mouse
            event::Event::DeviceEvent {
                event: event::DeviceEvent::MouseMotion { delta },
                ..
            } => {
                camera_dir = Quaternion::from_angle_y(Rad(-delta.0 as f32 / 300.0))
                    .rotate_vector(camera_dir);
                camera_dir = Quaternion::from_axis_angle(right, Rad(delta.1 as f32 / 300.0))
                    .rotate_vector(camera_dir);
            }

            event::Event::WindowEvent { event, .. } => match event {
                // Close window
                event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }

                // Keyboard input
                event::WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    match keycode {
                        // Exit
                        event::VirtualKeyCode::Escape => {
                            *control_flow = ControlFlow::Exit;
                        }
                        event::VirtualKeyCode::Key0 => {
                            globals.delta = 0.0;
                        }
                        event::VirtualKeyCode::Key1 => {
                            globals.delta = 1E0;
                        }
                        event::VirtualKeyCode::Key2 => {
                            globals.delta = 2E0;
                        }
                        event::VirtualKeyCode::Key3 => {
                            globals.delta = 4E0;
                        }
                        event::VirtualKeyCode::Key4 => {
                            globals.delta = 8E0;
                        }
                        event::VirtualKeyCode::Key5 => {
                            globals.delta = 16E0;
                        }
                        event::VirtualKeyCode::Key6 => {
                            globals.delta = 32E0;
                        }
                        event::VirtualKeyCode::F => {
                            let delta = last_tick.elapsed();
                            println!("delta: {:?}, fps: {:.2}", delta, 1.0 / delta.as_secs_f32());
                        }
                        event::VirtualKeyCode::F11 => {
                            if window.fullscreen().is_some() {
                                window.set_fullscreen(None);
                            } else {
                                window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(
                                    window.primary_monitor(),
                                )));
                            }
                        }
                        _ => {}
                    }
                    pressed_keys.insert(keycode);
                }

                // Release key
                event::WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state: event::ElementState::Released,
                            ..
                        },
                    ..
                } => {
                    pressed_keys.remove(&keycode);
                }

                // Mouse scroll
                event::WindowEvent::MouseWheel { delta, .. } => {
                    fly_speed *= (1.0
                        + (match delta {
                            event::MouseScrollDelta::LineDelta(_, c) => c as f32 / 8.0,
                            event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 64.0,
                        }))
                    .min(4.0)
                    .max(0.25);

                    fly_speed = fly_speed.min(1E13).max(1E9);
                }

                // Resize window
                event::WindowEvent::Resized(new_size) => {
                    size = new_size;

                    // Reset swap chain, it's outdated
                    config.width = new_size.width;
                    config.height = new_size.height;
                    surface.configure(&device, &config);

                    // Reset depth texture
                    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Front Particle Texture"),
                        size: wgpu::Extent3d {
                            width: new_size.width,
                            height: new_size.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    });
                    depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
                }
                _ => {}
            },

            // Simulate and redraw
            event::Event::RedrawRequested(_window_id) => {
                let delta = last_tick.elapsed();
                let dt = delta.as_secs_f32();
                last_tick = Instant::now();

                let frame = surface.get_current_texture().unwrap();
                
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { 
                    label: None 
                });

                camera_dir.normalize();
                right = camera_dir.cross(Vector3::new(0.0, 1.0, 0.0));
                right = right.normalize();

                let mut camera_pos = Vector3::from(globals.camera_pos);

                if pressed_keys.contains(&event::VirtualKeyCode::A) {
                    camera_pos += -right * fly_speed * dt;
                    globals.camera_pos = camera_pos.into();
                }

                if pressed_keys.contains(&event::VirtualKeyCode::D) {
                    camera_pos += right * fly_speed * dt;
                    globals.camera_pos = camera_pos.into();
                }

                if pressed_keys.contains(&event::VirtualKeyCode::W) {
                    camera_pos += camera_dir * fly_speed * dt;
                    globals.camera_pos = camera_pos.into();
                }

                if pressed_keys.contains(&event::VirtualKeyCode::S) {
                    camera_pos += -camera_dir * fly_speed * dt;
                    globals.camera_pos = camera_pos.into();
                }

                if pressed_keys.contains(&event::VirtualKeyCode::Space) {
                    camera_pos.y -= fly_speed * dt;
                    globals.camera_pos = camera_pos.into();
                }

                if pressed_keys.contains(&event::VirtualKeyCode::LShift) {
                    camera_pos.y += fly_speed * dt;
                    globals.camera_pos = camera_pos.into();
                }

                globals.matrix = build_matrix(
                    globals.camera_pos.into(),
                    camera_dir,
                    size.width as f32 / size.height as f32,
                );

                // Create new globals buffer
                let new_globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("New Globals Buffer"),
                    contents: bytemuck::cast_slice(&[globals]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
                });

                // Upload the new globals buffer to the GPU
                encoder.copy_buffer_to_buffer(
                    &new_globals_buffer,
                    0,
                    &globals_buffer,
                    0,
                    std::mem::size_of::<Globals>() as u64,
                );

                // Compute the simulation a few times
                for _ in 0..TICKS_PER_FRAME {
                    encoder.copy_buffer_to_buffer(
                        &current_buffer,
                        0,
                        &old_buffer,
                        0,
                        particles_size,
                    );
                    let mut cpass = encoder.begin_compute_pass(
                        &wgpu::ComputePassDescriptor {
                            label: None,
                        }
                    );
                    cpass.set_pipeline(&compute_pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch(work_group_count, 1, 1);
                }

                {
                    // Render the current state
                    let texture_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[wgpu::RenderPassColorAttachment {
                            view: &texture_view,
                            resolve_target: None,
                            ops: wgpu::Operations { 
                                load: wgpu::LoadOp::Clear(
                                    wgpu::Color {
                                        r: 0.03,
                                        g: 0.03,
                                        b: 0.03,
                                        a: 1.0,
                                    }), store: true },
                        }],
                        depth_stencil_attachment: Some(
                            wgpu::RenderPassDepthStencilAttachment {
                                view: &depth_view,
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: true }),   // maybe false
                                stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: true })    // maybe false
                            },
                        ),
                        label: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.draw(0..particles.len() as u32, 0..1);
                }

                queue.submit([encoder.finish()]);
            }

            // No more events in queue
            event::Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}
