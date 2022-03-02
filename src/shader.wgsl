//--------/// COMPUTE ///---------//

let G: f32 = 6.67408E-11;

struct Particle {
  pos: vec3<f32>;  // 0, 1, 2
  radius: f32;     // 7
  vel: vec3<f32>;  // 4, 5, 6
  mass: f32;       // 7, 8
};

struct GlobalsBuffer {
  matrix: mat4x4<f32>;
  camera_pos: vec3<f32>;
  particles: u32;
  safety: f32;
  delta: f32;
};

@group(0) 
@binding(0) 
var<uniform> globals: GlobalsBuffer;

@group(0) 
@binding(1) 
var<storage> data_old: array<Particle>;

@group(0) 
@binding(2) 
var<storage, read_write> data: array<Particle>;


fn length2(v: vec3<f32>) -> f32 {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}

@stage(compute) 
@workgroup_size(256)  // PARTICLES PER GROUP: 256
fn comp_main(
  @builtin(global_invocation_id) global_id: vec3<u32>) { 
  
    // Early return
    if(data_old[global_id.x].mass < 0.0) { 
        return;
    }

    // Gravity
    if(globals.delta > 0.0) {
        var temp = vec3<f32>(0.0, 0.0, 0.0);

        // Go through all other particles...
        var i: u32 = 0u;
        loop {
            if i >= globals.particles {
                break;
            }
            // Skip self
            if i == global_id.x {
                continue;
            }
            // If a single particle with no mass is encountered, the entire loop
            // terminates (because they are sorted by mass)
            if(data_old[i].mass == 0.0) { 
                break; 
            }

            var diff: vec3<f32> = data_old[i].pos - data_old[global_id.x].pos;
            temp += normalize(diff) * data_old[i].mass / (length2(diff) + globals.safety);

            continuing {
                i = i + 1u;
            }
        }

        // Update data
        data[global_id.x].vel += vec3<f32>(temp * G * globals.delta);
        data[global_id.x].pos += data[global_id.x].vel * globals.delta;

    }
}

//--------/// FRAGEMENT ///---------//

struct FragmentOutput {
    @location(0) outColor: vec4<f32>;
};

@stage(fragment) 
fn frag_main(
        @location(0) fragColor: vec3<f32>) -> FragmentOutput {
    var fragment_out: FragmentOutput;
    fragment_out.outColor = vec4<f32>(fragColor, 1.0);
    return fragment_out;
}

//--------/// VERTEX ///---------//

struct VertexOutput {
    @builtin(position) vertex_pos: vec4<f32>;
    @location(0) fragColor: vec3<f32>;
};

@stage(vertex)
fn vertex_main(
    @builtin(vertex_index) vertex_id: u32) -> VertexOutput {

    var vertex_out: VertexOutput;

    // Early return
    if(data[vertex_id].mass < 0.0) { 
        return vertex_out;
    }
    
    vertex_out.vertex_pos = globals.matrix * vec4<f32>(data[vertex_id].pos, 1.0);

    if(data[vertex_id].mass > 1.0E33) {
        // Color objects with big mass black
        vertex_out.fragColor = vec3<f32>(0.0, 0.0, 0.0);
    } else {
        // Give different colors to half of all particles
        if(vertex_id < globals.particles / 2u + 1u) {
            vertex_out.fragColor = vec3<f32>(0.722, 0.22, 0.231);
        }
        else {
            vertex_out.fragColor = vec3<f32>(0.345, 0.522, 0.635);
        }
    }
    return vertex_out;
}