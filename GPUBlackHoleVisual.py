import glfw
import moderngl
import numpy as np


VERTEX_SHADER = """
#version 330 core

in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = r"""
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform vec2 iResolution;

uniform vec3 camPos;
uniform vec3 camTarget;
uniform vec3 camWorldUp;
uniform float fovYDeg;

uniform float diskInner;
uniform float diskOuter;
uniform float escapeRadius;
uniform float stepSize;

const float R_HORIZON = 2.0;
const float R_CAPTURE_FALLBACK = 2.02;
const float R_ESCAPE_FALLBACK = 150.0;
const float PI = 3.141592653589793;
const int MAX_STEPS = 1024;


struct State {
    vec4 x;
    vec4 p;
};

struct Deriv {
    vec4 dx;
    vec4 dp;
};


float safeSqrt(float x) {
    return sqrt(max(x, 0.0));
}

bool finiteFloat(float x) {
    return abs(x) < 1e30 && x == x;
}

bool finiteVec4(vec4 v) {
    return finiteFloat(v.x) && finiteFloat(v.y) && finiteFloat(v.z) && finiteFloat(v.w);
}

bool finiteVec3(vec3 v) {
    return finiteFloat(v.x) && finiteFloat(v.y) && finiteFloat(v.z);
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 makeStarfield(vec3 dir) {
    float theta = acos(clamp(dir.z, -1.0, 1.0));
    float phi = atan(dir.y, dir.x);
    float u = phi / (2.0 * PI) + 0.5;
    float v = theta / PI;

    vec2 cell = floor(vec2(u, v) * vec2(800.0, 400.0));
    float h = hash21(cell);
    float star = smoothstep(0.997, 1.0, h);

    vec3 bg = mix(vec3(0.008, 0.008, 0.014), vec3(0.035, 0.032, 0.03), dir.z * 0.5 + 0.5);
    bg += star * vec3(1.0, 0.95, 0.85) * 1.6;
    return bg;
}

vec3 diskColor(float r) {
    float t = clamp((r - diskInner) / (diskOuter - diskInner), 0.0, 1.0);
    vec3 inner = vec3(1.0, 0.95, 0.85);
    vec3 outer = vec3(1.0, 0.35, 0.05);
    vec3 color = mix(inner, outer, t);
    float brightness = 2.0 / pow(max(r, 1.0), 0.8);
    return clamp(color * brightness, 0.0, 10.0);
}

void cameraBasis(out vec3 forward, out vec3 right, out vec3 up) {
    forward = normalize(camTarget - camPos);
    right = normalize(cross(forward, camWorldUp));
    up = normalize(cross(right, forward));
}

vec3 pixelRayDir(vec2 fragCoord) {
    vec3 forward;
    vec3 right;
    vec3 up;
    cameraBasis(forward, right, up);

    float aspect = iResolution.x / iResolution.y;
    float fov = radians(fovYDeg);

    float x = (2.0 * ((fragCoord.x + 0.5) / iResolution.x) - 1.0) * tan(fov * 0.5) * aspect;
    float y = (1.0 - 2.0 * ((fragCoord.y + 0.5) / iResolution.y)) * tan(fov * 0.5);

    return normalize(forward + x * right + y * up);
}

void cartesianToSchwarzschild(vec3 pos, out float r, out float theta, out float phi) {
    r = length(pos);
    theta = acos(clamp(pos.z / r, -1.0, 1.0));
    phi = atan(pos.y, pos.x);
}

vec3 schwarzschildToCartesian(float r, float theta, float phi) {
    float st = sin(theta);
    return vec3(
        r * st * cos(phi),
        r * st * sin(phi),
        r * cos(theta)
    );
}

void cartesianToCovariantMomentum(
    vec3 pos,
    vec3 dir,
    out float p_r,
    out float p_theta,
    out float p_phi
) {
    float x = pos.x;
    float y = pos.y;
    float z = pos.z;

    float dx = dir.x;
    float dy = dir.y;
    float dz = dir.z;

    float r = length(pos);
    float rho = max(length(pos.xy), 1e-6);
    float f = 1.0 - 2.0 / r;

    float p_r_contra = (x * dx + y * dy + z * dz) / r;
    float p_theta_contra = (z * (x * dx + y * dy) - rho * rho * dz) / (r * r * rho);
    float p_phi_contra = (x * dy - y * dx) / (rho * rho);

    p_r = p_r_contra / max(f, 1e-6);
    p_theta = p_theta_contra * r * r;
    p_phi = p_phi_contra * rho * rho;
}

float null_pt(float r, float theta, float p_r, float p_theta, float p_phi) {
    float s = max(sin(theta), 1e-6);
    float f = 1.0 - 2.0 / r;

    float term =
        f * f * p_r * p_r +
        f * p_theta * p_theta / (r * r) +
        f * p_phi * p_phi / (r * r * s * s);

    return -safeSqrt(term);
}

State initRay(vec3 pos, vec3 dir) {
    State s;

    float r;
    float theta;
    float phi;
    cartesianToSchwarzschild(pos, r, theta, phi);

    float p_r;
    float p_theta;
    float p_phi;
    cartesianToCovariantMomentum(pos, dir, p_r, p_theta, p_phi);

    float p_t = null_pt(r, theta, p_r, p_theta, p_phi);

    s.x = vec4(0.0, r, theta, phi);
    s.p = vec4(p_t, p_r, p_theta, p_phi);
    return s;
}

Deriv deriv(State s) {
    Deriv d;

    float r = s.x.y;
    float theta = s.x.z;

    float p_t = s.p.x;
    float p_r = s.p.y;
    float p_theta = s.p.z;
    float p_phi = s.p.w;

    float sin_t = max(sin(theta), 1e-6);
    float cos_t = cos(theta);
    float f = 1.0 - 2.0 / r;

    d.dx.x = -p_t / max(f, 1e-6);
    d.dx.y = f * p_r;
    d.dx.z = p_theta / (r * r);
    d.dx.w = p_phi / (r * r * sin_t * sin_t);

    d.dp.x = 0.0;
    d.dp.y =
        -(p_t * p_t) / (r * r * f * f)
        - (p_r * p_r) / (r * r)
        + (p_theta * p_theta) / (r * r * r)
        + (p_phi * p_phi) / (r * r * r * sin_t * sin_t);

    d.dp.z = cos_t * p_phi * p_phi / (r * r * sin_t * sin_t * sin_t);
    d.dp.w = 0.0;

    return d;
}

State addState(State s, Deriv d, float h) {
    State outS;
    outS.x = s.x + h * d.dx;
    outS.p = s.p + h * d.dp;
    return outS;
}

State rk4Step(State s, float h) {
    Deriv k1 = deriv(s);
    Deriv k2 = deriv(addState(s, k1, 0.5 * h));
    Deriv k3 = deriv(addState(s, k2, 0.5 * h));
    Deriv k4 = deriv(addState(s, k3, h));

    State outS;
    outS.x = s.x + (h / 6.0) * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
    outS.p = s.p + (h / 6.0) * (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp);
    return outS;
}

bool segmentHitsDisk(vec3 p0, vec3 p1, out vec3 hit) {
    float z0 = p0.z;
    float z1 = p1.z;

    if (z0 == 0.0) {
        hit = p0;
    } else if (z1 == 0.0) {
        hit = p1;
    } else if (z0 * z1 > 0.0) {
        return false;
    } else {
        float t = -z0 / (z1 - z0);
        if (t < 0.0 || t > 1.0) {
            return false;
        }
        hit = mix(p0, p1, t);
    }

    float r = length(hit);
    return r >= diskInner && r <= diskOuter;
}

vec3 traceRay(vec3 rayOrigin, vec3 rayDir) {
    State s = initRay(rayOrigin, rayDir);
    vec3 prevPos = schwarzschildToCartesian(s.x.y, s.x.z, s.x.w);
    float initialR = s.x.y;
    float maxRSeen = initialR;
    float minRSeen = initialR;
    bool becameInvalid = false;

    for (int i = 0; i < MAX_STEPS; i++) {
        s = rk4Step(s, stepSize);

        float r = s.x.y;
        maxRSeen = max(maxRSeen, r);
        minRSeen = min(minRSeen, r);

        if (r <= R_HORIZON) {
            return vec3(0.0);
        }

        if (!finiteVec4(s.x) || !finiteVec4(s.p) || !finiteFloat(r) || r <= 0.0) {
            becameInvalid = true;
            break;
        }

        vec3 pos = schwarzschildToCartesian(s.x.y, s.x.z, s.x.w);
        if (!finiteVec3(pos)) {
            becameInvalid = true;
            break;
        }

        vec3 hit;
        if (segmentHitsDisk(prevPos, pos, hit)) {
            return diskColor(length(hit));
        }

        if (r >= escapeRadius) {
            return makeStarfield(normalize(pos));
        }

        prevPos = pos;
    }

    if (becameInvalid && minRSeen <= R_CAPTURE_FALLBACK) {
        return vec3(0.0);
    }

    if (maxRSeen > initialR + 5.0) {
        return makeStarfield(normalize(prevPos));
    }

    if (maxRSeen >= R_ESCAPE_FALLBACK) {
        return makeStarfield(normalize(prevPos));
    }

    if (minRSeen < initialR - 5.0) {
        return vec3(0.08, 0.02, 0.02);
    }

    if (s.p.y < -0.05 || minRSeen < initialR - 1.0) {
        return vec3(0.12, 0.03, 0.03);
    }

    if (s.p.y > 0.05 || maxRSeen > initialR + 1.0) {
        return makeStarfield(normalize(prevPos));
    }

    return makeStarfield(normalize(prevPos));
}

void main() {
    vec2 fragCoord = v_uv * iResolution;
    vec3 rayDir = pixelRayDir(fragCoord);
    vec3 color = traceRay(camPos, rayDir);

    color = color / (1.0 + color);
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, 1.0);
}
"""


def main():
    window = None
    try:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        width, height = 1280, 720
        window = glfw.create_window(width, height, "GPU Black Hole", None, None)
        if not window:
            raise RuntimeError("glfw.create_window() failed")

        glfw.make_context_current(window)
        ctx = moderngl.create_context()
        print("OpenGL version:", ctx.info["GL_VERSION"])
        prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        quad = np.array(
            [
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                 1.0,  1.0,
            ],
            dtype="f4",
        )

        vbo = ctx.buffer(quad.tobytes())
        vao = ctx.simple_vertex_array(prog, vbo, "in_pos")

        cam_pos = np.array([0.0, -32.0, 9.0], dtype="f4")
        cam_target = np.array([0.0, 0.0, 0.0], dtype="f4")
        cam_up = np.array([0.0, 0.0, 1.0], dtype="f4")

        while not glfw.window_should_close(window):
            glfw.poll_events()
            fb_width, fb_height = glfw.get_framebuffer_size(window)
            ctx.viewport = (0, 0, fb_width, fb_height)
            ctx.clear(0.0, 0.0, 0.0, 1.0)

            prog["iResolution"].value = (fb_width, fb_height)
            prog["camPos"].value = tuple(cam_pos)
            prog["camTarget"].value = tuple(cam_target)
            prog["camWorldUp"].value = tuple(cam_up)
            prog["fovYDeg"].value = 45.0
            prog["diskInner"].value = 6.0
            prog["diskOuter"].value = 16.0
            prog["escapeRadius"].value = 180.0
            prog["stepSize"].value = 0.02

            vao.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(window)
    except Exception as exc:
        print("GPUBlackHoleVisual failed:")
        print(exc)
        input("Press Enter to close...")
    finally:
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
