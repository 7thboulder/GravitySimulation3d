from pathlib import Path

import glfw
import moderngl
import numpy as np
from PIL import Image


WIDTH = 1280
HEIGHT = 720
FOV_DEG = 45.0
SAMPLES_PER_AXIS = 3
GEODESIC_STEPS = 3000
PHI_MAX = 24.0 * np.pi
EXPOSURE = 1.0

CAMERA_POS = np.array([0.0, -32.0, 3.0], dtype="f4")
CAMERA_TARGET = np.array([0.0, 0.0, 3.0], dtype="f4")
CAMERA_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype="f4")

R_HORIZON = 2.0
R_DISK_IN = 6.0
R_DISK_OUT = 16.0
DISK_HALF_THICKNESS = 0.24
# DISK_HALF_THICKNESS = 0.24
R_ESCAPE = 180.0

BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_IMAGE_PATH = BASE_DIR / "space_background.png"
OUTPUT_IMAGE_PATH = BASE_DIR / "gpu_planar_black_hole_render.png"


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

uniform sampler2D backgroundTex;
uniform int useBackgroundTexture;

uniform float horizonRadius;
uniform float diskInner;
uniform float diskOuter;
uniform float diskHalfThickness;
uniform float escapeRadius;
uniform float phiMax;
uniform int geodesicSteps;
uniform int sampleAxis;
uniform float exposure;

const float PI = 3.14159265358979323846;
const int MAX_GEODESIC_STEPS = 4096;
const int MAX_AA_AXIS = 3;
const int SEGMENT_SAMPLES = 24;
const int INTERSECTION_REFINE_STEPS = 7;
const int VOLUME_STEPS = 14;


struct Plane {
    float r0;
    vec3 e1;
    vec3 e2;
    vec3 e3;
    float radialComponent;
    float tangentialComponent;
};

struct DiskHit {
    vec3 point;
    float surfaceSign;
    float thicknessMix;
    float zBlend;
    float radialTaper;
    float t0;
    float t1;
};


bool finiteFloat(float x) {
    return abs(x) < 1e20 && x == x;
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    mat2 warp = mat2(1.62, 1.18, -1.18, 1.62);

    for (int octave = 0; octave < 4; octave++) {
        sum += amp * valueNoise(p);
        norm += amp;
        p = warp * p + vec2(7.13, -3.71);
        amp *= 0.5;
    }

    return sum / max(norm, 1e-6);
}

vec3 safeNormalize(vec3 v) {
    float len = length(v);
    if (len < 1e-10) {
        return vec3(0.0, 0.0, 1.0);
    }
    return v / len;
}

void cameraBasis(out vec3 forward, out vec3 right, out vec3 up) {
    forward = safeNormalize(camTarget - camPos);
    right = cross(forward, camWorldUp);

    if (length(right) < 1e-10) {
        right = vec3(1.0, 0.0, 0.0);
    } else {
        right = normalize(right);
    }

    up = normalize(cross(right, forward));
}

float sampleOffset(int sampleIndex, int axis) {
    if (axis <= 1) {
        return 0.5;
    }
    return mix(0.25, 0.75, float(sampleIndex) / float(axis - 1));
}

vec3 subpixelRayDir(vec2 fragCoord, vec2 subpixelOffset) {
    vec3 forward;
    vec3 right;
    vec3 up;
    cameraBasis(forward, right, up);

    float fov = radians(fovYDeg);
    float aspect = iResolution.x / iResolution.y;
    vec2 pixel = fragCoord - vec2(0.5) + subpixelOffset;

    float x = (2.0 * (pixel.x / iResolution.x) - 1.0) * tan(fov * 0.5) * aspect;
    float y = (2.0 * (pixel.y / iResolution.y) - 1.0) * tan(fov * 0.5);

    return normalize(forward + x * right + y * up);
}

Plane makeOrbitalPlane(vec3 rayOrigin, vec3 rayDir) {
    float r0 = length(rayOrigin);
    vec3 e1 = rayOrigin / max(r0, 1e-10);

    vec3 normal = cross(rayOrigin, rayDir);
    float normalLen = length(normal);
    if (normalLen < 1e-10) {
        normal = cross(rayOrigin, vec3(0.0, 0.0, 1.0));
        normalLen = length(normal);
        if (normalLen < 1e-10) {
            normal = cross(rayOrigin, vec3(1.0, 0.0, 0.0));
            normalLen = length(normal);
        }
    }

    vec3 e3 = normal / max(normalLen, 1e-10);
    vec3 e2 = normalize(cross(e3, e1));

    float radialComponent = dot(rayDir, e1);
    float tangentialComponent = dot(rayDir, e2);

    if (tangentialComponent < 0.0) {
        e2 = -e2;
        e3 = -e3;
        tangentialComponent = -tangentialComponent;
    }

    return Plane(r0, e1, e2, e3, radialComponent, tangentialComponent);
}

vec2 orbitDeriv(vec2 state) {
    float u = state.x;
    float du = state.y;
    return vec2(du, 3.0 * u * u - u);
}

vec2 rk4OrbitStep(vec2 state, float h) {
    vec2 k1 = orbitDeriv(state);
    vec2 k2 = orbitDeriv(state + 0.5 * h * k1);
    vec2 k3 = orbitDeriv(state + 0.5 * h * k2);
    vec2 k4 = orbitDeriv(state + h * k3);
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

float diskHalfThicknessAtRadius(float rho) {
    float t = clamp((rho - diskInner) / (diskOuter - diskInner), 0.0, 1.0);
    return diskHalfThickness * (1.12 - 0.52 * t + 0.08 * sin(PI * t));
}

bool pointInsideDiskVolume(vec3 point) {
    float rho = length(point.xy);
    if (rho < diskInner || rho > diskOuter) {
        return false;
    }

    float localHalfThickness = diskHalfThicknessAtRadius(rho);
    return abs(point.z) <= localHalfThickness;
}

float refineDiskBoundary(vec3 p0, vec3 p1, float tA, float tB, bool aInside) {
    float lo = tA;
    float hi = tB;

    for (int i = 0; i < INTERSECTION_REFINE_STEPS; i++) {
        float mid = 0.5 * (lo + hi);
        bool midInside = pointInsideDiskVolume(mix(p0, p1, mid));
        if (midInside == aInside) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return 0.5 * (lo + hi);
}

bool segmentDiskIntersection(vec3 p0, vec3 p1, out DiskHit hit) {
    float z0 = p0.z;
    float z1 = p1.z;
    float dz = z1 - z0;

    if (abs(dz) < 1e-7) {
        return false;
    }

    float hitT = -z0 / dz;
    if (hitT < 0.0 || hitT > 1.0) {
        return false;
    }

    vec3 point = mix(p0, p1, hitT);
    float rho = length(point.xy);
    if (rho < diskInner || rho > diskOuter) {
        return false;
    }

    float localHalfThickness = diskHalfThicknessAtRadius(rho);
    float radialTaper = clamp(localHalfThickness / max(diskHalfThickness, 1e-10), 0.0, 1.5);

    hit.point = point;
    hit.surfaceSign = z0 >= 0.0 ? 1.0 : -1.0;
    hit.thicknessMix = 0.22;
    hit.zBlend = 1.0;
    hit.radialTaper = radialTaper;
    hit.t0 = max(hitT - 0.015, 0.0);
    hit.t1 = min(hitT + 0.015, 1.0);
    return true;
}

vec3 diskTangentVelocity(vec3 hitPoint) {
    float rho = length(hitPoint.xy);
    if (rho < 1e-10) {
        return vec3(0.0);
    }

    vec3 tangent = vec3(-hitPoint.y / rho, hitPoint.x / rho, 0.0);
    float beta = sqrt(clamp(1.0 / max(rho - 2.0, 1e-6), 0.0, 0.45));
    return beta * tangent;
}

float diskTexture(vec3 hitPoint) {
    float rho = length(hitPoint.xy);
    float phi = atan(hitPoint.y, hitPoint.x);
    float logR = log(max(rho, 1e-6));

    vec2 diskCoord = vec2(0.72 * logR - 0.045 * phi, 0.18 * phi + 0.012 * rho);
    float broad = fbm(diskCoord);
    float spiral = 0.5 + 0.5 * sin(2.2 * phi - 4.8 * logR + 0.55 * broad);

    float innerRim = exp(-pow((rho - diskInner) / 1.85, 2.0));
    float turbulence = 0.92 + 0.08 * (broad - 0.5) + 0.10 * spiral + 0.08 * innerRim;
    return clamp(turbulence, 0.78, 1.18);
}

float diskRadialFeather(float rho) {
    float inner = smoothstep(diskInner, diskInner + 1.8, rho);
    float outer = 1.0 - smoothstep(diskInner + 3.4, diskOuter, rho);
    return inner * outer;
}

float gasDensity(vec3 point) {
    float rho = length(point.xy);
    if (rho < diskInner || rho > diskOuter) {
        return 0.0;
    }

    float localHalfThickness = diskHalfThicknessAtRadius(rho);
    float z = abs(point.z) / max(localHalfThickness, 1e-6);
    float verticalCore = exp(-3.1 * z * z);
    float warmAtmosphere = 0.18 * exp(-0.75 * z * z);
    float radialFalloff = pow(diskInner / max(rho, diskInner), 2.35);
    float innerPileup = 1.0 + 0.85 * exp(-pow((rho - diskInner) / 1.4, 2.0));

    return diskRadialFeather(rho)
        * radialFalloff
        * innerPileup
        * diskTexture(point)
        * (verticalCore + warmAtmosphere)
        * 1.18;
}

vec3 diskColor(vec3 hitPoint, vec3 outgoingDir, float surfaceSign, float thicknessMix, float zBlend, float radialTaper) {
    float rho = length(hitPoint.xy);
    float t = clamp((rho - diskInner) / (diskOuter - diskInner), 0.0, 1.0);

    vec3 inner = vec3(1.0, 0.985, 0.93);
    vec3 mid = vec3(1.0, 0.58, 0.18);
    vec3 outer = vec3(0.72, 0.13, 0.02);

    vec3 color;
    if (t < 0.45) {
        color = mix(inner, mid, t / 0.45);
    } else {
        color = mix(mid, outer, (t - 0.45) / 0.55);
    }

    float brightness = 4.8 / pow(rho, 0.86);
    float glow = 0.18 / max(rho - horizonRadius, 0.35);
    float innerSharpen = 1.0 + 0.62 * exp(-pow((rho - diskInner) / 0.9, 2.0));

    vec3 velocity = diskTangentVelocity(hitPoint);
    float beta = length(velocity);
    float viewCos = 0.0;
    if (beta > 1e-8) {
        viewCos = clamp(dot(velocity / beta, -outgoingDir), -1.0, 1.0);
    }

    float gamma = 1.0 / sqrt(max(1.0 - beta * beta, 1e-6));
    float doppler = 1.0 / (gamma * max(1.0 - beta * viewCos, 1e-6));
    float gravShift = sqrt(max(1.0 - 2.0 / max(rho, horizonRadius + 1e-6), 1e-6));

    float intensityBoost = pow(doppler, 3.0);
    float netShift = doppler * gravShift;

    if (netShift >= 1.0) {
        float shiftStrength = min(netShift - 1.0, 0.5);
        color = mix(color, vec3(1.0, 0.97, 0.9), shiftStrength);
    } else {
        float shiftStrength = min(1.0 - netShift, 0.7);
        color = mix(color, vec3(0.75, 0.16, 0.03), shiftStrength);
    }

    float surfaceFactor = surfaceSign >= 0.0 ? 1.04 : 0.88;
    float textureFactor = diskTexture(hitPoint);
    float thicknessFactor = (0.92 + 0.12 * thicknessMix) * (0.94 + 0.12 * radialTaper);
    float viewAngle = clamp(abs(outgoingDir.z), 0.0, 1.0);
    float opticalDepth = 0.55 + 1.4 * exp(-pow((rho - diskInner) / 2.8, 2.0)) + 0.35 * textureFactor;
    float transmission = exp(-opticalDepth * (0.35 + 1.1 * (1.0 - viewAngle)) * (0.8 + 0.4 * thicknessMix));
    float opacityFactor = 0.32 + 0.68 * (1.0 - transmission);
    float edgeSoftening = 0.58 + 0.42 * pow(zBlend, 0.8);

    return clamp(
        color
        * (brightness + glow)
        * innerSharpen
        * intensityBoost
        * surfaceFactor
        * textureFactor
        * thicknessFactor
        * edgeSoftening
        * opacityFactor,
        0.0,
        6.0
    );
}

vec4 accumulateDiskVolume(vec3 p0, vec3 p1, vec3 rayDir, float t0, float t1) {
    vec3 segment = p1 - p0;
    float segmentLen = length(segment);
    if (segmentLen < 1e-10) {
        return vec4(0.0);
    }

    vec3 accum = vec3(0.0);
    float transmittance = 1.0;
    float localStepLen = segmentLen * max(t1 - t0, 0.0) / float(VOLUME_STEPS);

    for (int step = 0; step < VOLUME_STEPS; step++) {
        float s = (float(step) + 0.5) / float(VOLUME_STEPS);
        float t = mix(t0, t1, s);
        vec3 point = p0 + t * segment;

        float rho = length(point.xy);
        if (rho < diskInner || rho > diskOuter) {
            continue;
        }

        float localHalfThickness = diskHalfThicknessAtRadius(rho);
        float zBlend = clamp(1.0 - abs(point.z) / max(localHalfThickness, 1e-10), 0.0, 1.0);
        if (zBlend <= 0.0) {
            continue;
        }

        float density = gasDensity(point);
        if (density <= 1e-4) {
            continue;
        }

        float surfaceSign = point.z >= 0.0 ? 1.0 : -1.0;
        float radialTaper = clamp(localHalfThickness / max(diskHalfThickness, 1e-10), 0.0, 1.5);
        float thicknessMix = clamp(localStepLen / (2.0 * localHalfThickness + 1e-10), 0.0, 1.0);

        vec3 baseColor = diskColor(point, rayDir, surfaceSign, thicknessMix, zBlend, radialTaper);
        float emissivity = density * (0.85 + 1.45 * pow(zBlend, 1.4));
        float opticalDepth = clamp(density * localStepLen * 1.65, 0.0, 3.0);
        float absorption = 1.0 - exp(-opticalDepth);

        accum += transmittance * baseColor * emissivity * absorption * 1.55;
        transmittance *= exp(-opticalDepth * 0.92);

        if (transmittance < 0.035) {
            break;
        }
    }

    float alpha = clamp(1.0 - transmittance, 0.0, 1.0);
    return vec4(clamp(accum, 0.0, 6.0), alpha);
}

vec4 shadeGasDisk(DiskHit hit, vec3 outgoingDir) {
    float rho = length(hit.point.xy);
    float density = gasDensity(hit.point);
    float turbulence = diskTexture(hit.point);
    float radialFeather = diskRadialFeather(rho);

    float innerHeat = exp(-pow((rho - diskInner) / 1.7, 2.0));
    float surfaceDensity = smoothstep(0.05, 1.15, density) * radialFeather;
    float alpha = surfaceDensity * (0.20 + 0.20 * innerHeat);
    alpha *= 0.92 + 0.08 * turbulence;
    alpha = clamp(alpha, 0.0, 0.34);

    vec3 emission = diskColor(
        hit.point,
        outgoingDir,
        hit.surfaceSign,
        0.12 + 0.22 * alpha,
        0.42 + 0.48 * alpha,
        hit.radialTaper
    );

    float hotFilament = smoothstep(0.98, 1.16, turbulence);
    float thermalGlow = 0.85 + 0.52 * density + 0.28 * hotFilament + 0.58 * innerHeat;
    emission *= thermalGlow;

    float filament = 0.78 + 0.22 * sin(2.4 * atan(hit.point.y, hit.point.x) - 5.1 * log(max(rho, 1e-6)) + 0.7 * turbulence);
    vec3 gasGlow = emission * alpha * (0.08 + 4.2 * alpha) * filament * 2.0;
    return vec4(clamp(gasGlow, 0.0, 6.0), alpha);
}

vec3 smoothBackgroundColor(vec3 direction) {
    float u = 0.5 * (direction.z + 1.0);
    vec3 horizon = vec3(0.006, 0.004, 0.005);
    vec3 zenith = vec3(0.0006, 0.0009, 0.0025);
    vec3 base = mix(horizon, zenith, u);

    float theta = acos(clamp(direction.z, -1.0, 1.0));
    float phi = atan(direction.y, direction.x);
    float bandPhase = phi + 0.35 * sin(2.0 * theta);
    float bandCenter = PI * 0.52 + 0.12 * sin(1.7 * bandPhase);
    float bandWidth = exp(-pow((theta - bandCenter) / 0.17, 2.0));

    return clamp(base + bandWidth * vec3(0.08, 0.075, 0.065), 0.0, 1.0);
}

vec3 sampleBackgroundImage(vec3 direction) {
    float phi = atan(direction.y, direction.x);
    float theta = acos(clamp(direction.z, -1.0, 1.0));

    float u = fract(phi / (2.0 * PI) + 0.5);
    float v = clamp(theta / PI, 0.0, 1.0);
    return clamp(texture(backgroundTex, vec2(u, v)).rgb, 0.0, 3.5);
}

vec3 proceduralBackgroundColor(vec3 direction) {
    float u = 0.5 * (direction.z + 1.0);
    vec3 horizon = vec3(0.006, 0.004, 0.005);
    vec3 zenith = vec3(0.0006, 0.0009, 0.0025);
    vec3 base = mix(horizon, zenith, u);

    float theta = acos(clamp(direction.z, -1.0, 1.0));
    float phi = atan(direction.y, direction.x);

    float bandPhase = phi + 0.35 * sin(2.0 * theta);
    float bandCenter = PI * 0.52 + 0.12 * sin(1.7 * bandPhase);
    float bandWidth = exp(-pow((theta - bandCenter) / 0.12, 2.0));
    float bandTexture =
        0.55
        + 0.25 * sin(18.0 * phi)
        + 0.18 * sin(47.0 * phi + 3.0 * theta)
        + 0.12 * sin(95.0 * phi - 11.0 * theta);
    bandTexture = clamp(bandTexture, 0.0, 1.2);
    float galacticBand = bandWidth * bandTexture;

    float a = sin(1234.567 * phi + 321.123 * theta);
    float b = sin(6789.123 * phi - 987.654 * theta);
    float c = sin(4567.891 * (phi + theta));
    float d = sin(15317.337 * phi + 2211.731 * theta);
    float e = sin(31111.113 * phi - 7133.219 * theta);

    float starField = (a + b + c + d + e) / 5.0;
    float starCore = step(0.998, starField);
    float starHalo = step(0.994, starField);
    float starMid = step(0.989, starField);

    vec3 color = base;
    color += galacticBand * vec3(0.13, 0.12, 0.10);
    color += starMid * vec3(0.015, 0.016, 0.02);
    color += starHalo * vec3(0.12, 0.13, 0.16);
    color += starCore * vec3(1.6, 1.5, 1.35);
    return clamp(color, 0.0, 3.5);
}

vec3 backgroundColor(vec3 direction) {
    if (useBackgroundTexture == 1) {
        return sampleBackgroundImage(direction);
    }
    return proceduralBackgroundColor(direction);
}

vec3 traceRay(vec3 rayOrigin, vec3 rayDir) {
    Plane plane = makeOrbitalPlane(rayOrigin, rayDir);

    if (plane.tangentialComponent < 1e-8) {
        if (plane.radialComponent < 0.0) {
            return vec3(0.0);
        }
        return backgroundColor(rayDir);
    }

    float u = 1.0 / plane.r0;
    float du = -plane.radialComponent / (plane.r0 * plane.tangentialComponent);
    float phi = 0.0;
    vec3 prevPos = plane.r0 * plane.e1;
    vec3 prevTravelDir = rayDir;
    float h = phiMax / max(float(geodesicSteps), 1.0);

    for (int step = 0; step < MAX_GEODESIC_STEPS; step++) {
        if (step >= geodesicSteps) {
            break;
        }

        vec2 nextState = rk4OrbitStep(vec2(u, du), h);
        float nextU = nextState.x;
        float nextPhi = phi + h;

        if (!finiteFloat(nextU) || !finiteFloat(nextState.y)) {
            return backgroundColor(prevTravelDir);
        }

        if (nextU <= 0.0 || nextU <= 1.0 / escapeRadius) {
            return backgroundColor(prevTravelDir);
        }

        float r = 1.0 / nextU;
        vec3 pos = r * cos(nextPhi) * plane.e1 + r * sin(nextPhi) * plane.e2;
        vec3 segment = pos - prevPos;
        float segmentLen = length(segment);
        vec3 outgoingDir = segmentLen < 1e-10 ? prevTravelDir : segment / segmentLen;

        DiskHit hit;
        if (segmentDiskIntersection(prevPos, pos, hit)) {
            vec4 diskVolume = shadeGasDisk(hit, outgoingDir);
            if (diskVolume.a < 0.025) {
                prevPos = pos;
                prevTravelDir = outgoingDir;
                u = nextState.x;
                du = nextState.y;
                phi = nextPhi;
                continue;
            }

            vec3 sharpBackground = backgroundColor(outgoingDir);
            vec3 smoothBackground = smoothBackgroundColor(outgoingDir);
            float gasMask = smoothstep(0.04, 0.20, diskVolume.a);
            vec3 backgroundBehindGas = mix(sharpBackground, smoothBackground, gasMask);
            float transmission = pow(1.0 - diskVolume.a, 1.15);
            return clamp(diskVolume.rgb + backgroundBehindGas * transmission * (0.35 + 0.45 * (1.0 - gasMask)), 0.0, 6.0);
        }

        if (nextU >= 1.0 / horizonRadius || r <= horizonRadius) {
            return vec3(0.0);
        }

        if (r >= escapeRadius) {
            return backgroundColor(outgoingDir);
        }

        u = nextState.x;
        du = nextState.y;
        phi = nextPhi;
        prevPos = pos;
        prevTravelDir = outgoingDir;
    }

    return backgroundColor(prevTravelDir);
}

vec3 toneMap(vec3 color) {
    vec3 mapped = color / (vec3(1.0) + color);
    return pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
}

void main() {
    int axis = clamp(sampleAxis, 1, MAX_AA_AXIS);
    vec3 color = vec3(0.0);

    for (int sy = 0; sy < MAX_AA_AXIS; sy++) {
        if (sy >= axis) {
            break;
        }
        for (int sx = 0; sx < MAX_AA_AXIS; sx++) {
            if (sx >= axis) {
                break;
            }
            vec2 offset = vec2(sampleOffset(sx, axis), sampleOffset(sy, axis));
            vec3 rayDir = subpixelRayDir(gl_FragCoord.xy, offset);
            color += traceRay(camPos, rayDir);
        }
    }

    color = color / float(axis * axis);
    color *= exposure;
    fragColor = vec4(toneMap(color), 1.0);
}
"""


def set_uniform(program, name, value):
    try:
        program[name].value = value
    except KeyError:
        pass


def load_background_texture(ctx):
    if not BACKGROUND_IMAGE_PATH.exists():
        print(f"No background image found at {BACKGROUND_IMAGE_PATH}; using procedural stars.")
        return None

    image = Image.open(BACKGROUND_IMAGE_PATH).convert("RGB")
    texture = ctx.texture(image.size, 3, image.tobytes())
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = True
    texture.repeat_y = False
    print(f"Loaded background texture: {BACKGROUND_IMAGE_PATH} ({image.width}x{image.height})")
    return texture


def save_frame(ctx, width, height):
    data = ctx.screen.read(components=3, alignment=1)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image.save(OUTPUT_IMAGE_PATH)
    print(f"Saved {OUTPUT_IMAGE_PATH}")


def main():
    window = None
    background_texture = None

    try:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(WIDTH, HEIGHT, "GPU Planar Black Hole", None, None)
        if not window:
            raise RuntimeError("glfw.create_window() failed")

        glfw.make_context_current(window)
        glfw.swap_interval(1)

        ctx = moderngl.create_context()
        print("OpenGL version:", ctx.info["GL_VERSION"])

        program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        background_texture = load_background_texture(ctx)

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
        vao = ctx.simple_vertex_array(program, vbo, "in_pos")

        save_requested = {"value": False}

        def on_key(active_window, key, scancode, action, mods):
            if action != glfw.PRESS:
                return
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(active_window, True)
            elif key == glfw.KEY_S:
                save_requested["value"] = True

        glfw.set_key_callback(window, on_key)

        print("GPU planar renderer running. Press S to save a PNG, Esc to close.")
        print(
            f"{WIDTH}x{HEIGHT}, samples={SAMPLES_PER_AXIS ** 2}, "
            f"geodesic steps={GEODESIC_STEPS}"
        )

        while not glfw.window_should_close(window):
            glfw.poll_events()

            fb_width, fb_height = glfw.get_framebuffer_size(window)
            ctx.viewport = (0, 0, fb_width, fb_height)
            ctx.clear(0.0, 0.0, 0.0, 1.0)

            if background_texture is not None:
                background_texture.use(location=0)

            set_uniform(program, "iResolution", (float(fb_width), float(fb_height)))
            set_uniform(program, "camPos", tuple(float(v) for v in CAMERA_POS))
            set_uniform(program, "camTarget", tuple(float(v) for v in CAMERA_TARGET))
            set_uniform(program, "camWorldUp", tuple(float(v) for v in CAMERA_WORLD_UP))
            set_uniform(program, "fovYDeg", FOV_DEG)
            set_uniform(program, "backgroundTex", 0)
            set_uniform(program, "useBackgroundTexture", 1 if background_texture is not None else 0)
            set_uniform(program, "horizonRadius", R_HORIZON)
            set_uniform(program, "diskInner", R_DISK_IN)
            set_uniform(program, "diskOuter", R_DISK_OUT)
            set_uniform(program, "diskHalfThickness", DISK_HALF_THICKNESS)
            set_uniform(program, "escapeRadius", R_ESCAPE)
            set_uniform(program, "phiMax", float(PHI_MAX))
            set_uniform(program, "geodesicSteps", min(GEODESIC_STEPS, 4096))
            set_uniform(program, "sampleAxis", min(max(SAMPLES_PER_AXIS, 1), 3))
            set_uniform(program, "exposure", EXPOSURE)

            vao.render(moderngl.TRIANGLE_STRIP)

            if save_requested["value"]:
                save_frame(ctx, fb_width, fb_height)
                save_requested["value"] = False

            glfw.swap_buffers(window)
    except Exception as exc:
        print("GPUPlanarBlackHoleVisual failed:")
        print(exc)
        input("Press Enter to close...")
    finally:
        if background_texture is not None:
            background_texture.release()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
