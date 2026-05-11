import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
from pathlib import Path
from PIL import Image


QUALITY_MODE = "preview"  # "preview" or "final"

WIDTH = 800
HEIGHT = 640
FOV_DEG = 45.0
SAMPLES_PER_AXIS = 3
USE_MULTIPROCESSING = True
MAX_PROCESSES = 12
BLOOM_THRESHOLD = 0.62
BLOOM_SIGMA = 2.2
BLOOM_STRENGTH = 0.65

#CAMERA_POS = np.array([0.0, 2.5, 2.5], dtype=float)
CAMERA_POS = np.array([0.0, -32.0, 3.0], dtype=float)
CAMERA_TARGET = np.array([0.0, 0.0, 3], dtype=float)

R_HORIZON = 2.0
R_DISK_IN = 6.0
R_DISK_OUT = 16.0
DISK_HALF_THICKNESS = 0.42
R_ESCAPE = 180.0
BACKGROUND_IMAGE_PATH = Path("space_background.png")
DISK_VOLUME_STEPS = 8

PHI_MAX = 24.0 * np.pi
N_SAMPLES = 3000
WORKER_FORWARD = None
WORKER_RIGHT = None
WORKER_UP = None
BACKGROUND_IMAGE = None

if QUALITY_MODE == "preview":
    WIDTH = 360
    HEIGHT = 288
    SAMPLES_PER_AXIS = 1
    N_SAMPLES = 1200
    MAX_PROCESSES = 4
elif QUALITY_MODE == "final":
    WIDTH = 800
    HEIGHT = 640
    SAMPLES_PER_AXIS = 3
    N_SAMPLES = 3000
    MAX_PROCESSES = 12


def make_camera_basis(pos, target):
    forward = target - pos
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-12:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return forward, right, up


def pixel_to_ray(i, j, width, height, fov_deg, forward, right, up):
    fov = np.radians(fov_deg)
    aspect = width / height

    x = (2.0 * ((i + 0.5) / width) - 1.0) * np.tan(fov / 2.0) * aspect
    y = (1.0 - 2.0 * ((j + 0.5) / height)) * np.tan(fov / 2.0)

    direction = forward + x * right + y * up
    direction /= np.linalg.norm(direction)
    return direction


def subpixel_to_ray(i, j, sx, sy, width, height, fov_deg, forward, right, up):
    fov = np.radians(fov_deg)
    aspect = width / height

    x = (2.0 * ((i + sx) / width) - 1.0) * np.tan(fov / 2.0) * aspect
    y = (1.0 - 2.0 * ((j + sy) / height)) * np.tan(fov / 2.0)

    direction = forward + x * right + y * up
    direction /= np.linalg.norm(direction)
    return direction


def make_orbital_plane(ray_origin, ray_dir):
    r0 = np.linalg.norm(ray_origin)
    e1 = ray_origin / r0

    normal = np.cross(ray_origin, ray_dir)
    norm_n = np.linalg.norm(normal)
    if norm_n < 1e-12:
        trial = np.cross(ray_origin, np.array([0.0, 0.0, 1.0], dtype=float))
        if np.linalg.norm(trial) < 1e-12:
            trial = np.cross(ray_origin, np.array([1.0, 0.0, 0.0], dtype=float))
        normal = trial
        norm_n = np.linalg.norm(normal)

    e3 = normal / norm_n
    e2 = np.cross(e3, e1)
    e2 /= np.linalg.norm(e2)

    radial_component = np.dot(ray_dir, e1)
    tangential_component = np.dot(ray_dir, e2)

    # Force phi to increase monotonically in the chosen plane basis.
    if tangential_component < 0.0:
        e2 = -e2
        e3 = -e3
        tangential_component = -tangential_component

    return r0, e1, e2, e3, radial_component, tangential_component


def trace_planar_geodesic(ray_origin, ray_dir):
    r0, e1, e2, e3, da, db = make_orbital_plane(ray_origin, ray_dir)

    if db < 1e-8:
        if da < 0.0:
            return None, "captured"
        return None, "escaped"

    u0 = 1.0 / r0
    du0 = -da / (r0 * db)

    def orbit_odes(phi, state):
        u, du = state
        return [du, 3.0 * u * u - u]

    def event_captured(phi, state):
        return state[0] - 0.5

    event_captured.terminal = True
    event_captured.direction = 1

    def event_escaped(phi, state):
        return state[0] - 1.0 / R_ESCAPE

    event_escaped.terminal = True
    event_escaped.direction = -1

    sol = solve_ivp(
        orbit_odes,
        t_span=(0.0, PHI_MAX),
        y0=[u0, du0],
        method="DOP853",
        events=[event_captured, event_escaped],
        rtol=1e-10,
        atol=1e-10,
        dense_output=True,
    )

    phi_end = sol.t[-1]
    phis = np.linspace(0.0, phi_end, N_SAMPLES)
    u_arr = sol.sol(phis)[0]

    valid = np.isfinite(u_arr) & (u_arr > 0.0)
    u_arr = u_arr[valid]
    phis = phis[valid]

    if len(u_arr) < 2:
        return None, "incomplete"

    outside = u_arr < 0.5
    u_arr = u_arr[outside]
    phis = phis[outside]
    if len(u_arr) < 2:
        return None, "captured"

    r_arr = 1.0 / u_arr
    positions = (
        (r_arr * np.cos(phis))[:, None] * e1[None, :]
        + (r_arr * np.sin(phis))[:, None] * e2[None, :]
    )

    if sol.t_events[0].size > 0:
        fate = "captured"
    elif sol.t_events[1].size > 0:
        fate = "escaped"
    else:
        fate = "incomplete"

    return positions, fate


def disk_half_thickness_at_radius(rho):
    t = np.clip((rho - R_DISK_IN) / (R_DISK_OUT - R_DISK_IN), 0.0, 1.0)
    # Slightly puffed inner disk, gently tapering outward.
    return DISK_HALF_THICKNESS * (1.12 - 0.52 * t + 0.08 * np.sin(np.pi * t))


def segment_disk_intersection(p0, p1):

    dz = p1[2] - p0[2]
    hit_candidates = []

    # Sample along the segment and find where we enter the tapered disk volume.
    sample_count = 10
    previous_inside = False
    previous_t = 0.0
    previous_point = p0

    for idx in range(sample_count + 1):
        t = idx / sample_count
        point = p0 + t * (p1 - p0)
        rho = np.hypot(point[0], point[1])
        if R_DISK_IN <= rho <= R_DISK_OUT:
            local_half_thickness = disk_half_thickness_at_radius(rho)
            inside = abs(point[2]) <= local_half_thickness
        else:
            inside = False

        if inside and not previous_inside:
            hit_candidates.append([previous_t, None, previous_point, point])
        elif previous_inside and not inside and hit_candidates and hit_candidates[-1][1] is None:
            hit_candidates[-1][1] = t
        previous_inside = inside
        previous_t = t
        previous_point = point

    if not hit_candidates:
        return None

    if hit_candidates[0][1] is None:
        hit_candidates[0][1] = 1.0

    t0, t1, q0, q1 = hit_candidates[0]
    t_mid = 0.5 * (t0 + t1)
    hit = p0 + t_mid * (p1 - p0)
    rho = np.hypot(hit[0], hit[1])
    if not (R_DISK_IN <= rho <= R_DISK_OUT):
        return None

    local_half_thickness = disk_half_thickness_at_radius(rho)
    path_length = np.linalg.norm((t1 - t0) * (p1 - p0))
    z_blend = np.clip(1.0 - abs(hit[2]) / max(local_half_thickness, 1e-12), 0.0, 1.0)
    surface = "top" if hit[2] >= 0.0 else "bottom"
    thickness_mix = np.clip(path_length / (2.0 * local_half_thickness + 1e-12), 0.0, 1.0)
    radial_taper = np.clip(local_half_thickness / max(DISK_HALF_THICKNESS, 1e-12), 0.0, 1.5)
    return hit, surface, thickness_mix, z_blend, radial_taper, t0, t1


def disk_tangent_velocity(hit_point):
    x, y, _ = hit_point
    rho = np.hypot(x, y)
    if rho < 1e-12:
        return np.zeros(3, dtype=float)

    tangent = np.array([-y / rho, x / rho, 0.0], dtype=float)

    # Approximate relativistic disk speed profile in geometric units.
    beta = np.sqrt(np.clip(1.0 / max(rho - 2.0, 1e-6), 0.0, 0.45))
    return beta * tangent


def disk_texture(hit_point):
    x, y, _ = hit_point
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)

    log_r = np.log(max(rho, 1e-6))

    spiral_1 = np.sin(5.0 * phi - 8.0 * log_r)
    spiral_2 = np.sin(9.0 * phi - 12.0 * log_r + 0.9)
    spiral_3 = np.sin(15.0 * phi - 5.0 * rho + 0.6)

    clump_1 = np.sin(23.0 * phi + 8.0 * rho)
    clump_2 = np.sin(37.0 * phi - 11.0 * rho + 0.4)

    large_scale = 1.0 + 0.10 * spiral_1 + 0.07 * spiral_2 + 0.04 * spiral_3
    fine_scale = 1.0 + 0.025 * clump_1 + 0.018 * clump_2

    hotspot_1 = np.exp(-((phi - 0.7 - 0.55 * np.log(max(rho / R_DISK_IN, 1e-6))) / 0.22) ** 2)
    hotspot_2 = np.exp(-((phi + 1.9 - 0.35 * np.log(max(rho / R_DISK_IN, 1e-6))) / 0.18) ** 2)
    shear_1 = 0.5 + 0.5 * np.sin(26.0 * phi - 19.0 * np.log(max(rho, 1e-6)))
    shear_2 = 0.5 + 0.5 * np.sin(41.0 * phi - 11.0 * rho + 0.8)

    inner_rim_boost = 1.0 + 0.22 * np.exp(-((rho - R_DISK_IN) / 1.1) ** 2) * (0.5 + 0.5 * np.sin(12.0 * phi))
    hotspot_boost = 1.0 + 0.16 * hotspot_1 + 0.12 * hotspot_2
    shear_boost = 1.0 + 0.08 * shear_1 + 0.05 * shear_2

    return np.clip(large_scale * fine_scale * inner_rim_boost * hotspot_boost * shear_boost, 0.76, 1.42)


def disk_color(hit_point, outgoing_dir, surface, thickness_mix, z_blend, radial_taper):
    rho = np.hypot(hit_point[0], hit_point[1])
    t = np.clip((rho - R_DISK_IN) / (R_DISK_OUT - R_DISK_IN), 0.0, 1.0)

    inner = np.array([1.0, 0.985, 0.93], dtype=float)
    mid = np.array([1.0, 0.58, 0.18], dtype=float)
    outer = np.array([0.72, 0.13, 0.02], dtype=float)
    if t < 0.45:
        blend = t / 0.45
        color = (1.0 - blend) * inner + blend * mid
    else:
        blend = (t - 0.45) / 0.55
        color = (1.0 - blend) * mid + blend * outer

    brightness = 4.0 / (rho ** 0.92)
    glow = 0.18 / max(rho - R_HORIZON, 0.35)
    inner_sharpen = 1.0 + 0.45 * np.exp(-((rho - R_DISK_IN) / 0.75) ** 2)

    velocity = disk_tangent_velocity(hit_point)
    beta = np.linalg.norm(velocity)
    if beta > 1e-8:
        view_cos = np.clip(np.dot(velocity / beta, -outgoing_dir), -1.0, 1.0)
    else:
        view_cos = 0.0

    gamma = 1.0 / np.sqrt(max(1.0 - beta ** 2, 1e-6))
    doppler = 1.0 / (gamma * max(1.0 - beta * view_cos, 1e-6))
    grav_shift = np.sqrt(max(1.0 - 2.0 / max(rho, R_HORIZON + 1e-6), 1e-6))

    intensity_boost = doppler ** 3
    net_shift = doppler * grav_shift

    if net_shift >= 1.0:
        shift_strength = min(net_shift - 1.0, 0.5)
        color = (1.0 - shift_strength) * color + shift_strength * np.array([1.0, 0.97, 0.9], dtype=float)
    else:
        shift_strength = min(1.0 - net_shift, 0.7)
        color = (1.0 - shift_strength) * color + shift_strength * np.array([0.75, 0.16, 0.03], dtype=float)

    # Slightly favor the upper surface and dim the underside.
    surface_factor = 1.04 if surface == "top" else 0.88
    texture_factor = disk_texture(hit_point)
    thickness_factor = (0.92 + 0.12 * thickness_mix) * (0.94 + 0.12 * radial_taper)
    view_angle = np.clip(abs(outgoing_dir[2]), 0.0, 1.0)
    optical_depth = 0.55 + 1.4 * np.exp(-((rho - R_DISK_IN) / 2.8) ** 2) + 0.35 * texture_factor
    transmission = np.exp(-optical_depth * (0.35 + 1.1 * (1.0 - view_angle)) * (0.8 + 0.4 * thickness_mix))
    opacity_factor = 0.45 + 0.55 * (1.0 - transmission)

    # Feather the top/bottom slab boundary so the disk edge is less hard.
    edge_softening = 0.72 + 0.28 * (z_blend ** 0.6)
    return np.clip(
        color
        * (brightness + glow)
        * inner_sharpen
        * intensity_boost
        * surface_factor
        * texture_factor
        * thickness_factor
        * edge_softening
        * opacity_factor,
        0.0,
        6.0,
    )


def accumulate_disk_volume(p0, p1, ray_dir, t0, t1):
    segment = p1 - p0
    segment_len = np.linalg.norm(segment)
    if segment_len < 1e-12:
        return None

    accum = np.zeros(3, dtype=float)
    transmittance = 1.0

    for step in range(DISK_VOLUME_STEPS):
        s = (step + 0.5) / DISK_VOLUME_STEPS
        t = (1.0 - s) * t0 + s * t1
        point = p0 + t * segment

        rho = np.hypot(point[0], point[1])
        if not (R_DISK_IN <= rho <= R_DISK_OUT):
            continue

        local_half_thickness = disk_half_thickness_at_radius(rho)
        z_blend = np.clip(1.0 - abs(point[2]) / max(local_half_thickness, 1e-12), 0.0, 1.0)
        if z_blend <= 0.0:
            continue

        surface = "top" if point[2] >= 0.0 else "bottom"
        radial_taper = np.clip(local_half_thickness / max(DISK_HALF_THICKNESS, 1e-12), 0.0, 1.5)
        thickness_mix = np.clip((t1 - t0) * segment_len / (2.0 * local_half_thickness + 1e-12), 0.0, 1.0)

        base_color = disk_color(point, ray_dir, surface, thickness_mix, z_blend, radial_taper)
        density = (0.22 + 0.95 * z_blend ** 1.6) * (1.15 - 0.35 * np.clip((rho - R_DISK_IN) / (R_DISK_OUT - R_DISK_IN), 0.0, 1.0))
        optical_depth = density * segment_len * (t1 - t0) / DISK_VOLUME_STEPS * 1.15
        absorption = 1.0 - np.exp(-optical_depth)

        accum += transmittance * base_color * absorption
        transmittance *= np.exp(-optical_depth * 1.25)

        if transmittance < 0.02:
            break

    return np.clip(accum, 0.0, 6.0)


def background_color(direction):
    if BACKGROUND_IMAGE is not None:
        return sample_background_image(direction)

    return procedural_background_color(direction)


def sample_background_image(direction):
    phi = np.arctan2(direction[1], direction[0])
    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))

    u = (phi / (2.0 * np.pi) + 0.5) % 1.0
    v = np.clip(theta / np.pi, 0.0, 1.0)

    h, w = BACKGROUND_IMAGE.shape[:2]
    x = u * (w - 1)
    y = v * (h - 1)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    tx = x - x0
    ty = y - y0

    c00 = BACKGROUND_IMAGE[y0, x0]
    c10 = BACKGROUND_IMAGE[y0, x1]
    c01 = BACKGROUND_IMAGE[y1, x0]
    c11 = BACKGROUND_IMAGE[y1, x1]

    c0 = (1.0 - tx) * c00 + tx * c10
    c1 = (1.0 - tx) * c01 + tx * c11
    return np.clip((1.0 - ty) * c0 + ty * c1, 0.0, 3.5)


def procedural_background_color(direction):
    u = 0.5 * (direction[2] + 1.0)
    horizon = np.array([0.006, 0.004, 0.005], dtype=float)
    zenith = np.array([0.0006, 0.0009, 0.0025], dtype=float)
    base = (1.0 - u) * horizon + u * zenith

    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    phi = np.arctan2(direction[1], direction[0])

    # Bright galactic band to make lensing distortions easier to see.
    band_phase = phi + 0.35 * np.sin(2.0 * theta)
    band_width = np.exp(-((theta - (np.pi * 0.52 + 0.12 * np.sin(1.7 * band_phase))) / 0.12) ** 2)
    band_texture = (
        0.55
        + 0.25 * np.sin(18.0 * phi)
        + 0.18 * np.sin(47.0 * phi + 3.0 * theta)
        + 0.12 * np.sin(95.0 * phi - 11.0 * theta)
    )
    band_texture = np.clip(band_texture, 0.0, 1.2)
    galactic_band = band_width * band_texture

    a = np.sin(1234.567 * phi + 321.123 * theta)
    b = np.sin(6789.123 * phi - 987.654 * theta)
    c = np.sin(4567.891 * (phi + theta))
    d = np.sin(15317.337 * phi + 2211.731 * theta)
    e = np.sin(31111.113 * phi - 7133.219 * theta)

    star_field = (a + b + c + d + e) / 5.0
    star_core = 1.0 if star_field > 0.998 else 0.0
    star_halo = 1.0 if star_field > 0.994 else 0.0
    star_mid = 1.0 if star_field > 0.989 else 0.0

    color = base
    color += galactic_band * np.array([0.13, 0.12, 0.10], dtype=float)
    color += star_mid * np.array([0.015, 0.016, 0.02], dtype=float)
    color += star_halo * np.array([0.12, 0.13, 0.16], dtype=float)
    color += star_core * np.array([1.6, 1.5, 1.35], dtype=float)
    return np.clip(color, 0.0, 3.5)


def tone_map(color):
    mapped = color / (1.0 + color)
    return np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2)


def apply_bloom(image):
    luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    bright_mask = np.clip(luminance - BLOOM_THRESHOLD, 0.0, None)
    bright_pass = image * bright_mask[:, :, None]

    blurred = np.zeros_like(image)
    for channel in range(3):
        blurred[:, :, channel] = gaussian_filter(bright_pass[:, :, channel], sigma=BLOOM_SIGMA)

    return np.clip(image + BLOOM_STRENGTH * blurred, 0.0, 1.0)


def load_background_image():
    global BACKGROUND_IMAGE

    if not BACKGROUND_IMAGE_PATH.exists():
        BACKGROUND_IMAGE = None
        return

    image = np.asarray(Image.open(BACKGROUND_IMAGE_PATH).convert("RGB"), dtype=float)
    if image.ndim != 3:
        BACKGROUND_IMAGE = None
        return

    image /= 255.0

    BACKGROUND_IMAGE = image


def trace_ray(ray_origin, ray_dir):
    positions, fate = trace_planar_geodesic(ray_origin, ray_dir)

    if positions is None or len(positions) < 2:
        return np.zeros(3, dtype=float) if fate == "captured" else background_color(ray_dir)

    for i in range(len(positions) - 1):
        disk_hit = segment_disk_intersection(positions[i], positions[i + 1])
        if disk_hit is not None:
            hit, surface, thickness_mix, z_blend, radial_taper, t0, t1 = disk_hit
            segment_dir = positions[i + 1] - positions[i]
            norm = np.linalg.norm(segment_dir)
            outgoing_dir = ray_dir if norm < 1e-12 else segment_dir / norm
            volume_color = accumulate_disk_volume(positions[i], positions[i + 1], outgoing_dir, t0, t1)
            if volume_color is not None:
                return volume_color
            return disk_color(hit, outgoing_dir, surface, thickness_mix, z_blend, radial_taper)

    if fate == "captured":
        return np.zeros(3, dtype=float)

    final_dir = positions[-1] - positions[-2]
    norm = np.linalg.norm(final_dir)
    if norm < 1e-12:
        return np.zeros(3, dtype=float)

    return background_color(final_dir / norm)


def init_worker(forward, right, up):
    global WORKER_FORWARD, WORKER_RIGHT, WORKER_UP
    WORKER_FORWARD = np.array(forward, dtype=float)
    WORKER_RIGHT = np.array(right, dtype=float)
    WORKER_UP = np.array(up, dtype=float)
    load_background_image()


def render_row(j):
    row = np.zeros((WIDTH, 3), dtype=float)
    offsets = np.linspace(0.25, 0.75, SAMPLES_PER_AXIS)

    for i in range(WIDTH):
        color = np.zeros(3, dtype=float)
        for sy in offsets:
            for sx in offsets:
                ray_dir = subpixel_to_ray(i, j, sx, sy, WIDTH, HEIGHT, FOV_DEG, WORKER_FORWARD, WORKER_RIGHT, WORKER_UP)
                color += trace_ray(CAMERA_POS, ray_dir)
        row[i] = tone_map(color / (SAMPLES_PER_AXIS ** 2))

    return j, row


def render():
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
    forward, right, up = make_camera_basis(CAMERA_POS, CAMERA_TARGET)
    load_background_image()
    print(
        f"quality mode: {QUALITY_MODE} | {WIDTH}x{HEIGHT} | spp={SAMPLES_PER_AXIS ** 2} | orbit samples={N_SAMPLES}",
        flush=True,
    )

    if USE_MULTIPROCESSING:
        print(f"multiprocessing enabled with {MAX_PROCESSES} workers", flush=True)
        with Pool(
            processes=MAX_PROCESSES,
            initializer=init_worker,
            initargs=(forward, right, up),
        ) as pool:
            for completed_rows, (j, row) in enumerate(
                pool.imap_unordered(render_row, range(HEIGHT)),
                start=1,
            ):
                image[j] = row
                print(f"row {completed_rows}/{HEIGHT} complete", flush=True)
    else:
        init_worker(forward, right, up)
        for j in range(HEIGHT):
            print(f"row {j + 1}/{HEIGHT}", flush=True)
            _, row = render_row(j)
            image[j] = row

    return image


def main():
    image = render()
    image = apply_bloom(image)

    plt.figure(figsize=(12, 7))
    plt.imshow(image, origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.imsave("planar_black_hole_render.png", image)


if __name__ == "__main__":
    main()
