import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
from pathlib import Path
from PIL import Image


WIDTH = 800
HEIGHT = 640
FOV_DEG = 45.0
SAMPLES_PER_AXIS = 2
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
DISK_HALF_THICKNESS = 0.35
R_ESCAPE = 180.0
BACKGROUND_IMAGE_PATH = Path("space_background.png")

PHI_MAX = 24.0 * np.pi
N_SAMPLES = 3000
WORKER_FORWARD = None
WORKER_RIGHT = None
WORKER_UP = None
BACKGROUND_IMAGE = None


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


def segment_disk_intersection(p0, p1):
    dz = p1[2] - p0[2]

    # Find the t-interval where the segment is inside the z-slab
    if abs(dz) > 1e-12:
        t_top = (DISK_HALF_THICKNESS - p0[2]) / dz
        t_bot = (-DISK_HALF_THICKNESS - p0[2]) / dz
        t_enter = max(0.0, min(t_top, t_bot))
        t_exit  = min(1.0, max(t_top, t_bot))
    else:
        # Segment is parallel to disk plane
        if abs(p0[2]) > DISK_HALF_THICKNESS:
            return None  # entirely outside the slab
        t_enter, t_exit = 0.0, 1.0

    if t_enter >= t_exit:
        return None  # segment doesn't cross the slab

    # Use the entry point as the hit
    hit = p0 + t_enter * (p1 - p0)
    rho = np.hypot(hit[0], hit[1])
    if not (R_DISK_IN <= rho <= R_DISK_OUT):
        return None

    surface = "top" if p0[2] > 0.0 else "bottom"
    return hit, surface


def disk_tangent_velocity(hit_point):
    x, y, _ = hit_point
    rho = np.hypot(x, y)
    if rho < 1e-12:
        return np.zeros(3, dtype=float)

    tangent = np.array([-y / rho, x / rho, 0.0], dtype=float)

    # Approximate relativistic disk speed profile in geometric units.
    beta = np.sqrt(np.clip(1.0 / max(rho - 2.0, 1e-6), 0.0, 0.45))
    return beta * tangent


def disk_color(hit_point, outgoing_dir, surface):
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
    surface_factor = 1.06 if surface == "top" else 0.82
    return np.clip(color * (brightness + glow) * intensity_boost * surface_factor, 0.0, 6.0)


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
            hit, surface = disk_hit
            segment_dir = positions[i + 1] - positions[i]
            norm = np.linalg.norm(segment_dir)
            outgoing_dir = ray_dir if norm < 1e-12 else segment_dir / norm
            return disk_color(hit, outgoing_dir, surface)

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

    if USE_MULTIPROCESSING:
        print(f"multiprocessing enabled with {MAX_PROCESSES} workers")
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
                print(f"row {completed_rows}/{HEIGHT} complete")
    else:
        init_worker(forward, right, up)
        for j in range(HEIGHT):
            print(f"row {j + 1}/{HEIGHT}")
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
