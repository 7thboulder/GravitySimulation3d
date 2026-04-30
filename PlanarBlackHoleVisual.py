import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


WIDTH = 320
HEIGHT = 180
FOV_DEG = 45.0

CAMERA_POS = np.array([0.0, -32.0, -3.0], dtype=float)
CAMERA_TARGET = np.array([0.0, 0.0, 0.0], dtype=float)

R_HORIZON = 2.0
R_DISK_IN = 6.0
R_DISK_OUT = 16.0
R_ESCAPE = 180.0

PHI_MAX = 24.0 * np.pi
N_SAMPLES = 3000


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
    z0 = p0[2]
    z1 = p1[2]

    if z0 == 0.0:
        hit = p0
    elif z1 == 0.0:
        hit = p1
    elif z0 * z1 > 0.0:
        return None
    else:
        t = -z0 / (z1 - z0)
        if t < 0.0 or t > 1.0:
            return None
        hit = p0 + t * (p1 - p0)

    r_hit = np.linalg.norm(hit)
    if R_DISK_IN <= r_hit <= R_DISK_OUT:
        return hit
    return None


def disk_color(hit_point):
    r = np.linalg.norm(hit_point)
    t = np.clip((r - R_DISK_IN) / (R_DISK_OUT - R_DISK_IN), 0.0, 1.0)

    inner = np.array([1.0, 0.96, 0.85], dtype=float)
    outer = np.array([1.0, 0.38, 0.08], dtype=float)
    color = (1.0 - t) * inner + t * outer

    brightness = 2.2 / (r ** 0.8)
    return np.clip(color * brightness, 0.0, 1.0)


def background_color(direction):
    u = 0.5 * (direction[2] + 1.0)
    base = (1.0 - u) * np.array([0.01, 0.01, 0.02]) + u * np.array([0.06, 0.05, 0.04])

    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    phi = np.arctan2(direction[1], direction[0])
    seed = np.sin(43758.5453 * (phi + 0.5 * theta))
    star = 1.0 if seed > 0.9994 else 0.0
    return np.clip(base + star * np.array([1.0, 0.95, 0.85]), 0.0, 1.0)


def trace_ray(ray_origin, ray_dir):
    positions, fate = trace_planar_geodesic(ray_origin, ray_dir)

    if positions is None or len(positions) < 2:
        return np.zeros(3, dtype=float) if fate == "captured" else background_color(ray_dir)

    for i in range(len(positions) - 1):
        hit = segment_disk_intersection(positions[i], positions[i + 1])
        if hit is not None:
            return disk_color(hit)

    if fate == "captured":
        return np.zeros(3, dtype=float)

    final_dir = positions[-1] - positions[-2]
    norm = np.linalg.norm(final_dir)
    if norm < 1e-12:
        return np.zeros(3, dtype=float)

    return background_color(final_dir / norm)


def render():
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
    forward, right, up = make_camera_basis(CAMERA_POS, CAMERA_TARGET)

    for j in range(HEIGHT):
        print(f"row {j + 1}/{HEIGHT}")
        for i in range(WIDTH):
            ray_dir = pixel_to_ray(i, j, WIDTH, HEIGHT, FOV_DEG, forward, right, up)
            image[j, i] = trace_ray(CAMERA_POS, ray_dir)

    return image


def main():
    image = render()

    plt.figure(figsize=(12, 7))
    plt.imshow(image, origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.imsave("planar_black_hole_render.png", image)


if __name__ == "__main__":
    main()
