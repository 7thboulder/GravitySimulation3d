import numpy as np


WIDTH = 1280
HEIGHT = 720
FOV_DEG = 45.0

CAMERA_POS = np.array([0.0, -32.0, 9.0], dtype=float)
CAMERA_TARGET = np.array([0.0, 0.0, 0.0], dtype=float)

R_HORIZON = 2.0
R_CAPTURE_EPSILON = 0.005
R_CAPTURE_FALLBACK = 2.02
R_DISK_IN = 6.0
R_DISK_OUT = 16.0
R_ESCAPE = 180.0
R_ESCAPE_FALLBACK = 150.0

STEP_SIZE = 0.02
MAX_STEPS = 10000


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


def cartesian_to_schwarzschild(pos):
    r = np.linalg.norm(pos)
    theta = np.arccos(np.clip(pos[2] / r, -1.0, 1.0))
    phi = np.arctan2(pos[1], pos[0])
    return r, theta, phi


def schwarzschild_to_cartesian(r, theta, phi):
    sin_theta = np.sin(theta)
    return np.array(
        [
            r * sin_theta * np.cos(phi),
            r * sin_theta * np.sin(phi),
            r * np.cos(theta),
        ],
        dtype=float,
    )


def cartesian_to_covariant_momentum(pos, direction):
    x, y, z = pos
    dx, dy, dz = direction

    r = np.linalg.norm(pos)
    rho = max(np.linalg.norm(pos[:2]), 1e-12)
    f = 1.0 - 2.0 / r

    p_r_contra = (x * dx + y * dy + z * dz) / r
    p_theta_contra = (z * (x * dx + y * dy) - rho ** 2 * dz) / (r ** 2 * rho)
    p_phi_contra = (x * dy - y * dx) / (rho ** 2)

    p_r = p_r_contra / max(f, 1e-12)
    p_theta = p_theta_contra * r ** 2
    p_phi = p_phi_contra * rho ** 2
    return p_r, p_theta, p_phi


def null_pt(r, theta, p_r, p_theta, p_phi):
    sin_theta = max(np.sin(theta), 1e-12)
    f = 1.0 - 2.0 / r
    term = (
        f * f * p_r ** 2
        + f * p_theta ** 2 / (r ** 2)
        + f * p_phi ** 2 / (r ** 2 * sin_theta ** 2)
    )
    return -np.sqrt(max(term, 0.0))


def init_state(ray_origin, ray_dir):
    r, theta, phi = cartesian_to_schwarzschild(ray_origin)
    p_r, p_theta, p_phi = cartesian_to_covariant_momentum(ray_origin, ray_dir)
    p_t = null_pt(r, theta, p_r, p_theta, p_phi)

    x = np.array([0.0, r, theta, phi], dtype=float)
    p = np.array([p_t, p_r, p_theta, p_phi], dtype=float)
    return x, p


def deriv(x, p):
    _, r, theta, _ = x
    p_t, p_r, p_theta, p_phi = p

    sin_theta = max(np.sin(theta), 1e-12)
    cos_theta = np.cos(theta)
    f = 1.0 - 2.0 / r
    f_safe = max(f, 1e-12)

    dx = np.array(
        [
            -p_t / f_safe,
            f * p_r,
            p_theta / (r ** 2),
            p_phi / (r ** 2 * sin_theta ** 2),
        ],
        dtype=float,
    )

    dp = np.array(
        [
            0.0,
            -(p_t ** 2) / (r ** 2 * f_safe ** 2)
            - (p_r ** 2) / (r ** 2)
            + (p_theta ** 2) / (r ** 3)
            + (p_phi ** 2) / (r ** 3 * sin_theta ** 2),
            cos_theta * p_phi ** 2 / (r ** 2 * sin_theta ** 3),
            0.0,
        ],
        dtype=float,
    )

    return dx, dp


def rk4_step(x, p, h):
    k1x, k1p = deriv(x, p)
    k2x, k2p = deriv(x + 0.5 * h * k1x, p + 0.5 * h * k1p)
    k3x, k3p = deriv(x + 0.5 * h * k2x, p + 0.5 * h * k2p)
    k4x, k4p = deriv(x + h * k3x, p + h * k3p)

    x_next = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    p_next = p + (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
    return x_next, p_next


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


def trace_custom_ray(ray_origin, ray_dir):
    x, p = init_state(ray_origin, ray_dir)

    prev_pos = schwarzschild_to_cartesian(x[1], x[2], x[3])
    positions = [prev_pos.copy()]

    r_min = x[1]
    r_max = x[1]
    disk_hit = False
    disk_radius = None
    fate = "incomplete"

    for _ in range(MAX_STEPS):
        x, p = rk4_step(x, p, STEP_SIZE)
        r = x[1]

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)) or not np.isfinite(r) or r <= 0.0:
            fate = "invalid"
            break

        r_min = min(r_min, r)
        r_max = max(r_max, r)
        pos = schwarzschild_to_cartesian(x[1], x[2], x[3])
        positions.append(pos.copy())

        hit = segment_disk_intersection(prev_pos, pos)
        if hit is not None and not disk_hit:
            disk_hit = True
            disk_radius = np.linalg.norm(hit)

        if r <= R_HORIZON + R_CAPTURE_EPSILON:
            fate = "captured"
            break

        if r >= R_ESCAPE:
            fate = "escaped"
            break

        prev_pos = pos

    positions = np.array(positions, dtype=float)
    r_final = float(np.linalg.norm(positions[-1]))

    if fate == "invalid" and r_min <= R_CAPTURE_FALLBACK:
        fate = "captured"
    elif fate == "incomplete" and r_final >= R_ESCAPE_FALLBACK and r_max >= R_ESCAPE_FALLBACK:
        fate = "escaped"

    return {
        "fate": fate,
        "disk_hit": disk_hit,
        "disk_radius": disk_radius,
        "point_count": len(positions),
        "r_min": float(r_min),
        "r_final": r_final,
        "final_point": positions[-1],
    }


def main():
    forward, right, up = make_camera_basis(CAMERA_POS, CAMERA_TARGET)

    cx = WIDTH // 2
    cy = HEIGHT // 2
    sample_pixels = [
        (cx, cy),
        (cx, cy - 40),
        (cx, cy + 40),
        (cx, cy - 80),
        (cx, cy + 80),
        (cx, cy - 120),
        (cx, cy + 120),
        (cx - 80, cy),
        (cx + 80, cy),
        (cx - 160, cy),
        (cx + 160, cy),
        (cx - 240, cy),
        (cx + 240, cy),
    ]

    for pixel_i, pixel_j in sample_pixels:
        ray_dir = pixel_to_ray(pixel_i, pixel_j, WIDTH, HEIGHT, FOV_DEG, forward, right, up)
        result = trace_custom_ray(CAMERA_POS, ray_dir)

        print("-" * 60)
        print(f"pixel: {(pixel_i, pixel_j)}")
        print(f"ray_dir: {np.array2string(ray_dir, precision=6)}")
        print(f"fate: {result['fate']}")
        print(f"disk_hit: {result['disk_hit']}")
        print(f"disk_radius: {result['disk_radius']}")
        print(f"point_count: {result['point_count']}")
        print(f"r_min: {result['r_min']}")
        print(f"r_final: {result['r_final']}")
        print(f"final_point: {result['final_point']}")


if __name__ == "__main__":
    main()
