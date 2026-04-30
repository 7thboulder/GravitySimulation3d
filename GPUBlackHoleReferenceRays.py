import numpy as np

from SchwarzschildBlackHoleSimulation import BlackHoleSimulation


WIDTH = 1280
HEIGHT = 720
FOV_DEG = 45.0

CAMERA_POS = np.array([0.0, -32.0, 9.0], dtype=float)
CAMERA_TARGET = np.array([0.0, 0.0, 0.0], dtype=float)

R_HORIZON = 2.0
R_CAPTURE_EPSILON = 0.005
R_DISK_IN = 6.0
R_DISK_OUT = 16.0
R_ESCAPE = 180.0

GEOD_STEPS = 10000
GEOD_DELTA = 0.01


class ReferenceSimulation(BlackHoleSimulation):
    def null_pt(self, r, theta, p_r, p_theta, p_phi):
        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-12:
            sin_theta = 1e-12

        f = 1.0 - 2.0 / r
        term = (
            f * p_r ** 2
            + (p_theta ** 2) / (r ** 2)
            + (p_phi ** 2) / (r ** 2 * sin_theta ** 2)
        )
        return -np.sqrt(max(term, 0.0))


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


def extract_positions_M_and_fate(geod):
    _, vecs = geod.trajectory
    pos = vecs[:, 1:4]
    r = np.linalg.norm(pos, axis=1)
    r_min = float(np.min(r))
    r_final = float(r[-1])

    if r_min <= R_HORIZON + R_CAPTURE_EPSILON:
        fate = "captured"
    elif r_final >= R_ESCAPE:
        fate = "escaped"
    else:
        fate = "incomplete"

    outside = r > R_HORIZON
    if not outside.any():
        return None, fate, r_min, r_final

    if fate == "captured":
        last_out = np.where(outside)[0][-1]
        if last_out + 1 < len(r):
            t = (R_HORIZON - r[last_out]) / (r[last_out + 1] - r[last_out])
            crossing = pos[last_out] + t * (pos[last_out + 1] - pos[last_out])
            pos = np.vstack([pos[:last_out + 1], crossing])
        else:
            pos = pos[outside]
    else:
        pos = pos[outside]

    if len(pos) < 2:
        return None, fate, r_min, r_final

    return pos, fate, r_min, r_final


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


def trace_reference_ray(sim, pixel_i, pixel_j, forward, right, up):
    ray_dir = pixel_to_ray(pixel_i, pixel_j, WIDTH, HEIGHT, FOV_DEG, forward, right, up)

    geod = sim.calculate_null_geodesic(
        CAMERA_POS[0],
        CAMERA_POS[1],
        CAMERA_POS[2],
        ray_dir[0],
        ray_dir[1],
        ray_dir[2],
        steps=GEOD_STEPS,
        delta=GEOD_DELTA,
    )

    points, fate, r_min, r_final = extract_positions_M_and_fate(geod)
    disk_hit = False
    disk_radius = None

    if points is not None:
        for k in range(len(points) - 1):
            hit = segment_disk_intersection(points[k], points[k + 1])
            if hit is not None:
                disk_hit = True
                disk_radius = np.linalg.norm(hit)
                break

    return {
        "pixel": (pixel_i, pixel_j),
        "ray_dir": ray_dir,
        "fate": fate,
        "disk_hit": disk_hit,
        "disk_radius": disk_radius,
        "point_count": 0 if points is None else len(points),
        "r_min": r_min,
        "r_final": r_final,
        "final_point": None if points is None else points[-1],
    }


def main():
    sim = ReferenceSimulation(
        central_mass=1.0e30,
        central_color="black",
        central_name="Black Hole",
        dt=1.0,
    )

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
        result = trace_reference_ray(sim, pixel_i, pixel_j, forward, right, up)
        print("-" * 60)
        print(f"pixel: {result['pixel']}")
        print(f"ray_dir: {np.array2string(result['ray_dir'], precision=6)}")
        print(f"fate: {result['fate']}")
        print(f"disk_hit: {result['disk_hit']}")
        print(f"disk_radius: {result['disk_radius']}")
        print(f"point_count: {result['point_count']}")
        print(f"r_min: {result['r_min']}")
        print(f"r_final: {result['r_final']}")
        print(f"final_point: {result['final_point']}")


if __name__ == "__main__":
    main()
