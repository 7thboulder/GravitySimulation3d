import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from SchwarzschildBlackHoleSimulation import BlackHoleSimulation


WIDTH = 64
HEIGHT = 64
FOV_DEG = 45.0

CAMERA_POS = np.array([0.0, -25.0, 8.0], dtype=float)
CAMERA_TARGET = np.array([0.0, 0.0, 0.0], dtype=float)

R_HORIZON = 2.0
R_DISK_IN = 6.0
R_DISK_OUT = 20.0
R_ESCAPE = 120.0

GEOD_STEPS = 800
GEOD_DELTA = 0.02

CENTRAL_MASS_KG = 1.0e30
USE_MULTIPROCESSING = True
MAX_PROCESSES = 2
WORKER_SIM = None
WORKER_FORWARD = None
WORKER_RIGHT = None
WORKER_UP = None


class RaytraceSimulation(BlackHoleSimulation):
    def null_pt(self, r, theta, p_r, p_theta, p_phi):
        """
        Patch the missing helper used by calculate_null_geodesic().
        """
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
    """
    Extract Cartesian positions in M-units from the EinsteinPy trajectory.
    """
    _, vecs = geod.trajectory
    pos = vecs[:, 1:4]
    r = np.linalg.norm(pos, axis=1)

    if r[-1] <= R_HORIZON:
        fate = "captured"
    elif r[-1] >= R_ESCAPE:
        fate = "escaped"
    else:
        fate = "incomplete"

    outside = r > R_HORIZON
    if not outside.any():
        return None, fate

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
        return None, fate

    return pos, fate


def segment_disk_intersection(p0, p1):
    """
    Intersect a trajectory segment with the equatorial disk plane z=0.
    """
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

    inner = np.array([1.0, 0.95, 0.85], dtype=float)
    outer = np.array([1.0, 0.35, 0.05], dtype=float)
    color = (1.0 - t) * inner + t * outer

    brightness = 2.0 / (r ** 0.8)
    return np.clip(color * brightness, 0.0, 1.0)


def background_color(direction):
    u = 0.5 * (direction[2] + 1.0)
    low = np.array([0.0, 0.0, 0.0], dtype=float)
    high = np.array([0.03, 0.03, 0.07], dtype=float)
    return (1.0 - u) * low + u * high


def trace_ray(sim, ray_origin, ray_dir):
    geod = sim.calculate_null_geodesic(
        ray_origin[0],
        ray_origin[1],
        ray_origin[2],
        ray_dir[0],
        ray_dir[1],
        ray_dir[2],
        steps=GEOD_STEPS,
        delta=GEOD_DELTA,
    )

    points, fate = extract_positions_M_and_fate(geod)
    if points is None:
        return np.array([1.0, 0.0, 1.0], dtype=float)

    for k in range(len(points) - 1):
        hit = segment_disk_intersection(points[k], points[k + 1])
        if hit is not None:
            return disk_color(hit)

    if fate == "captured":
        return np.array([0.0, 0.0, 0.0], dtype=float)

    final_dir = points[-1] - points[-2]
    norm = np.linalg.norm(final_dir)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    final_dir /= norm
    return background_color(final_dir)


def init_worker(forward, right, up):
    global WORKER_SIM, WORKER_FORWARD, WORKER_RIGHT, WORKER_UP

    WORKER_SIM = RaytraceSimulation(
        central_mass=CENTRAL_MASS_KG,
        central_color="black",
        central_name="Black Hole",
        dt=1.0,
    )
    WORKER_FORWARD = np.array(forward, dtype=float)
    WORKER_RIGHT = np.array(right, dtype=float)
    WORKER_UP = np.array(up, dtype=float)


def render_row(row_index):
    row = np.zeros((WIDTH, 3), dtype=float)

    for i in range(WIDTH):
        ray_dir = pixel_to_ray(
            i,
            row_index,
            WIDTH,
            HEIGHT,
            FOV_DEG,
            WORKER_FORWARD,
            WORKER_RIGHT,
            WORKER_UP,
        )
        row[i] = trace_ray(WORKER_SIM, CAMERA_POS, ray_dir)

    return row_index, row


def render():
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
    forward, right, up = make_camera_basis(CAMERA_POS, CAMERA_TARGET)
    total_geodesics = WIDTH * HEIGHT
    completed_geodesics = 0

    if USE_MULTIPROCESSING:
        print(f"multiprocessing enabled with {MAX_PROCESSES} workers")
        with Pool(
            processes=MAX_PROCESSES,
            initializer=init_worker,
            initargs=(forward, right, up),
        ) as pool:
            for completed_rows, (row_index, row) in enumerate(
                pool.imap_unordered(render_row, range(HEIGHT)),
                start=1,
            ):
                image[row_index] = row
                completed_geodesics += WIDTH
                remaining_geodesics = total_geodesics - completed_geodesics
                print(f"row {completed_rows}/{HEIGHT} complete")
                print(f"geodesics left: {remaining_geodesics}")
    else:
        print("multiprocessing disabled; using single process")
        sim = RaytraceSimulation(
            central_mass=CENTRAL_MASS_KG,
            central_color="black",
            central_name="Black Hole",
            dt=1.0,
        )

        for j in range(HEIGHT):
            print(f"row {j + 1}/{HEIGHT}")
            for i in range(WIDTH):
                ray_dir = pixel_to_ray(i, j, WIDTH, HEIGHT, FOV_DEG, forward, right, up)
                image[j, i] = trace_ray(sim, CAMERA_POS, ray_dir)
                completed_geodesics += 1
                remaining_geodesics = total_geodesics - completed_geodesics
                print(f"geodesics left: {remaining_geodesics}")

    return image


def main():
    image = render()

    plt.figure(figsize=(8, 8))
    plt.imshow(image, origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.imsave("black_hole_render.png", image)


if __name__ == "__main__":
    main()
