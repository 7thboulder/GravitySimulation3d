import numpy as np
import pyvista as pv
from einsteinpy.plotting import GeodesicPlotter
from numpy.f2py.auxfuncs import throw_error
from einsteinpy.metric import Schwarzschild
from einsteinpy.geodesic import Geodesic, Timelike, Nulllike
import astropy.units as u
import matplotlib as mp
import matplotlib.pyplot as plt

G = 6.6743e-11
C = 299792458
ISCO = 6
rs = 2

def r_to_meters(r_geom, M_kg):
    return r_geom * G * M_kg / C ** 2


# r = radial coordinate in M-units (r=40 means 40 gravitational radii)
# θ = polar angle. Always set to pi/2 for equatorial orbits
# ϕ = azimuthal angle. Set to 0 as reference

# p_r = radial momentum. 0 if starting at perihelion/aphelion
# p_θ = polar momentum. 0 for equatorial (θ = pi/2, stays there)
# p_ϕ angular momentum L. This is what is tuned to set orbit shape

# For a CIRCULAR orbit at radius r (in M-units):
# L_circular = r / sqrt(r - 3)
# (exact GR result — diverges at r=3, the photon sphere)
#
# For an ELLIPTICAL orbit:
# Start at perihelion with p_r = 0.
# L > L_circular → aphelion further out (more eccentric)
# L < L_circular → orbit spirals inward (unstable if r < ISCO)
#
# For ESCAPE trajectory:
# Set p_r > 0 (moving outward) and L to whatever azimuthal kick you want.

def circular_L(r):
    """
    Angular momentum for a circular orbit at r in M-units
    Returns nan if r <= 3 which means r is in the photon sphere and there is no circular timelike orbit
    """
    if r <= 3.0:
        return float("nan")
    return r / np.sqrt(r - 3.0)


def is_stable(r):
    """Circular orbits are only stable when r >= 6 as the ISCO = 6"""
    return r >= ISCO


def metres_to_M(r_metres, M_kg):
    r_gravitational = G * M_kg / C ** 2  # 1 M-unit in metres
    return r_metres / r_gravitational


def eccentric_orbit_params(r_peri, r_apo):
    # Circular orbit: skip the solve, use the known analytic result
    if np.isclose(r_peri, r_apo):
        r0 = r_peri
        L = circular_L(r0)
        E = np.sqrt(1 - 2 / r0) / np.sqrt(1 - 3 / r0)  # E for circular orbit
        return E, L

    A = 1 - 2 / r_peri
    B = 1 - 2 / r_apo

    L_sq = (A - B) / (B / r_apo ** 2 - A / r_peri ** 2)
    L = np.sqrt(L_sq)

    E_sq = A * (1 + L_sq / r_peri ** 2)
    E = np.sqrt(E_sq)

    return E, L


def calculate_full_orbit(r_peri, r_apo,
                         inclination_deg=0.,
                         longitude_deg=0.,
                         arg_periapsis_deg=0.,
                         n_orbits=1):
    """
    Full control over all orbital elements.

    r_peri, r_apo       : periapsis and apoapsis in M-units (r_peri==r_apo → circular)
    inclination_deg     : tilt of orbital plane (0=equatorial, 90=polar)
    longitude_deg       : rotation around z-axis (φ at start)
    arg_periapsis_deg   : direction of periapsis within the orbital plane
    """
    E, L = eccentric_orbit_params(r_peri, r_apo)

    inclination = np.radians(inclination_deg)
    phi0 = np.radians(longitude_deg + arg_periapsis_deg)

    p_phi = L * np.cos(inclination)
    p_theta = -L * np.sin(inclination)

    position = [r_peri, np.pi / 2, phi0]
    momentum = [0., p_theta, p_phi]

    T = 2 * np.pi * ((r_peri + r_apo) / 2) ** (3 / 2)  # approx period via semi-major axis
    delta = 0.1
    steps = int((n_orbits * T) / delta)

    geod = Timelike(
        metric="Schwarzschild",
        metric_params=(),
        position=position,
        momentum=momentum,
        steps=steps,
        delta=delta,
        omega=0.01,
        rtol=1e-6,
        atol=1e-6,
        return_cartesian=True,
        suppress_warnings=True,
    )
    return geod


def calculate_circular_orbit(r0):
    """
    r0 is the radius from the central mass' center in M-units
    """

    L = circular_L(r0)

    position = [r0, np.pi / 2, 0.]
    momentum = [0., 0., circular_L(r0)]

    # Calculating steps needed for full orbit calculation
    T = 2 * np.pi * r0 ** (3 / 2)  # Period in M units
    delta = 0.1
    n_orbits = 1

    steps = int((n_orbits * T) / delta)

    geod = Timelike(
        metric="Schwarzschild",
        metric_params=(),
        position=position,
        momentum=momentum,
        steps=steps,
        delta=delta,
        omega=0.01,
        rtol=1e-6,
        atol=1e-6,
        return_cartesian=True,
        suppress_warnings=True,
    )

    lambdas, vecs = geod.trajectory
    # print("Columns [t, r, θ, φ, p_t, p_r, p_θ, p_φ]")
    # print("First row:", vecs[0])
    # print("Last row:", vecs[-1])
    # print("r range:", vecs[:, 1].min(), "to", vecs[:, 1].max())
    # print("phi range:", vecs[:, 3].min(), "to", vecs[:, 3].max())

    # phi_unwrapped = np.unwrap(vecs[:, 3])
    # n_orbits = (phi_unwrapped[-1] - phi_unwrapped[0]) / (2 * np.pi)
    # print(f"Completed {n_orbits:.2f} orbits")

    x, y, z = vecs[:, 1], vecs[:, 2], vecs[:, 3]
    p_x, p_y, p_z = vecs[:, 5], vecs[:, 6], vecs[:, 7]

    phi = np.arctan2(y, x)  # full -π to π
    phi_unwrapped = np.unwrap(phi)  # accumulates continuously

    n_completed = (phi_unwrapped[-1] - phi_unwrapped[0]) / (2 * np.pi)
    print(f"Completed {n_completed:.2f} orbits")

    # CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    x = r_to_meters(r0, 1.3e41)
    y = r_to_meters(r0, 1.3e41)
    z = r_to_meters(r0, 1.3e41)

    p_x *= C
    p_y *= C
    p_z *= C

    return


def get_trajectory_state_vector(geod_or_vecs, mass, m_units=False):
    G = 6.674e-11
    M_to_meters = G * mass / C**2

    # Accept either a geodesic object or a raw (N,8) array
    if isinstance(geod_or_vecs, np.ndarray):
        vecs = geod_or_vecs
    else:
        lambdas, vecs = geod_or_vecs.trajectory

    r = np.sqrt(vecs[:,1]**2 + vecs[:,2]**2 + vecs[:,3]**2)

    r_min = 2.0
    r_max = 1e5

    if r[-1] <= r_min:
        fate = "captured"
    elif r[-1] >= r_max:
        fate = "escaped"
    else:
        fate = "incomplete"

    outside = r > r_min
    if not outside.any():
        return None, None, fate

    last_out = np.where(outside)[0][-1]

    if fate == "captured" and last_out + 1 < len(r):
        # Interpolate exact surface crossing — ray ends precisely on sphere
        t = (r_min - r[last_out]) / (r[last_out + 1] - r[last_out])
        crossing = vecs[last_out, 1:4] + t * (vecs[last_out + 1, 1:4] - vecs[last_out, 1:4])
        pos = np.vstack([vecs[:last_out + 1, 1:4], crossing])
        mom = np.vstack([vecs[:last_out + 1, 5:8],
                         vecs[last_out, 5:8] + t * (vecs[last_out + 1, 5:8] - vecs[last_out, 5:8])])
    else:
        mask = outside & (r < r_max)
        pos = vecs[mask, 1:4]
        mom = vecs[mask, 5:8]

    if len(pos) < 2:
        return None, None, fate

    if m_units:
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        px, py, pz = mom[:, 0], mom[:, 1], mom[:, 2]
    else:
        x, y, z = pos[:, 0] * M_to_meters, pos[:, 1] * M_to_meters, pos[:, 2] * M_to_meters
        px, py, pz = mom[:, 0] * C, mom[:, 1] * C, mom[:, 2] * C

    print(f"Ray fate: {fate}, {len(x)} points")
    print(f"Ray r range: {r[outside].min():.1f} to {r[outside].max():.1f} M-units")

    return np.array([x, y, z]), np.array([px, py, pz]), fate

def calculate_null_geodesic(x, y, z, dx, dy, dz, steps=200000, delta=0.01):
    """
    x,y,z   : starting position in M-units (Cartesian)
    dx,dy,dz: light ray direction (Cartesian, any scale)
    """
    # Convert position
    r, theta, phi = cartesian_to_schwarzschild(x, y, z)

    # Convert direction to spherical momentum
    p_r, p_theta, p_phi = cartesian_to_spherical_momentum(
                        x, y, z, dx, dy, dz)

    # Solve null condition for p^t
    p_t = null_pt(r, theta, p_r, p_theta, p_phi)

    position = [r, theta, phi]
    momentum = [p_r, p_theta, p_phi]   # einsteinpy handles p_t internally
                                        # via the null constraint

    geod = Nulllike(
        metric="Schwarzschild",
        metric_params=(),
        position=position,
        momentum=momentum,
        steps=steps,
        delta=delta,
        omega=0.0,
        rtol=1e-9,
        atol=1e-9,
        return_cartesian=True,
        suppress_warnings=True,
    )

    return geod

def calculate_null_geodesic_smart(x, y, z, dx, dy, dz, r_max=1e4):
    """
    Integrates in chunks and stops early once fate is determined.
    Avoids wasting steps on a ray that escaped at step 100 out of 200000.
    """
    r, theta, phi = cartesian_to_schwarzschild(x, y, z)
    p_r, p_theta, p_phi = cartesian_to_spherical_momentum(x, y, z, dx, dy, dz)

    position = [r, theta, phi]
    momentum = [p_r, p_theta, p_phi]

    chunk_size = 5000
    delta = 0.1 if r > 20 else 0.01
    all_vecs = []

    for _ in range(40):   # max 40 chunks = 200000 steps
        geod = Nulllike(
            metric="Schwarzschild",
            metric_params=(),
            position=position,
            momentum=momentum,
            steps=chunk_size,
            delta=delta,
            omega=0.0,
            rtol=1e-9,
            atol=1e-9,
            return_cartesian=True,
            suppress_warnings=True,
        )

        lambdas, vecs = geod.trajectory
        all_vecs.append(vecs)

        # Check current r
        r_current = np.sqrt(vecs[-1,1]**2 + vecs[-1,2]**2 + vecs[-1,3]**2)

        if r_current <= 2.0:
            print(f"Captured after {len(all_vecs)*chunk_size} steps")
            break
        if r_current >= r_max:
            print(f"Escaped after {len(all_vecs)*chunk_size} steps")
            break

        # Tighten step near the black hole
        delta = 0.01 if r_current < 20 else 0.1

        # Resume from where we left off
        position = list(vecs[-1, 1:4])    # NOTE: these are cartesian from einsteinpy
        momentum = list(vecs[-1, 5:8])

    return np.vstack(all_vecs)

def cartesian_to_schwarzschild(x, y, z):
    """All inputs in M-units"""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def cartesian_to_spherical_momentum(x, y, z, dx, dy, dz):
    """
    Returns COVARIANT momentum components [p_r, p_theta, p_phi]
    which is what einsteinpy's geodesic integrator expects.
    """
    r   = np.sqrt(x**2 + y**2 + z**2)
    rho = np.sqrt(x**2 + y**2)
    f   = 1 - 2/r   # lapse factor (M=1)

    # Contravariant components from the Jacobian
    p_r_contra     = (x*dx + y*dy + z*dz) / r
    p_theta_contra = (z*(x*dx + y*dy) - rho**2*dz) / (r**2 * rho)
    p_phi_contra   = (x*dy - y*dx) / rho**2

    # Convert to covariant via metric (g is diagonal in Schwarzschild)
    p_r     = p_r_contra     / f        # g_rr     = 1/f
    p_theta = p_theta_contra * r**2     # g_θθ     = r²
    p_phi   = p_phi_contra   * rho**2   # g_φφ     = r²sin²θ = rho²
    # Note: p_phi simplifies to just x*dy - y*dx = L (angular momentum)

    return p_r, p_theta, p_phi

def null_pt(r, theta, p_r, p_theta, p_phi):
    """
    Solve g^{µν} p_µ p_ν = 0 for p_t, given covariant spatial components.
    g^tt=-1/f, g^rr=f, g^θθ=1/r², g^φφ=1/(r²sin²θ)
    """
    f = 1 - 2/r
    spatial_term = (f * p_r**2
                  + p_theta**2 / r**2
                  + p_phi**2  / (r**2 * np.sin(theta)**2))
    p_t = -np.sqrt(f * spatial_term)   # negative → future-directed
    return p_t

fig, ax = plt.subplots(figsize=(6, 6))
# Setting boundaries and scale for solar system
# Scale Value
scale = 50
ax.set_xlim(-scale, scale)
ax.set_ylim(-scale, scale)
ax.set_aspect('equal')

black_hole_mass = 1.31268720076e41

black_hole = plt.Circle((0,0), 2, color='black')
ax.add_artist(black_hole)

# test_rays = [
#         (50, 0, 0, -50, 5.1, 0, "captured"),
#         (50, 0, 0, -50, 5.196, 0, "near miss - strong bending"),
#         (50, 0, 0, -50, 5.3, 0, "far miss - slight bending"),
#     ]

# for x, y, z, dx, dy, dz, label in test_rays:
#     print(f"\n--- {label} ---")
#     geod = calculate_null_geodesic(x, y, z, dx, dy, dz)
#     position, momentum, fate = get_trajectory_state_vector(geod, black_hole_mass)
#     if position is not None:
#         points = np.column_stack([position[0], position[1], position[2]])
#         x = []
#         y = []
#         r_is_greater = True
#         for i in range(position[0].size):
#             if np.sqrt(position[0][i] ** 2 + position[1][i] ** 2) >= 2.05 and r_is_greater:
#                 x.append(position[0][i])
#                 y.append(position[1][i])
#             else:
#                 r_is_greater = False
#         plt.plot(x, y, markersize=6)

def impact_param_to_dy(r0, b_target):
    """
    Given a starting position r0 on the x-axis and a desired impact parameter b,
    returns the dy offset needed for calculate_null_geodesic(r0, 0, 0, -r0, dy, 0)
    """
    f = 1 - 2/r0
    # From: b = r0*dy / sqrt(f*(r0^2/f + dy^2))
    # Solving for dy:
    # b^2 * f * (r0^2/f + dy^2) = r0^2 * dy^2
    # b^2 * r0^2 + b^2*f*dy^2 = r0^2 * dy^2
    # dy^2 * (r0^2 - b^2*f) = b^2 * r0^2
    dy_sq = (b_target**2 * r0**2) / (r0**2 - b_target**2 * f)
    return np.sqrt(dy_sq)

b_crit = 3 * np.sqrt(3)   # ≈ 5.196

r0 = 50
dy_captured  = impact_param_to_dy(r0, b_crit * 0.99)  # just inside
dy_critical  = impact_param_to_dy(r0, b_crit)          # exactly critical
dy_escaped   = impact_param_to_dy(r0, b_crit * 1.01)  # just outside


black_hole_null_geod = calculate_null_geodesic_smart(r0, 0 , 0, -r0, dy_critical, 0)

position, momentum, fate = get_trajectory_state_vector(black_hole_null_geod, black_hole_mass, True)
points = np.column_stack([position[0], position[1], position[2]])

x = []
y = []
r_is_greater = True
for i in range(position[0].size):
    if np.sqrt(position[0][i] ** 2 + position[1][i] ** 2) >= 2.05 and r_is_greater:
        x.append(position[0][i])
        y.append(position[1][i])
    else:
        r_is_greater = False
light_ray = plt.plot(x, y, markersize=6)

plt.show()