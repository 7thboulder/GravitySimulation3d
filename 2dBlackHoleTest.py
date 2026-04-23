import numpy as np
import pyvista as pv
from einsteinpy.plotting import GeodesicPlotter
from numpy.f2py.auxfuncs import throw_error
from einsteinpy.metric import Schwarzschild
from einsteinpy.geodesic import Geodesic, Timelike, Nulllike
import astropy.units as u
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

G = 6.6743e-11
C = 299792458
ISCO = 6
rs = 2



def geodesic_odes(lamb, state):
    """
    Schwarzschild null geodesic equations in Schwarzschild coordinates.
    state = [r, theta, phi, p_r, p_theta, p_phi]
    M = 1 (geometrized units)
    """
    r, theta, phi, p_r, p_theta, p_phi = state

    f     = 1 - 2/r
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # Conserved quantities
    # E = f * p^t,  L = p_phi  (covariant)
    # From normalization: E² = f(p_r²*f + p_theta²/r² + p_phi²/(r²sin²θ))
    E_sq = f * (p_r**2 * f + p_theta**2/r**2 + p_phi**2/(r**2 * sin_t**2))

    # Equations of motion (geodesic equations)
    dr_dl      = f * p_r
    dtheta_dl  = p_theta / r**2
    dphi_dl    = p_phi   / (r**2 * sin_t**2)

    # Hamilton-Jacobi: dp_i/dλ = -½ ∂g^{µν}/∂x^i * p_µ p_ν
    dp_r_dl     = ( -(1/r**2) * p_r**2 * f
                    + (1/r**3) * p_theta**2
                    + (1/r**3) * p_phi**2 / sin_t**2
                    - (1/r**2) * E_sq / f )

    dp_theta_dl = ( cos_t / (r**2 * sin_t**3) ) * p_phi**2

    dp_phi_dl   = 0.0   # phi is cyclic → p_phi is conserved

    return [dr_dl, dtheta_dl, dphi_dl, dp_r_dl, dp_theta_dl, dp_phi_dl]


def event_captured(lamb, state):
    """Stop when ray hits event horizon r = 2"""
    return state[0] - 2.0
event_captured.terminal  = True
event_captured.direction = -1   # only trigger when r is decreasing


def event_escaped(lamb, state, r_max=1e4):
    """Stop when ray escapes to r_max"""
    return state[0] - r_max
event_escaped.terminal  = True
event_escaped.direction = 1    # only trigger when r is increasing


def calculate_null_geodesic_fast(x, y, z, dx, dy, dz, r_max=1e4, max_steps=10000):
    """
    Direct scipy integration — ~10-50x faster than einsteinpy,
    with proper early termination.
    """
    r, theta, phi = cartesian_to_schwarzschild(x, y, z)
    p_r, p_theta, p_phi = cartesian_to_spherical_momentum(x, y, z, dx, dy, dz)

    state0 = [r, theta, phi, p_r, p_theta, p_phi]

    # λ span — scale to starting r so we don't under/over-integrate
    lambda_max = r * 500

    sol = solve_ivp(
        geodesic_odes,
        t_span=(0, lambda_max),
        y0=state0,
        method='RK45',
        events=[event_captured, lambda l, s: event_escaped(l, s, r_max)],
        rtol=1e-9,
        atol=1e-9,
        dense_output=False,
        max_step=r/10,      # never take steps larger than r/10
    )

    # Convert Schwarzschild trajectory back to Cartesian for plotting
    r_arr     = sol.y[0]
    theta_arr = sol.y[1]
    phi_arr   = sol.y[2]

    x_arr = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
    y_arr = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
    z_arr = r_arr * np.cos(theta_arr)

    # Determine fate
    if sol.t_events[0].size > 0:
        fate = "captured"
    elif sol.t_events[1].size > 0:
        fate = "escaped"
    else:
        fate = "incomplete"

    print(f"Fate: {fate}, {len(r_arr)} steps, r range: {r_arr.min():.2f} to {r_arr.max():.2f}")

    return np.array([x_arr, y_arr, z_arr]), fate

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

    if isinstance(geod_or_vecs, np.ndarray):
        vecs = geod_or_vecs
    else:
        lambdas, vecs = geod_or_vecs.trajectory

    r = np.sqrt(vecs[:,1]**2 + vecs[:,2]**2 + vecs[:,3]**2)

    r_min = 2.0
    r_max = 1e4

    # ---- KEY FIX: strip any points where r jumps discontinuously ----
    # A real geodesic changes r smoothly — a sudden jump means the integrator diverged
    dr = np.abs(np.diff(r))
    dr_median = np.median(dr)
    diverge_idx = np.where(dr > dr_median * 1000)[0]  # 1000x median step = diverged

    if diverge_idx.size > 0:
        cutoff = diverge_idx[0] + 1   # trim everything after first big jump
        vecs = vecs[:cutoff]
        r    = r[:cutoff]
        print(f"Trimmed divergent tail at step {cutoff}/{len(r)}")

    # Now determine fate from the trimmed trajectory
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
        t = (r_min - r[last_out]) / (r[last_out+1] - r[last_out])
        crossing = vecs[last_out, 1:4] + t * (vecs[last_out+1, 1:4] - vecs[last_out, 1:4])
        pos = np.vstack([vecs[:last_out+1, 1:4], crossing])
        mom = np.vstack([vecs[:last_out+1, 5:8],
                         vecs[last_out, 5:8] + t * (vecs[last_out+1, 5:8] - vecs[last_out, 5:8])])
    else:
        mask = outside & (r < r_max)
        pos = vecs[mask, 1:4]
        mom = vecs[mask, 5:8]

    if len(pos) < 2:
        return None, None, fate

    if m_units:
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        px, py, pz = mom[:,0], mom[:,1], mom[:,2]
    else:
        x, y, z = pos[:,0]*M_to_meters, pos[:,1]*M_to_meters, pos[:,2]*M_to_meters
        px, py, pz = mom[:,0]*C, mom[:,1]*C, mom[:,2]*C

    print(f"Ray fate: {fate}, {len(x)} points")
    return np.array([x,y,z]), np.array([px,py,pz]), fate

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
    r, theta, phi = cartesian_to_schwarzschild(x, y, z)
    p_r, p_theta, p_phi = cartesian_to_spherical_momentum(x, y, z, dx, dy, dz)

    position = [r, theta, phi]
    momentum = [p_r, p_theta, p_phi]

    chunk_size = 2000
    delta = 0.1 if r > 20 else 0.01
    all_vecs = []

    for i in range(100):
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
            return_cartesian=True,   # Cartesian output for plotting
            suppress_warnings=True,
        )

        lambdas, vecs = geod.trajectory
        all_vecs.append(vecs)

        # Last point in Cartesian
        cx, cy, cz = vecs[-1, 1], vecs[-1, 2], vecs[-1, 3]
        r_current = np.sqrt(cx**2 + cy**2 + cz**2)

        if r_current <= 2.0:
            print(f"Captured after {(i+1)*chunk_size} steps")
            break
        if r_current >= r_max:
            print(f"Escaped after {(i+1)*chunk_size} steps")
            break

        # ---- KEY FIX: convert Cartesian back to Schwarzschild for next chunk ----
        # Position
        r_new, theta_new, phi_new = cartesian_to_schwarzschild(cx, cy, cz)

        # Momentum: vecs[:,5:8] are Cartesian p_x, p_y, p_z
        # Convert back to covariant Schwarzschild using current position
        dpx, dpy, dpz = vecs[-1, 5], vecs[-1, 6], vecs[-1, 7]
        p_r_new, p_theta_new, p_phi_new = cartesian_to_spherical_momentum(
            cx, cy, cz, dpx, dpy, dpz
        )

        position = [r_new, theta_new, phi_new]
        momentum = [p_r_new, p_theta_new, p_phi_new]

        # Adjust step size based on new r
        delta = 0.01 if r_current < 20 else 0.1

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

# b_crit = 3 * np.sqrt(3)   # ≈ 5.196
#
# r0 = 50
# dy_captured  = impact_param_to_dy(r0, b_crit * 0.99)  # just inside
# dy_critical  = impact_param_to_dy(r0, b_crit)          # exactly critical
# dy_escaped   = impact_param_to_dy(r0, b_crit * 1.01)  # just outside
#
#
# black_hole_null_geod = calculate_null_geodesic_smart(r0, 0 , 0, -r0, dy_critical, 0)
#
# position, momentum, fate = get_trajectory_state_vector(black_hole_null_geod, black_hole_mass, True)
# points = np.column_stack([position[0], position[1], position[2]])
#
# x = []
# y = []
# r_is_greater = True
# for i in range(position[0].size):
#     if np.sqrt(position[0][i] ** 2 + position[1][i] ** 2) >= 2.05 and r_is_greater:
#         x.append(position[0][i])
#         y.append(position[1][i])
#     else:
#         r_is_greater = False
# light_ray = plt.plot(x, y, markersize=6)
#
# plt.show()

b_crit = 3 * np.sqrt(3)
r0 = 50

COLORS = {"captured": "red", "escaped": "yellow", "incomplete": "grey"}

test_rays = [
    (b_crit * 0.95, "captured"),
    (b_crit,        "photon sphere"),
    (b_crit * 1.05, "escaped"),
    (b_crit * 2.0,  "far escape"),
]

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-r0, r0)
ax.set_ylim(-r0, r0)
ax.set_aspect('equal')
ax.set_facecolor('black')

black_hole   = plt.Circle((0, 0), 2, color='black',  zorder=5)
photon_sphere = plt.Circle((0, 0), 3, color='white', fill=False, linestyle='--', zorder=4)
ax.add_artist(black_hole)
ax.add_artist(photon_sphere)

for b, label in test_rays:
    dy = impact_param_to_dy(r0, b)
    position, fate = calculate_null_geodesic_fast(r0, 0, 0, -r0, dy, 0)
    ax.plot(position[0], position[1], color=COLORS[fate], label=f"{label} b={b:.2f}", linewidth=1)

ax.legend(loc='upper right', facecolor='black', labelcolor='white')
plt.show()