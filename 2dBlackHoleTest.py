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



def calculate_null_geodesic_fast(x, y, z, dx, dy, dz, r_max=1e4):
    r, theta, phi = cartesian_to_schwarzschild(x, y, z)
    p_r, p_theta, p_phi = cartesian_to_spherical_momentum(x, y, z, dx, dy, dz)

    state0 = [r, theta, phi, p_r, p_theta, p_phi]

    # Compute E once from initial conditions and keep it conserved
    # This is more numerically stable than recomputing from state each step
    f0 = 1 - 2/r
    E = np.sqrt(f0**2 * p_r**2
              + f0 * p_theta**2 / r**2
              + f0 * p_phi**2   / (r**2 * np.sin(theta)**2))

    def geodesic_odes(lamb, state):
        r, theta, phi, p_r, p_theta, p_phi = state
        f     = 1 - 2/r
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        dr_dl     = f * p_r
        dtheta_dl = p_theta / r**2
        dphi_dl   = p_phi   / (r**2 * sin_t**2)

        # Corrected — E is the conserved constant from initial conditions
        dp_r_dl = (- E**2         / (r**2 * f**2)   # was: E_sq/f  ← wrong
                   - p_r**2       /  r**2            # was: p_r²*f  ← wrong
                   + p_theta**2   /  r**3
                   + p_phi**2     / (r**3 * sin_t**2))

        dp_theta_dl = cos_t * p_phi**2 / (r**2 * sin_t**3)
        dp_phi_dl   = 0.0

        return [dr_dl, dtheta_dl, dphi_dl, dp_r_dl, dp_theta_dl, dp_phi_dl]

    def event_captured(lamb, state):
        return state[0] - 2.0
    event_captured.terminal  = True
    event_captured.direction = -1

    def event_escaped(lamb, state):
        return state[0] - r_max
    event_escaped.terminal  = True
    event_escaped.direction = 1

    sol = solve_ivp(
        geodesic_odes,
        t_span=(0, r * 500),
        y0=state0,
        method='RK45',
        events=[event_captured, event_escaped],
        rtol=1e-9,
        atol=1e-9,
        max_step=r/10,
    )

    r_arr     = sol.y[0]
    theta_arr = sol.y[1]
    phi_arr   = sol.y[2]

    x_arr = r_arr * np.sin(theta_arr) * np.cos(phi_arr)
    y_arr = r_arr * np.sin(theta_arr) * np.sin(phi_arr)
    z_arr = r_arr * np.cos(theta_arr)

    if   sol.t_events[0].size > 0: fate = "captured"
    elif sol.t_events[1].size > 0: fate = "escaped"
    else:                           fate = "incomplete"

    print(f"Fate: {fate}, {len(r_arr)} steps, r: {r_arr.min():.2f} to {r_arr.max():.2f}")

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
ax.set_facecolor('white')

black_hole   = plt.Circle((0, 0), 2, color='black',  zorder=5)
photon_sphere = plt.Circle((0, 0), 3, color='white', fill=False, linestyle='--', zorder=4)
ax.add_artist(black_hole)
ax.add_artist(photon_sphere)

for b, label in test_rays:
    dy = impact_param_to_dy(r0, b)
    position, fate = calculate_null_geodesic_fast(r0, 0, 0, -r0, dy, 0)
    ax.plot(position[0], position[1], color=COLORS[fate], label=f"{label} b={b:.2f}", linewidth=1.5)

ax.legend(loc='upper right', facecolor='black', labelcolor='white')
plt.show()