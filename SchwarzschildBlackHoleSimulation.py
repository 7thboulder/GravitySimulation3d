import numpy as np
import pyvista as pv
from einsteinpy.plotting import GeodesicPlotter
from numpy.f2py.auxfuncs import throw_error
from einsteinpy.metric import Schwarzschild
from einsteinpy.geodesic import Geodesic, Timelike, Nulllike
import astropy.units as u
import matplotlib as mp

class BlackHoleSimulation:
    def __init__(
            self,
            central_mass: float,
            central_color: str,
            central_name: str,
            dt: float,
            timer_interval=16,
            rotation_speed=1.5,
    ):

        self.central_mass = central_mass
        self.G = 6.6743e-11
        self.C = 299792458

        # 1 M-unit in meters
        self.M_in_meters = self.G * self.central_mass / self.C ** 2

        # Fix: 1 render unit = 1 M-unit
        # so render_scale = M_in_meters, and all M-unit distances map 1:1 to render units
        self.render_scale = self.M_in_meters

        self.central_radius = 2 * self.M_in_meters  # exact event horizon in meters

        # Camera starts 200 M-units back — always sensible regardless of mass
        self.camera_position = np.array([0.0, -200.0 * self.M_in_meters,
                                         50.0 * self.M_in_meters], dtype=float)
        self.camera_speed = 10.0 * self.M_in_meters  # 10 M-units/sec

        self.central_color = central_color
        self.central_name = central_name




        #Schwarzschild Metric stuff
        self.rs = 2
        self.ISCO = 6
        self.negligible_bend_distance = 30
        self.light_rays = []

        # Graphics variables
        self.plotter = None
        self.timer_interval = timer_interval
        self.pressed_keys = set()

        self.yaw = np.pi / 2.0
        self.pitch = -0.2
        self.rotation_speed = rotation_speed

        # Handles to scene objects that need to be updated over time.
        self.central_actor = None
        self.body_actors = {}
        self.label_points = None
        self.label_names = []
        self.is_paused = False
        self.status_text = None
        self.speed_text = None
        self.trail_meshes = {}
        self.trail_length = 5000
        self.camera_speed_step = 1.5
        self.speed_of_light = 299_792_458.0
        self.au_in_meters = 149_597_870_700.0



    def r_to_meters(self, r_geom, M_kg):
        return r_geom * self.G * M_kg / self.C**2

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

    def circular_L(self, r):
        """
        Angular momentum for a circular orbit at r in M-units
        Returns nan if r <= 3 which means r is in the photon sphere and there is no circular timelike orbit
        """
        if r <= 3.0:
            return float("nan")
        return r / np.sqrt(r - 3.0)

    def is_stable(self, r):
        """Circular orbits are only stable when r >= 6 as the ISCO = 6"""
        return r >= self.ISCO

    def metres_to_M(self, r_metres, M_kg):

        r_gravitational = self.G * M_kg / self.C ** 2  # 1 M-unit in metres
        return r_metres / r_gravitational

    def eccentric_orbit_params(self, r_peri, r_apo):
        # Circular orbit: skip the solve, use the known analytic result
        if np.isclose(r_peri, r_apo):
            r0 = r_peri
            L = self.circular_L(r0)
            E = np.sqrt(1 - 2 / r0) / np.sqrt(1 - 3 / r0)  # E for circular orbit
            return E, L

        A = 1 - 2 / r_peri
        B = 1 - 2 / r_apo

        L_sq = (A - B) / (B / r_apo ** 2 - A / r_peri ** 2)
        L = np.sqrt(L_sq)

        E_sq = A * (1 + L_sq / r_peri ** 2)
        E = np.sqrt(E_sq)

        return E, L

    def calculate_full_orbit(self, r_peri, r_apo,
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
        E, L = self.eccentric_orbit_params(r_peri, r_apo)

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

    def calculate_circular_orbit(self, r0):
        """
        r0 is the radius from the central mass' center in M-units
        """

        L = self.circular_L(r0)

        position = [r0, np.pi/2, 0.]
        momentum = [0. , 0., self.circular_L(r0)]

        # Calculating steps needed for full orbit calculation
        T = 2 * np.pi * r0**(3/2) # Period in M units
        delta = 0.1
        n_orbits = 1

        steps = int((n_orbits * T)/delta)

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
        x = self.r_to_meters(r0, 1.3e41)
        y = self.r_to_meters(r0, 1.3e41)
        z = self.r_to_meters(r0, 1.3e41)


        p_x *= self.C
        p_y *= self.C
        p_z *= self.C

        return

    def get_trajectory_state_vector(self, geod, mass):
        G = 6.674e-11
        M_to_meters = G * mass / self.C ** 2

        lambdas, vecs = geod.trajectory
        r = np.sqrt(vecs[:, 1] ** 2 + vecs[:, 2] ** 2 + vecs[:, 3] ** 2)

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

        x, y, z = pos[:, 0] * M_to_meters, pos[:, 1] * M_to_meters, pos[:, 2] * M_to_meters
        px, py, pz = mom[:, 0] * self.C, mom[:, 1] * self.C, mom[:, 2] * self.C

        print(f"Ray fate: {fate}, {len(x)} points")
        print(f"Ray r range: {r[outside].min():.1f} to {r[outside].max():.1f} M-units")

        return np.array([x, y, z]), np.array([px, py, pz]), fate



    def plot_orbit_with_mass(self, geod):
        gpl = GeodesicPlotter()
        gpl.plot(geod, color="green")
        gpl.show()

        gpl.clear()  # In Interactive mode, `clear()` must be called before drawing another plot, to avoid overlap
        gpl.plot2D(geod, coordinates=(1, 2), color="green", title="Top/Bottom View")  # "top" / "bottom" view
        gpl.show()

        gpl.clear()
        gpl.plot2D(geod, coordinates=(1, 3), color="green", title="Face-On View")  # "face-on" view
        gpl.show()

# Raytracing physics

    def calculate_null_geodesic(self, x, y, z, dx, dy, dz, steps=10000, delta=0.01):
        """
        x,y,z   : starting position in M-units (Cartesian)
        dx,dy,dz: light ray direction (Cartesian, any scale)
        """
        # Convert position
        r, theta, phi = self.cartesian_to_schwarzschild(x, y, z)

        # Convert direction to spherical momentum
        p_r, p_theta, p_phi = self.cartesian_to_spherical_momentum(
                                x, y, z, dx, dy, dz)

        # Solve null condition for p^t
        p_t = self.null_pt(r, theta, p_r, p_theta, p_phi)

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

    def cartesian_to_schwarzschild(self, x, y, z):
        """All inputs in M-units"""
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    def cartesian_to_spherical_momentum(self, x, y, z, dx, dy, dz):
        """
        (x,y,z)     : position in M-units
        (dx,dy,dz)  : light direction vector (scale doesn't matter for null geodesics)
        Returns: p_r, p_theta, p_phi (contravariant, in Schwarzschild coords)
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        rho = np.sqrt(x ** 2 + y ** 2)  # cylindrical radius

        p_r = (x * dx + y * dy + z * dz) / r
        p_theta = (z * (x * dx + y * dy) - rho ** 2 * dz) / (r ** 2 * rho)
        p_phi = (x * dy - y * dx) / rho ** 2

        return p_r, p_theta, p_phi

    def null_pt(self, r, theta, p_r, p_theta, p_phi):
        """Solve g_μν p^μ p^ν = 0 for p^t (Schwarzschild, M=1)"""
        f = 1 - 2 / r  # lapse factor

        spatial_term = (p_r ** 2 / f
                        + r ** 2 * p_theta ** 2
                        + r ** 2 * np.sin(theta) ** 2 * p_phi ** 2)

        p_t = -np.sqrt(spatial_term / f)  # negative → future-directed
        return p_t

    def test_raytracing(self):

        # 1. Create source mesh
        sphere = pv.Sphere(radius=3000)


        # 2. Define line segment (ray)
        start = [10, 0.5, 0]
        stop = [-10, 0.5, 0]

        x = 10.0
        y = 4
        z = 0.0

        dx = -10.0
        dy = 4
        dz = 0.0

        test_rays = [
            (200, 0, 0, -200, 4.9, 0, "captured"),
            (200, 0, 0, -200, 5.2, 0, "near miss - strong bending"),
            (200, 0, 0, -200, 20.0, 0, "far miss - slight bending"),
        ]

        for x, y, z, dx, dy, dz, label in test_rays:
            print(f"\n--- {label} ---")
            geod = self.calculate_null_geodesic(x, y, z, dx, dy, dz)
            position, momentum, fate = self.get_trajectory_state_vector(geod, self.central_mass)
            if position is not None:
                points = np.column_stack([position[0], position[1], position[2]])
                self.create_light_ray(points, fate=fate)

        pl = pv.Plotter()
        pl.add_mesh(sphere, color='w', opacity=0.5, show_edges=True)
        pl.add_mesh(line, color='blue', line_width=5)
        pl.show()

    # End of Schwarzschild math

    def _as_3d_vector(self, vector):
        # Normalize any 2D/3D input into a 3D vector for rendering math.
        vector_array = np.array(vector, dtype=float)
        padded_vector = np.zeros(3, dtype=float)
        padded_vector[:min(3, len(vector_array))] = vector_array[:3]
        return padded_vector

    def _body_position_3d(self, body):
        return self._as_3d_vector(body.get_position())

    def _camera_basis(self):
        # Convert yaw/pitch into an orthonormal forward/right/up basis.
        cos_pitch = np.cos(self.pitch)
        forward = np.array(
            [
                cos_pitch * np.cos(self.yaw),
                cos_pitch * np.sin(self.yaw),
                np.sin(self.pitch),
            ],
            dtype=float,
        )
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return forward, right, up

    def _render_position(self, real_position):
        # Floating-origin rendering keeps large astronomical coordinates near the camera.
        return (self._as_3d_vector(real_position) - self.camera_position) / self.render_scale

    def _camera_key_map(self):
        # Arrow keys rotate the camera rather than translating it.
        return {
            'Left': ('yaw', 1.0),
            'Right': ('yaw', -1.0),
            'Up': ('pitch', 1.0),
            'Down': ('pitch', -1.0),
        }

    def _movement_key_map(self):
        # Movement keys translate the camera through the scene.
        return {
            'w': ('forward', 1.0),
            's': ('forward', -1.0),
            'a': ('right', -1.0),
            'd': ('right', 1.0),
            'q': ('up', 1.0),
            'e': ('up', -1.0),
        }

    def _set_camera_view(self):
        # The actual VTK camera stays near the origin; the world is shifted around it.
        forward, _, up = self._camera_basis()
        self.plotter.camera.position = (0.0, 0.0, 0.0)
        self.plotter.camera.focal_point = tuple(forward * 10.0)
        self.plotter.camera.up = tuple(up)

    def initialize_camera(self):
        if self.plotter is None:
            raise RuntimeError('Call setup_scene before initializing the camera.')
        self._set_camera_view()

    def _on_key_press(self, obj, _event):
        # Track held keys so movement stays smooth between timer ticks.
        self.pressed_keys.add(obj.GetKeySym())

    def _on_key_release(self, obj, _event):
        self.pressed_keys.discard(obj.GetKeySym())

    def bind_inputs(self):
        if self.plotter is None:
            raise RuntimeError('Call setup_scene before binding inputs.')

        # Raw VTK observers capture held keys, while PyVista key events handle one-shot toggles.
        interactor = self.plotter.iren.interactor
        interactor.AddObserver('KeyPressEvent', self._on_key_press)
        interactor.AddObserver('KeyReleaseEvent', self._on_key_release)
        self.plotter.add_key_event('space', self.toggle_pause)
        self.plotter.add_key_event('p', self.toggle_pause)
        self.plotter.add_key_event('bracketright', self.increase_camera_speed)
        self.plotter.add_key_event('bracketleft', self.decrease_camera_speed)

    def _create_central_visual(self):
        # central_radius is already the Schwarzschild radius in meters
        # just convert to render units — no artificial minimum
        radius_render = self.central_radius / self.render_scale
        central_mesh = pv.Sphere(radius=radius_render)
        self.central_actor = self.plotter.add_mesh(
            central_mesh,
            color=self.central_color,
            name='central-body'
        )

    def setup_scene(self, plotter=None):

        print(f"1 M-unit = {self.M_in_meters:.1f} meters")
        print(f"1 M-unit = 1 render unit")
        print(f"Event horizon at {self.central_radius / self.render_scale:.2f} render units")
        print(f"Camera at {self.camera_position / self.render_scale} M-units")


        # Build the PyVista scene once, then reuse the stored actors each frame.
        self.plotter = plotter if plotter is not None else pv.Plotter()
        self.plotter.set_background('black')
        self.plotter.add_axes()
        self.plotter.add_text(
            'Arrow keys: look | W/A/S/D/Q/E: move | [ / ]: camera speed | Space/P: pause',
            position='upper_left',
            font_size=10,
            color='white',
        )
        self.status_text = self.plotter.add_text('Running', position='upper_right', font_size=10, color='white')
        self.speed_text = self.plotter.add_text('', position='lower_left', font_size=10, color='white')

        self._create_central_visual()

        test_rays = [
            (200, 0, 0, -200, 4.9, 0, "captured"),
            (200, 0, 0, -200, 5.2, 0, "near miss - strong bending"),
            (200, 0, 0, -200, 20.0, 0, "far miss - slight bending"),
        ]

        for x, y, z, dx, dy, dz, label in test_rays:
            print(f"\n--- {label} ---")
            geod = self.calculate_null_geodesic(x, y, z, dx, dy, dz)
            position, momentum, fate = self.get_trajectory_state_vector(geod, self.central_mass)
            if position is not None:
                points = np.column_stack([position[0], position[1], position[2]])
                self.create_light_ray(points, fate=fate)

        self.bind_inputs()
        self.initialize_camera()
        self._update_speed_text()
        self.sync_visuals()
        return self.plotter

    def create_light_ray(self, position_meters, fate="escaped"):
        FATE_COLORS = {"captured": "red", "escaped": "yellow", "incomplete": "grey"}

        # Use first point as the actor's world-space anchor
        center = position_meters[0]

        # Build mesh relative to that anchor, already in render scale
        relative_points = (position_meters - center) / self.render_scale

        if len(relative_points) < 2:
            return

        light_mesh = pv.Spline(relative_points, n_points=min(1000, len(relative_points)))
        light_actor = self.plotter.add_mesh(
            light_mesh,
            color=FATE_COLORS.get(fate, "white"),
            line_width=2
        )

        # Store (actor, world-space anchor in meters, fate)
        self.light_rays.append((light_actor, center, fate))

    def update_all(self, _frame=None):
        pass

    def update_camera(self, dt):
        # Apply rotation first so movement uses the latest look direction.
        for key_sym, (axis_name, direction) in self._camera_key_map().items():
            if key_sym not in self.pressed_keys:
                continue
            if axis_name == 'yaw':
                self.yaw += direction * self.rotation_speed * dt
            elif axis_name == 'pitch':
                self.pitch += direction * self.rotation_speed * dt

        # Clamp pitch so the camera cannot flip over at the poles.
        self.pitch = float(np.clip(self.pitch, -np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05))

        forward, right, up = self._camera_basis()
        basis_vectors = {
            'forward': forward,
            'right': right,
            'up': up,
        }
        for key_sym, (vector_name, direction) in self._movement_key_map().items():
            if key_sym not in self.pressed_keys:
                continue
            self.camera_position += basis_vectors[vector_name] * direction * self.camera_speed * dt

        self._set_camera_view()

    def _force_surface_rendering(self):
        # PyVista binds `W` to wireframe by default; this keeps body meshes solid.
        actors = list(self.body_actors.values())
        if self.central_actor is not None:
            actors.append(self.central_actor)

        for actor in actors:
            actor.GetProperty().SetRepresentationToSurface()

    def _update_status_text(self):
        if self.status_text is not None:
            self.status_text.SetText(3, 'Paused' if self.is_paused else 'Running')

    def _format_camera_speed(self):
        # Show camera speed in the most readable unit for the current scale.
        if self.camera_speed >= self.au_in_meters:
            return f'{self.camera_speed / self.au_in_meters:.2f} AU/s'
        speed_ratio = self.camera_speed / self.speed_of_light
        if speed_ratio >= 0.99:
            return f'{speed_ratio:.2f} C'
        return f'{self.camera_speed:.2e} m/s'

    def _update_speed_text(self):
        if self.speed_text is not None:
            self.speed_text.SetText(0, f'Camera speed: {self._format_camera_speed()}')

    def increase_camera_speed(self):
        # Speed controls are multiplicative so they stay useful across huge scales.
        self.camera_speed *= self.camera_speed_step
        self._update_speed_text()
        if self.plotter is not None:
            self.plotter.render()

    def decrease_camera_speed(self):
        self.camera_speed /= self.camera_speed_step
        self._update_speed_text()
        if self.plotter is not None:
            self.plotter.render()

    def toggle_pause(self):
        # Pausing freezes physics but still lets the user move the camera.
        self.is_paused = not self.is_paused
        self._update_status_text()
        self._force_surface_rendering()
        if self.plotter is not None:
            self.plotter.render()

    def sync_visuals(self):
        if self.central_actor is not None:
            sun_render_position = self._render_position([0.0, 0.0, 0.0])
            self.central_actor.SetPosition(*sun_render_position)

        # Reposition each light ray relative to current camera position
        for actor, center, fate in self.light_rays:
            render_pos = self._render_position(center)
            actor.SetPosition(*render_pos)

    def update_frame(self, _caller=None, _event=None):
        # Timer callback: step physics, move camera, sync meshes, then render.
        frame_dt = self.timer_interval / 1000.0
        if not self.is_paused:
            self.update_all()
        self.update_camera(frame_dt)
        self.sync_visuals()
        self._update_status_text()
        self._update_speed_text()
        self._force_surface_rendering()
        self.plotter.render()

    def run(self):
        # Start the render window and attach the repeating timer loop.
        if self.plotter is None:
            self.setup_scene()

        self.plotter.iren.initialize()
        self.plotter.iren.add_observer('TimerEvent', self.update_frame)
        self.plotter.iren.create_timer(self.timer_interval)
        self.plotter.show()