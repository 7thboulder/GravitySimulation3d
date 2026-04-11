import numpy as np
import pyvista as pv


class System:
    # Main controller for the simulation: physics stepping, rendering, camera, and UI.
    def __init__(
        self,
        interacting_bodies: list,
        dt: float,
        central_mass=0.0,
        central_radius=0.0,
        central_color='yellow',
        render_scale=1e10,
        timer_interval=16,
        camera_speed=2e10,
        rotation_speed=1.5,
    ):
        # Bodies whose positions are advanced each frame.
        self.listOfInteractingBodies = interacting_bodies

        # Physics constants and integration timestep.
        self.G = 6.6743e-11
        self.dt = dt

        # Optional fixed central mass lets planets orbit a sun without storing the sun as a body.
        self.requiresCentral = central_mass != 0.0
        self.centralMass = central_mass
        self.central_radius = central_radius
        self.central_color = central_color

        # Renderer state and real-space camera state.
        self.plotter = None
        self.timer_interval = timer_interval
        self.render_scale = render_scale
        self.pressed_keys = set()

        self.camera_position = np.array([0.0, -3.0 * render_scale, render_scale], dtype=float)
        self.yaw = np.pi / 2.0
        self.pitch = -0.2
        self.camera_speed = camera_speed
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

    def _create_body_visual(self, body):
        # Each planet gets a sphere actor and a polyline mesh for its trail.
        body_mesh = pv.Sphere(radius=max(body.get_radius() / self.render_scale, 0.02))
        actor = self.plotter.add_mesh(body_mesh, color=body.get_color(), name=body.get_name())
        body.assign_body_visuals(actor)
        self.body_actors[body.get_name()] = actor

        # Start each trail with one point so PyVista has valid geometry immediately.
        initial_point = np.array([self._render_position(body.get_position())], dtype=float)
        trail_mesh = pv.PolyData(initial_point)
        self.plotter.add_mesh(
            trail_mesh,
            color=body.get_color(),
            line_width=2,
            name=f'{body.get_name()}-trail',
            render_lines_as_tubes=False,
        )
        body.assign_trail_visuals(trail_mesh)
        self.trail_meshes[body.get_name()] = trail_mesh

    def _create_central_visual(self):
        if not self.requiresCentral:
            return

        # Render the central body separately from the list of orbiting bodies.
        central_mesh = pv.Sphere(radius=max(self.central_radius / self.render_scale, 0.03))
        self.central_actor = self.plotter.add_mesh(central_mesh, color=self.central_color, name='central-body')

    def _setup_labels(self):
        # Labels are stored in one shared point cloud so they can all be updated together.
        label_positions = []
        self.label_names = []

        if self.requiresCentral:
            label_positions.append(self._render_position([0.0, 0.0, 0.0]))
            self.label_names.append('Sun')

        for body in self.listOfInteractingBodies:
            label_positions.append(self._render_position(body.get_position()))
            self.label_names.append(body.get_name())

        self.label_points = pv.PolyData(np.array(label_positions, dtype=float))
        self.label_points['labels'] = self.label_names
        self.plotter.add_point_labels(
            self.label_points,
            'labels',
            font_size=14,
            text_color='white',
            shape_color='black',
            shape_opacity=0.35,
            margin=4,
            always_visible=True,
            show_points=False,
        )

    def setup_scene(self, plotter=None):
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
        for body in self.listOfInteractingBodies:
            self._create_body_visual(body)
            body.append_position_history(body.get_position())
        self._setup_labels()

        self.bind_inputs()
        self.initialize_camera()
        self._update_speed_text()
        self.sync_visuals()
        return self.plotter

    def get_single_body_acceleration(self, pos1, mass_val):
        # Newtonian point-mass gravity toward `mass_val` from the displacement vector `pos1`.
        dist = np.linalg.norm(pos1)
        if dist == 0.0:
            return np.zeros_like(pos1, dtype=float)
        ag = -((self.G * mass_val) / dist ** 3) * pos1
        return ag

    def update_all(self, _frame=None):
        # Advance all bodies by one simulation timestep.
        for body in self.listOfInteractingBodies:
            total_acc = np.zeros_like(body.get_position(), dtype=float)
            if self.requiresCentral:
                total_acc += self.get_single_body_acceleration(body.get_position(), self.centralMass)

            # Sum the contribution from every other body in the system.
            for body_to_compare in self.listOfInteractingBodies:
                if body is body_to_compare:
                    continue
                total_acc += self.get_single_body_acceleration(
                    body.get_position() - body_to_compare.get_position(),
                    body_to_compare.get_mass(),
                )

            body.set_velocity(self.dt * total_acc)
            body.set_position(body.get_velocity() * self.dt)
            body.append_x_history()
            body.append_y_history()

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
        # Push the current simulation state into every render object.
        label_positions = []

        if self.central_actor is not None:
            sun_render_position = self._render_position([0.0, 0.0, 0.0])
            self.central_actor.SetPosition(*sun_render_position)
            label_positions.append(sun_render_position)

        for body in self.listOfInteractingBodies:
            render_position = self._render_position(body.get_position())
            body.set_body_visuals(render_position)
            if not self.is_paused:
                body.append_position_history(body.get_position())
                if len(body.get_position_history()) > self.trail_length:
                    body.position_history = body.get_position_history()[-self.trail_length:]
            self._update_trail(body)
            label_positions.append(render_position)

        if self.label_points is not None:
            self.label_points.points = np.array(label_positions, dtype=float)

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

    def _update_trail(self, body):
        # Rebuild the visible polyline from the stored world-space trail history.
        trail_mesh = body.get_trail_visuals()
        history = body.get_position_history()
        if trail_mesh is None:
            return

        # One point is valid geometry, but it is not yet a visible line.
        if len(history) < 2:
            trail_mesh.points = (
                np.array([self._render_position(point) for point in history], dtype=float)
                if history
                else np.empty((0, 3), dtype=float)
            )
            trail_mesh.lines = np.empty(0, dtype=np.int64)
            return

        points = np.array([self._render_position(point) for point in history], dtype=float)
        cell_count = len(points) - 1
        line_cells = np.empty(cell_count * 3, dtype=np.int64)
        line_cells[0::3] = 2
        line_cells[1::3] = np.arange(0, cell_count, dtype=np.int64)
        line_cells[2::3] = np.arange(1, len(points), dtype=np.int64)
        trail_mesh.points = points
        trail_mesh.lines = line_cells

    def run(self):
        # Start the render window and attach the repeating timer loop.
        if self.plotter is None:
            self.setup_scene()

        self.plotter.iren.initialize()
        self.plotter.iren.add_observer('TimerEvent', self.update_frame)
        self.plotter.iren.create_timer(self.timer_interval)
        self.plotter.show()
