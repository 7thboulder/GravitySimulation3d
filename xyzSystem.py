import numpy as np
import pyvista as pv


class System:
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
        self.listOfInteractingBodies = interacting_bodies

        self.G = 6.6743e-11
        self.dt = dt

        self.requiresCentral = central_mass != 0.0
        self.centralMass = central_mass
        self.central_radius = central_radius
        self.central_color = central_color

        self.plotter = None
        self.timer_interval = timer_interval
        self.render_scale = render_scale
        self.pressed_keys = set()

        self.camera_position = np.array([0.0, -3.0 * render_scale, render_scale], dtype=float)
        self.yaw = np.pi / 2.0
        self.pitch = -0.2
        self.camera_speed = camera_speed
        self.rotation_speed = rotation_speed

        self.central_actor = None
        self.body_actors = {}
        self.label_points = None
        self.label_names = []
        self.is_paused = False
        self.status_text = None
        self.speed_text = None
        self.trail_meshes = {}
        self.trail_length = 50000
        self.camera_speed_step = 1.5

    def _as_3d_vector(self, vector):
        vector_array = np.array(vector, dtype=float)
        padded_vector = np.zeros(3, dtype=float)
        padded_vector[:min(3, len(vector_array))] = vector_array[:3]
        return padded_vector

    def _body_position_3d(self, body):
        return self._as_3d_vector(body.get_position())

    def _camera_basis(self):
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
        return (self._as_3d_vector(real_position) - self.camera_position) / self.render_scale

    def _camera_key_map(self):
        return {
            'Left': ('yaw', 1.0),
            'Right': ('yaw', -1.0),
            'Up': ('pitch', 1.0),
            'Down': ('pitch', -1.0),
        }

    def _movement_key_map(self):
        return {
            'w': ('forward', 1.0),
            's': ('forward', -1.0),
            'a': ('right', -1.0),
            'd': ('right', 1.0),
            'q': ('up', 1.0),
            'e': ('up', -1.0),
        }

    def _set_camera_view(self):
        forward, _, up = self._camera_basis()
        self.plotter.camera.position = (0.0, 0.0, 0.0)
        self.plotter.camera.focal_point = tuple(forward * 10.0)
        self.plotter.camera.up = tuple(up)

    def initialize_camera(self):
        if self.plotter is None:
            raise RuntimeError('Call setup_scene before initializing the camera.')
        self._set_camera_view()

    def _on_key_press(self, obj, _event):
        self.pressed_keys.add(obj.GetKeySym())

    def _on_key_release(self, obj, _event):
        self.pressed_keys.discard(obj.GetKeySym())

    def bind_inputs(self):
        if self.plotter is None:
            raise RuntimeError('Call setup_scene before binding inputs.')

        interactor = self.plotter.iren.interactor
        interactor.AddObserver('KeyPressEvent', self._on_key_press)
        interactor.AddObserver('KeyReleaseEvent', self._on_key_release)
        self.plotter.add_key_event('space', self.toggle_pause)
        self.plotter.add_key_event('p', self.toggle_pause)
        self.plotter.add_key_event('bracketright', self.increase_camera_speed)
        self.plotter.add_key_event('bracketleft', self.decrease_camera_speed)

    def _create_body_visual(self, body):
        body_mesh = pv.Sphere(radius=max(body.get_radius() / self.render_scale, 0.02))
        actor = self.plotter.add_mesh(body_mesh, color=body.get_color(), name=body.get_name())
        body.assign_body_visuals(actor)
        self.body_actors[body.get_name()] = actor

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

        central_mesh = pv.Sphere(radius=max(self.central_radius / self.render_scale, 0.03))
        self.central_actor = self.plotter.add_mesh(central_mesh, color=self.central_color, name='central-body')

    def _setup_labels(self):
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
        self.plotter = plotter if plotter is not None else pv.Plotter()
        self.plotter.set_background('black')
        self.plotter.add_axes()
        self.plotter.add_text(
            'Arrow keys: look | W/A/S/D/Q/E: move | [ / ]: camera speed | Space/P: pause',
            position='upper_left',
            font_size=10,
        )
        self.status_text = self.plotter.add_text('Running', position='upper_right', font_size=10)
        self.speed_text = self.plotter.add_text('', position='lower_left', font_size=10)

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
        dist = np.linalg.norm(pos1)
        if dist == 0.0:
            return np.zeros_like(pos1, dtype=float)
        ag = -((self.G * mass_val) / dist ** 3) * pos1
        return ag

    def update_all(self, _frame=None):
        for body in self.listOfInteractingBodies:
            total_acc = np.zeros_like(body.get_position(), dtype=float)
            if self.requiresCentral:
                total_acc += self.get_single_body_acceleration(body.get_position(), self.centralMass)

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
        for key_sym, (axis_name, direction) in self._camera_key_map().items():
            if key_sym not in self.pressed_keys:
                continue
            if axis_name == 'yaw':
                self.yaw += direction * self.rotation_speed * dt
            elif axis_name == 'pitch':
                self.pitch += direction * self.rotation_speed * dt

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
        actors = list(self.body_actors.values())
        if self.central_actor is not None:
            actors.append(self.central_actor)

        for actor in actors:
            actor.GetProperty().SetRepresentationToSurface()

    def _update_status_text(self):
        if self.status_text is not None:
            self.status_text.SetText(3, 'Paused' if self.is_paused else 'Running')

    def _update_speed_text(self):
        if self.speed_text is not None:
            self.speed_text.SetText(0, f'Camera speed: {self.camera_speed:.2e}')

    def increase_camera_speed(self):
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
        self.is_paused = not self.is_paused
        self._update_status_text()
        self._force_surface_rendering()
        if self.plotter is not None:
            self.plotter.render()

    def sync_visuals(self):
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
        trail_mesh = body.get_trail_visuals()
        history = body.get_position_history()
        if trail_mesh is None:
            return

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
        if self.plotter is None:
            self.setup_scene()

        self.plotter.iren.initialize()
        self.plotter.iren.add_observer('TimerEvent', self.update_frame)
        self.plotter.iren.create_timer(self.timer_interval)
        self.plotter.show()
