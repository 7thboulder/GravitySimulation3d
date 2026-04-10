import numpy as np
import pyvista as pv


class SpaceCamera:
    def __init__(self, plotter, scale=1e10):
        self.pl = plotter
        self.scale = scale

        # ------------------------
        # CAMERA STATE (REAL SPACE)
        # ------------------------
        self.position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.pitch = 0.0
        self.speed = 1e9

        # input
        self.keys = set()

        # objects: store REAL positions + mesh
        self.objects = []

        # init VTK interactor
        self.pl.iren.initialize()

        # input system (robust)
        self.pl.iren.add_observer("KeyPressEvent", self._on_key_press)
        self.pl.iren.add_observer("KeyReleaseEvent", self._on_key_release)

        # timer loop
        self.pl.iren.add_observer("TimerEvent", self.update)
        self.pl.iren.create_timer(16)

    # ------------------------
    # ADD OBJECT
    # ------------------------
    def add_object(self, mesh, real_position):
        self.objects.append({
            "mesh": mesh,
            "real": np.array(real_position, dtype=float)
        })

    # ------------------------
    # INPUT
    # ------------------------
    def _on_key_press(self, obj, event):
        self.keys.add(obj.GetKeySym())

    def _on_key_release(self, obj, event):
        self.keys.discard(obj.GetKeySym())

    # ------------------------
    # DIRECTION VECTORS
    # ------------------------
    def get_directions(self):
        forward = np.array([
            np.cos(self.pitch) * np.cos(self.yaw),
            np.cos(self.pitch) * np.sin(self.yaw),
            np.sin(self.pitch)
        ])
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)

        return forward, right, up

    # ------------------------
    # MOVEMENT
    # ------------------------
    def move(self, dt):
        forward, right, up = self.get_directions()

        if "i" in self.keys:
            self.position += forward * self.speed * dt
        if "k" in self.keys:
            self.position -= forward * self.speed * dt
        if "j" in self.keys:
            self.position -= right * self.speed * dt
        if "l" in self.keys:
            self.position += right * self.speed * dt
        if "q" in self.keys:
            self.position += up * self.speed * dt
        if "e" in self.keys:
            self.position -= up * self.speed * dt

    # ------------------------
    # UPDATE LOOP
    # ------------------------
    def update(self, *args):
        dt = 0.016

        # move camera in REAL space
        self.move(dt)

        forward, right, up = self.get_directions()

        # ------------------------
        # FLOATING ORIGIN RENDER
        # ------------------------
        for obj in self.objects:
            render_pos = (obj["real"] - self.position) / self.scale
            obj["mesh"].SetCenter(*render_pos)

        # ------------------------
        # CAMERA (stable model)
        # ------------------------
        self.pl.camera.position = (0, 0, 0)
        self.pl.camera.focal_point = forward * 1e6
        self.pl.camera.up = up

        self.pl.render()