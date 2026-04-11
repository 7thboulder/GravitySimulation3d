import numpy as np


class Planet:
    def __init__(
            self,
            name,
            velocity_vector,
            position_vector,
            mass,
            radius,
            color):
        self.name = name
        self.velocityVector = np.array(velocity_vector, dtype=float)
        self.positionVector = np.array(position_vector, dtype=float)
        self.mass = mass
        self.radius = radius
        self.x_history = []
        self.y_history = []
        self.position_history = []
        self.color = color

        self.visualBody = None
        self.visualTrail = None
        self.label = None

    def get_body_visuals(self):
        return self.visualBody

    def assign_body_visuals(self, visual_body):
        self.visualBody = visual_body

    def get_trail_visuals(self):
        return self.visualTrail

    def assign_trail_visuals(self, visual_trail):
        self.visualTrail = visual_trail

    def get_label(self):
        return self.label

    def assign_label(self, label):
        self.label = label

    def set_trail_visuals(self):
        if self.visualTrail is not None and self.position_history:
            self.visualTrail.points = np.array(self.position_history, dtype=float)

    def set_body_visuals(self, position=None):
        if self.visualBody is None:
            return

        target_position = self.positionVector if position is None else np.array(position, dtype=float)
        padded_position = np.zeros(3, dtype=float)
        padded_position[:min(3, len(target_position))] = target_position[:3]
        self.visualBody.SetPosition(*padded_position)

    def set_label(self):
        if self.label is not None:
            self.label.set_position((self.positionVector[0], self.positionVector[1]))

    def get_mass(self):
        return self.mass
    def get_velocity(self):
        return self.velocityVector
    def get_position(self):
        return self.positionVector
    def get_radius(self):
        return self.radius

    def set_mass(self, add_mass, set_mass = False):
        set_mass = set_mass
        if self.set_mass:
            self.mass = add_mass
        else:
            self.mass += add_mass

    def set_velocity(self, add_velocity, set_velocity = False):
        set_velocity = set_velocity
        if set_velocity:
            self.velocityVector = add_velocity
        else:
            self.velocityVector += add_velocity

    def set_position(self, add_position, set_position = False):
        set_position = set_position
        if set_position:
            self.positionVector = add_position
        else:
            self.positionVector += add_position

    def get_x_history(self):
        return self.x_history
    def get_y_history(self):
        return self.y_history

    def append_x_history(self):
        self.x_history.append(self.positionVector[0])

    def append_y_history(self):
        self.y_history.append(self.positionVector[1])

    def append_position_history(self, position=None):
        target_position = self.positionVector if position is None else np.array(position, dtype=float)
        padded_position = np.zeros(3, dtype=float)
        padded_position[:min(3, len(target_position))] = target_position[:3]
        self.position_history.append(padded_position)

    def get_position_history(self):
        return self.position_history

    def get_color(self):
        return self.color

    def get_name(self):
        return self.name
