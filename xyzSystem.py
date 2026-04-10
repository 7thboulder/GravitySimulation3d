import numpy as np
import pyvista as pv


import xyzPlanet


class System:

    def __init__(self, interacting_bodies: list, dt: float, central_mass=0.0, central_radius = 0.0, central_color = 'yellow'):
        self.listOfInteractingBodies = interacting_bodies

        self.G = 6.6743e-11
        self.dt = dt

        if central_mass == 0.0:
            self.requiresCentral = False
            self.centralMass = central_mass
            self.central_radius = central_radius
            self.central_color = central_color
        else:
            self.requiresCentral = True
            self.centralMass = central_mass
            self.central_radius = central_radius
            self.central_color = central_color

        self.visualsList = []
        self.visualTrailList = []
        self.visualBodyList = []
        self.labels = []


    def get_single_body_acceleration(self, pos1, mass_val):
        dist = np.linalg.norm(pos1)
        ag = -((self.G * mass_val) / dist ** 3) * pos1
        return ag


    def update_all(self, frame):
        for body in self.listOfInteractingBodies:
            total_acc = np.array([0.0, 0.0])
            if self.requiresCentral:
                total_acc += self.get_single_body_acceleration(body.get_position(), self.centralMass)
                for bodyToCompare in self.listOfInteractingBodies:
                    if body != bodyToCompare:
                        total_acc += self.get_single_body_acceleration(
                            body.get_position()-bodyToCompare.get_position(),
                            bodyToCompare.get_mass())
            else:
                for bodyToCompare in self.listOfInteractingBodies:
                    if body != bodyToCompare:
                        total_acc += self.get_single_body_acceleration(
                            body.get_position()-bodyToCompare.get_position(),
                            bodyToCompare.get_mass())

            body.set_velocity(self.dt * total_acc)
            body.set_position(body.get_velocity() * self.dt)
            body.append_x_history()
            body.append_y_history()
