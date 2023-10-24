from cellworld import Location
from myPaths import *
from random import choice
from Agent import *


class Predator(Agent):
    def __init__(self,
                 pworld: World,
                 ppath_builder: Paths_builder,
                 pvisibility: Location_visibility,
                 pP_value: float = .1,
                 pI_value: float = .1,
                 pD_value: float = .1,
                 pmax_speed: float = .4,
                 pmax_turning_speed: float = .2):
        self.world = pworld
        self.paths = myPaths(ppath_builder, pworld)
        self.visibility = pvisibility
        self.destination = None
        self.destination_cell = None
        self.P_value = pP_value
        self.I_value = pI_value
        self.D_value = pD_value
        self.last_theta = None
        self.accum_theta_error = 0
        self.max_speed = pmax_speed
        self.max_turning_speed = pmax_turning_speed

    def __update_destination__(self, predator_location: Location, prey_location: Location = None):
        if self.destination_cell and \
                predator_location.dist(self.destination_cell.location) < self.world.implementation.cell_transformation.size / 2:
            self.destination_cell = None
        if prey_location:
            self.destination_cell = self.world.cells[self.world.cells.find(prey_location)]
        if self.destination_cell is None:
            hidden_cells = Cell_group()
            for c in self.world.cells:
                if c.occluded:
                    continue
                if not self.visibility.is_visible(predator_location, c.location):
                    hidden_cells.append(c)
            self.destination_cell = choice(hidden_cells)
        predator_cell = self.world.cells[self.world.cells.find(predator_location)]
        self.path = self.paths.get_path(predator_cell, self.destination_cell)
        for cd in self.path:
            if self.visibility.is_visible(predator_location, cd.location):
                self.destination = cd.location

    @staticmethod
    def normalized_error(theta_error: float):
        pi_err = math.pi * theta_error / 2
        return 1 / (pi_err * pi_err + 1)

    def get_action(self, observation: dict) -> AgentAction:
        prey = observation["prey"]
        predator = observation["predator"]
        if prey:
            self.__update_destination__(predator.location, prey.location)
        else:
            self.__update_destination__(predator.location)
        desired_theta = predator.location.atan(self.destination)
        theta_error, direction = angle_difference(predator.theta, desired_theta)
        dist_error = predator.location.dist(self.destination)
        self.accum_theta_error += theta_error
        turning_speed_P = theta_error * self.P_value
        turning_speed_D = 0
        if self.last_theta is not None:
            turning_speed_D = (self.last_theta - predator.theta) * self.D_value
        turning_speed_I = self.accum_theta_error * self.I_value
        turning_speed = turning_speed_P - turning_speed_D + turning_speed_I
        if turning_speed > self.max_turning_speed:
            turning_speed = self.max_turning_speed
        turning_speed = turning_speed * (-direction)
        speed = self.normalized_error(theta_error) * (1 + dist_error)
        if speed > self.max_speed:
            speed = self.max_speed
        self.last_theta = predator.theta
        return AgentAction(speed, turning_speed)
