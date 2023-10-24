from cellworld import *
import threading
import time

class Environment:

    def __init__(self, w: World = None, freq: int = 100):
        self.world = w
        self.prey_location = Location(.05, .5)
        self.prey_theta = 0
        self.prey_speed = 0
        self.prey_turning_speed = 0
        self.predator_location = Location(1, .5)
        self.running = False
        self.freq = freq
        self.arena_polygon = Polygon(self.world.implementation.space.center, 6,
                                     self.world.implementation.space.transformation.size/2,
                                     self.world.implementation.space.transformation.rotation)
        self.occlusions_polygons = Polygon_list()
        for cl in self.world.cells.occluded_cells():
            self.occlusions_polygons.append(Polygon(cl.location, 6,
                                     self.world.implementation.cell_transformation.size/2,
                                     self.world.implementation.cell_transformation.rotation))
        self.display = Display(self.world, animated=True)

    def get_observation(self) -> tuple:
        return self.prey_location, self.prey_theta, Location(1, .5)

    def set_action(self, speed: float, turning: float) -> None:
        self.prey_turning_speed = turning
        self.prey_speed = speed

    def run(self) -> None:
        self.running = True
        self.t = threading.Thread(target=self.__process__)
        self.t.start()

    def stop(self) -> None:
        self.running = False

    def __process__(self) -> None:
        while self.running:
            time.sleep(1 / self.freq)
            self.prey_theta += self.prey_turning_speed / self.freq
            new_prey_location = Location(self.prey_location.x, self.prey_location.y)
            new_prey_location.move(self.prey_theta, self.prey_speed / self.freq)
            if self.arena_polygon.contains(new_prey_location):
                for p in self.occlusions_polygons:
                    if p.contains(new_prey_location):
                        break
                else:
                    self.prey_location = new_prey_location

    def show(self) -> None:
        self.display.agent(location=self.prey_location, rotation=to_degrees(self.prey_theta), color="r", size=60)
        self.display.update()

#Loads the world from github library
world = World.get_from_parameters_names("hexagonal", "canonical", "21_05")
#Creates the environment using the world
environment = Environment(world)
#runs the environment
environment.run()
while True:
    #gets the observation
    observation = environment.get_observation()
    # Observation is a tuple with (Prey_location, Prey_orientation, Goal_location)

    #set the action
    environment.set_action(speed=.2, turning=.2)
    #speed in habitat lenghts per second.
    #turning in radians per second

    #shows the environment
    environment.show()

    #sleeps for .1 seconds
    time.sleep(.1)
