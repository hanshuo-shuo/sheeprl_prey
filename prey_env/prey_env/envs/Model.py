import time
from threading import Thread
from cellworld import *
from Agent import Agent, AgentData, AgentAction
import gc
import matplotlib.pyplot as plt
class Model:

    def __init__(self,
                 pworld: World,
                 freq: int = 100,
                 real_time: bool = False):
        self.real_time = real_time
        self.world = pworld
        self.agents = dict()
        self.agents_data = dict()
        self.display = Display(self.world, animated=True, fig_size=(6, 6))
        self.thread = None
        self.running = False
        self.interval = 1 / freq
        self.arena_polygon = Polygon(self.world.implementation.space.center, 6,
                                     self.world.implementation.space.transformation.size / 2,
                                     self.world.implementation.space.transformation.rotation)
        self.occlusions_polygons = Polygon_list.get_polygons([c.location for c in self.world.cells.occluded_cells()],
                                                             6,
                                                             self.world.implementation.cell_transformation.size / 2 * 1.05,
                                                             self.world.implementation.cell_transformation.rotation)
        self.visibility = Location_visibility(occlusions=self.occlusions_polygons)

    def is_valid_location(self, plocation):
        if self.arena_polygon.contains(plocation):
            for p in self.occlusions_polygons:
                if p.contains(plocation):
                    return False
            return True
        else:
            return False

    def run(self):
        self.running = True
        self.thread = Thread(target=self.__process__)
        self.thread.start()

    def __move_agent__(self, agent_name):
        agent = self.agents_data[agent_name]
        agent.theta = normalize(agent.theta + agent.turning_speed * self.interval)
        new_location = Location(agent.location.x, agent.location.y)
        new_location.move(theta=agent.theta, dist=agent.speed * self.interval)
        if self.is_valid_location(new_location):
            self.agents_data[agent_name].location = new_location

    def stop(self):
        if self.thread:
            self.running = False
            self.thread.join()

    def __process__(self):
        t = Timer(self.interval)
        while self.running:
            t.reset()
            self.step()
            if self.real_time:
                pending_wait = self.interval - t.to_seconds()
                if pending_wait > 0:
                    time.sleep(pending_wait)


    def step(self):
        for agent_name in self.agents.keys():
            if self.agents_data[agent_name].auto_update:
                action = self.agents[agent_name].get_action(self.get_observation(agent_name))
                self.agents_data[agent_name].speed = action.speed
                self.agents_data[agent_name].turning_speed = action.turning_speed
        for agent_name in self.agents.keys():
            self.__move_agent__(agent_name)

    def set_agent_action(self, agent_name: str, action: AgentAction):
        self.agents_data[agent_name].speed = action.speed
        self.agents_data[agent_name].turning_speed = action.turning_speed

    def __create_observation__(self, agent_name: str) -> dict:
        observation = dict()
        src = self.agents_data[agent_name].location
        for dst_agent_name in self.agents_data:
            dst_agent_data = self.agents_data[dst_agent_name]
            if self.visibility.is_visible(src, dst_agent_data.location):
                observation[dst_agent_name] = dst_agent_data
            else:
                observation[dst_agent_name] = None
        return observation

    def get_observation(self, agent_name: str) -> dict:
        return self.__create_observation__(agent_name)

    def show(self):
        for agent_name in self.agents_data:
            agent_data = self.agents_data[agent_name]
            self.display.agent(agent_name=agent_name,
                               location=agent_data.location,
                               rotation=to_degrees(agent_data.theta),
                               color=agent_data.color,
                               size=15, # show_trajectory=False
                               )
        self.display.update()

    def set_agent_position(self, pagent_name: str,
                           plocation: Location,
                           ptheta: float):
        self.agents_data[pagent_name].location = plocation
        self.agents_data[pagent_name].theta = ptheta

    def add_agent(self,
                  pagent_name: str,
                  pagent: Agent,
                  plocation: Location,
                  ptheta: float,
                  pcolor: str = "b",
                  pauto_update: bool = True):
        self.agents[pagent_name] = pagent
        self.agents_data[pagent_name] = AgentData(plocation=plocation,
                                                  ptheta=ptheta,
                                                  pspeed=0,
                                                  pturning_speed=0,
                                                  pcolor=pcolor,
                                                  pauto_update=pauto_update)
        self.display.set_agent_marker(pagent_name, Agent_markers.arrow())

    def clear_memory(self):
        # Stop any running threads
        self.stop()

        # Clear agents and agent data
        self.agents.clear()
        self.agents_data.clear()

        # Delete display object
        del self.display

        # Close any matplotlib figures
        plt.clf()
        plt.close('all')

        # Nullify references
        self.world = None
        self.thread = None
        self.arena_polygon = None
        self.occlusions_polygons = None
        self.visibility = None

        # Invoke garbage collector to free up memory
        gc.collect()