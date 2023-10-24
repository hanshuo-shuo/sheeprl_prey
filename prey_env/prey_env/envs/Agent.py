from cellworld import *


class AgentAction(JsonObject):
    def __init__(self, speed: float = 0, turning_speed: float = 0):
        self.speed = speed
        self.turning_speed = turning_speed*10


class AgentData(JsonObject):

    def __init__(self,
                 plocation: Location,
                 ptheta: float,
                 pspeed: float,
                 pturning_speed: float,
                 pcolor: str = "b",
                 pauto_update=True):
        self.location = plocation
        self.theta = ptheta
        self.speed = pspeed
        self.turning_speed = pturning_speed
        self.color = pcolor
        self.auto_update = pauto_update


class Agent:

    def get_action(self, observation: dict) -> AgentAction:
        return AgentAction(0, 0)
