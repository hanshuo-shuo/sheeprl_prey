import time
from Environment import Environment
from Agent import Agent, AgentAction
from cellworld import *

#creates the environment

e = Environment("16_09", freq=100, has_predator=True, real_time=True)

counter = 0
for i in range(2):
    #runs the environment
    e.start()

    #loops until the predator captures the prey or the prey reaches the goal
    while not e.complete:
        #starts a timer
        t = Timer()
        #reads an observation from the environment.
        #sets an action for the prey
        #speed float in habitat lenghts per second.
        #turning float in radians per second
        pre_o = e.get_observation()
        e.set_action(.1, .1)
        e.step()
        post_o = e.get_observation()
        counter += 1
        if counter % 1 == 0:
            e.show()
            print(pre_o, post_o)
        #computes the remaining time for 1/10 of a second to make the action interval consistent.
        # observation format: Tuple
        # [prey location, prey theta, goal location, predator location, predator theta, captured, goal_reached]
        # prey location: Type Location
        # prey theta: Type float in radians
        # goal location: Type Location
        # predator location: Type Location (None when predator is not visible)
        # predator theta: Type float in radians (None when predator is not visible)
        # captured: Type boolean : Prey has been captured by the predator. environment.complete becomes true.
        # goal_reached : Type boolean : Prey reached the goal location. environment.complete becomes true.

    #stops the environment
    e.stop()

