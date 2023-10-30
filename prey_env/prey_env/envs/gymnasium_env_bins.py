from Predator import Predator
from Agent import *
from Model import Model
from myPaths import *
from cellworld import *
import random
from gymnasium import Env
from gymnasium import spaces
import numpy as np


class Environment(Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self,
                 e: int = 2,
                 freq: int = 100,
                 has_predator = True,
                 real_time: bool = False,
                 prey_agent: Agent = None,
                 max_step: int = 300,
                 predator_speed: float = 0.2,
                 env_type: str = "train",
                 env_random: bool = False,
                 penalty: int = -50,
                 reward: int = 100,
                 render_mode = None,
                 action_noise: bool = False):
        if env_type == "train":
            world_name = "%02i_%02i" % (random.randint(0, 10), e)
        elif env_type == "test":
            world_name = "%02i_%02i" % (random.randint(11, 19), e)
        self.freq = freq
        self.penalty = penalty
        self.reward = reward
        self.real_time = real_time
        self.prey_agent = prey_agent
        self.env_type = env_type
        self.env_random = env_random
        self.action_noise = action_noise
        self.e = e
        self.world = World.get_from_parameters_names("hexagonal", "canonical", world_name)
        self.model = Model(pworld=self.world, freq=self.freq, real_time=self.real_time)
        self.goal_location = Location(1, .5)
        self.start_location = Location(0, .5)
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        self.action_space = spaces.Discrete(100)
        self.has_predator = has_predator
        self.max_step = max_step
        self.current_step = 0
        self.episode_reward_history = []
        self.current_episode_reward = 0
        self.predator = None
        self.predator_speed = predator_speed
        self.goal_threshold = self.world.implementation.cell_transformation.size
        self.capture_threshold = self.world.implementation.cell_transformation.size
        self.goal_area = self.model.display.circle(location=self.goal_location,
                                                   color="g",
                                                   alpha=.5,
                                                   radius=self.goal_threshold)

    def is_goal_reached(self, prey_location: Location):
        return prey_location.dist(self.goal_location) <= self.goal_threshold

    def is_prey_captured(self, prey_location: Location, predator_location: Location):
        return prey_location.dist(predator_location) <= self.capture_threshold


    def get_observation(self) -> tuple:
        o = self.model.get_observation("prey")
        prey = o["prey"]
        goal_reached = self.is_goal_reached(prey.location)
        if goal_reached:
            self.complete = True

        # occlusions = Location_list([c.location for c in self.world.cells if c.occluded])
        occlusions = Location_list([(c.location.dist(prey.location), c.location) for c in self.world.cells if c.occluded])
        occlusions.sort(key=lambda a: a[0])

        def dif_dir(a,b):
            diff, direc = angle_difference(a, b)
            return diff * direc

        max_number_of_occlusions_in_observation = 3
        closest_occlusions = [(c[0], dif_dir(prey.location.atan(c[1]), prey.theta)) for c in occlusions[:min(len(occlusions), max_number_of_occlusions_in_observation)]]

        if self.has_predator:
            if o["predator"]:
                predator = o["predator"]
                captured = self.is_prey_captured(prey.location, predator.location)
                if captured:
                    self.complete = True
                return prey.location, \
                       prey.theta, \
                       self.goal_location, \
                       predator.location, \
                       predator.theta, \
                       captured, \
                       goal_reached, \
                       closest_occlusions
            else:
                return prey.location, \
                       prey.theta, \
                       self.goal_location, \
                       None, \
                       None, \
                       False, \
                       goal_reached, \
                       closest_occlusions
        else:
            return prey.location, prey.theta, self.goal_location, goal_reached, closest_occlusions


    def is_complete(self):
        return self.complete


    def run(self) -> None:
        self.start()
        self.model.run()

    def start(self) -> None:
        self.complete = False
        self.model.set_agent_position("prey", self.start_location, math.pi / 2)
        if self.has_predator:
            predator_location = random.choice(self.spawn_locations)
            predator_theta = math.pi * 2 * random.random()
            self.model.set_agent_position("predator", predator_location, predator_theta)

    def map_discrete_to_continuous(self, discrete_val, n_bins=10):
        # Convert the single discrete value into two discrete values
        # assuming a 10x10 grid
        row = discrete_val // n_bins  # Integer division to get the row
        col = discrete_val % n_bins  # Modulus to get the column
        # Now map each discrete value to the continuous range [-1, 1]
        bin_width = 2.0 / (n_bins - 1)
        continuous_row = -1 + row * bin_width
        continuous_col = -1 + col * bin_width


        if self.action_noise:
            noise_level, action_shape = 0.5, 2
            rand_noise = np.random.randn(action_shape) * noise_level
            continuous_row += rand_noise[0]
            continuous_col += rand_noise[1]

        return continuous_row, continuous_col

    def set_action(self, speed: float, turning: float) -> None:
        action = AgentAction(speed, turning)
        self.model.set_agent_action("prey", action)

    def step(self, action):
        reward = 0
        info = {}
        done, truncated = False, False
        speed, turning = self.map_discrete_to_continuous(action)
        self.set_action(speed, turning)
        if self.has_predator:
            self.model.step()
            prey_location, prey_theta, \
                goal_location, pred_location, \
                pred_theta, captured, \
                goal_reached, closest_occlusions = self.get_observation()
            closest_distance, closest_angle = closest_occlusions[0]
            se_closest_distance, se_closest_angle = closest_occlusions[1]
            th_closest_distance, th_closest_angle = closest_occlusions[2]
            if pred_location is not None:
                obs = np.array(
                    [prey_location.x, prey_location.y, prey_theta,
                     speed, turning,
                     pred_location.x, pred_location.y, pred_theta,
                     closest_distance, closest_angle,
                     se_closest_distance, se_closest_angle,
                     th_closest_distance, th_closest_angle], dtype=np.float32)
            else:
                obs = np.array(
                    [prey_location.x, prey_location.y, prey_theta,
                     speed, turning,
                     -1.0, -1.0, 0.0,
                     closest_distance, closest_angle,
                     se_closest_distance, se_closest_angle,
                     th_closest_distance, th_closest_angle], dtype=np.float32)
            dx, dy = abs(obs[0] - 1), abs(obs[1] - 0.5)
            d = math.sqrt(dx ** 2 + dy ** 2)
            # reach the goal
            if self.is_goal_reached(prey_location):
                reward = self.reward
                done = True
            else:
                reward = -d

            if captured:
                truncated = True
                reward = self.penalty
            info = {"is success": done, "is truncated": truncated}
            self.current_step += 1
            if self.current_step > self.max_step:
                truncated = True
            self.current_episode_reward += reward
            if done or truncated:
                self.episode_reward_history.append(self.current_episode_reward)
                self.current_episode_reward = 0
        else:
            self.model.step()
            location, theta, goal_location, goal_reached, closest_occlusions = self.get_observation()
            closest_distance,  closest_angle = closest_occlusions[0]
            se_closest_distance, se_closest_angle = closest_occlusions[1]
            th_closest_distance, th_closest_angle = closest_occlusions[2]
            obs = np.array(
                [location.x, location.y, theta,
                 speed, turning,
                 -1.0, -1.0, 0,
                 closest_distance, closest_angle,
                 se_closest_distance, se_closest_angle,
                 th_closest_distance, th_closest_angle],dtype=np.float32)
            dx = abs(obs[0] - 1)
            dy = abs(obs[1] - 0.5)
            d = math.sqrt(dx ** 2 + dy ** 2)
            if self.is_goal_reached(location):
                reward = self.reward
                done = True
            else:
                reward = -d
            info = {"is success": done}
            self.current_step += 1
            if self.current_step > self.max_step:
                truncated = True
            self.current_episode_reward += reward
            if done or truncated:
                self.episode_reward_history.append(self.current_episode_reward)
                self.current_episode_reward = 0
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options = None):
        self.complete = True
        self.model.clear_memory()
        e = self.e  # Assuming 'e' is an instance variable
        if self.env_type == "train":
            world_name = "%02i_%02i" % (random.randint(0, 10), e)
        else:
            world_name = "%02i_%02i" % (random.randint(11, 19), e)
        #occlusions = Cell_group_builder.get_from_name("hexagonal", world_name, "occlusions")
        #self.world.set_occlusions(occlusions)
        self.world = World.get_from_parameters_names("hexagonal", "canonical", world_name)
        #self.model.world=self.world
        self.model = Model(pworld=self.world, freq=self.freq, real_time=self.real_time)
        self.goal_location = Location(1, .5)
        self.start_location = Location(0, .5)
        self.model.add_agent("prey",
                             self.prey_agent,
                             Location(0, 0), 0, "b",
                             pauto_update=self.prey_agent is not None)
        self.goal_threshold = self.world.implementation.cell_transformation.size
        self.capture_threshold = self.world.implementation.cell_transformation.size
        self.goal_area = self.model.display.circle(location=self.goal_location,
                                                   color="g",
                                                   alpha=.5,
                                                   radius=self.goal_threshold)
        if self.env_random:
            random_number = random.random()
            self.has_predator = random_number > 0.5
        if self.has_predator:
            paths_builder = Paths_builder.get_from_name("hexagonal", world_name)
            self.predator = Predator(self.world,
                                     ppath_builder=paths_builder,
                                     pvisibility=self.model.visibility,
                                     pP_value=2,
                                     pI_value=0,
                                     pD_value=0,
                                     pmax_speed=self.predator_speed,
                                     pmax_turning_speed=math.pi)

            self.spawn_locations = Location_list()
            for c in self.world.cells.free_cells():
                if not self.model.visibility.is_visible(self.start_location, c.location):
                    self.spawn_locations.append(c.location)
            self.model.add_agent("predator", self.predator, Location(0, 0), 0, "r")
            self.predator_destination = self.model.display.circle(location=Location(0,0),
                                                                  color="b",
                                                                  alpha=.5,
                                                                  radius=0.01)
            self.predator_destination_cell = self.model.display.circle(location=Location(0,0),
                                                                       color="g",
                                                                       alpha=.5,
                                                                       radius=0.02)
            self.predator_capture_area = self.model.display.circle(location=Location(0,0),
                                                                   color="r",
                                                                   alpha=.5,
                                                                   radius=self.capture_threshold)
            # start has to be here, after all the setup()
            self.start()
            prey_location, prey_theta, \
            goal_location, pred_location, \
            pred_theta, captured, \
            goal_reached, closest_occlusions = self.get_observation()
            closest_distance, closest_angle = closest_occlusions[0]
            se_closest_distance, se_closest_angle = closest_occlusions[1]
            th_closest_distance, th_closest_angle = closest_occlusions[2]
            if pred_location is not None:
                obs = np.array(
                    [prey_location.x, prey_location.y, prey_theta,
                     0.0, 0.0,
                     pred_location.x, pred_location.y, pred_theta,
                     closest_distance, closest_angle,
                     se_closest_distance, se_closest_angle,
                     th_closest_distance, th_closest_angle], dtype=np.float32)
            else:
                obs = np.array([prey_location.x, prey_location.y, prey_theta,
                                0.0, 0.0,
                                -1.0, -1.0, 0.0,
                                closest_distance, closest_angle,
                                se_closest_distance, se_closest_angle,
                                th_closest_distance, th_closest_angle], dtype=np.float32)
        else:
            self.start()
            location, theta, goal_location, goal_reached, closest_occlusions = self.get_observation()
            closest_distance,  closest_angle = closest_occlusions[0]
            se_closest_distance, se_closest_angle = closest_occlusions[1]
            th_closest_distance, th_closest_angle = closest_occlusions[2]
            obs = np.array(
                [location.x, location.y, theta,
                 0.0, 0.0,
                 -1.0, -1.0, 0.0,
                 closest_distance, closest_angle,
                 se_closest_distance, se_closest_angle,
                 th_closest_distance, th_closest_angle],
                dtype=np.float32)
        self.current_step = 1
        self.stop()
        return obs, {}

    def stop(self) -> None:
        self.model.stop()

    def show(self) -> None:
        if self.predator:
            if self.predator.destination:
                self.predator_destination.set(center=(self.predator.destination.x, self.predator.destination.y), color="b")
            if self.predator.destination_cell:
                self.predator_destination_cell.set(center=(self.predator.destination_cell.location.x, self.predator.destination_cell.location.y), color="g")
            self.predator_capture_area.set(center=(self.model.agents_data["predator"].location.x, self.model.agents_data["predator"].location.y), color="r")
        self.model.show()

    def render(self):
        return self.show()

    def seed(self, seed=123):
        self.np_random = np.random.default_rng(seed)

    def recorded_frames(self):
        pass