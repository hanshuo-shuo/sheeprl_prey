import gymnasium as gym
import prey_env
id = "prey_d_1"
import warnings
# This will ignore all UserWarnings, adjust if you want finer granularity
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    Env = gym.make(id)
    Env.reset()
    for i in range(50000):
        Env.step(Env.action_space.sample())
        Env.render()
        if i%5:
            Env.reset()
