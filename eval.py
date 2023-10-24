import pathlib
from functools import partial

import gymnasium as gym
import torch
from lightning import Fabric
from omegaconf import OmegaConf

from sheeprl.algos.dreamer_v3.agent import PlayerDV3, build_models
from sheeprl.algos.dreamer_v3.utils import test_vis
from sheeprl.envs.wrappers import RestartOnException
from sheeprl.utils.env import make_env
import prey_env
import warnings
# This will ignore all UserWarnings, adjust if you want finer granularity
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    fabric = Fabric(accelerator="cpu", devices="1")
    ckpt_path = pathlib.Path("logs/runs/dreamer_v3/prey_d_1/2023-10-23_23-13-38_default_42/version_0/checkpoint/ckpt_400000_0.ckpt")
    cfg = OmegaConf.load("logs/runs/dreamer_v3/prey_d_1/2023-10-23_23-13-38_default_42/.hydra/config.yaml")
    ckpt = fabric.load(ckpt_path)

    ## SEED
    # fabric.seed_everything(cfg.seed)

    # Change configs as needed
    cfg.seed = 0
    cfg.env.num_envs = 1
    cfg.env.capture_video = False

    # Recreate a single environment
    rank = 0
    vectorized_env = gym.vector.SyncVectorEnv
    envs = vectorized_env(
        [
            partial(
                RestartOnException,
                env_fn=make_env(
                    cfg,
                    cfg.seed + rank * cfg.env.num_envs + i,
                    rank * cfg.env.num_envs,
                    None,
                    "",
                    vector_env_idx=i,
                ),
            )
            for i in range(1)
        ],
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space
    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = (
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    clip_rewards_fn = lambda r: torch.tanh(r) if cfg.clip_rewards else r
    cnn_keys = cfg.cnn_keys.encoder
    mlp_keys = cfg.mlp_keys.encoder
    fabric.print("CNN keys:", cnn_keys)
    fabric.print("MLP keys:", mlp_keys)
    obs_keys = cnn_keys + mlp_keys

    # Close the environment since it will be recreated inside the `test` function
    envs.close()

    # Create models and load weights from checkpoint
    world_model, actor, critic, target_critic = build_models(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        ckpt["world_model"],
        ckpt["actor"],
        ckpt["critic"],
        ckpt["target_critic"],
    )

    # Create the player agent
    player = PlayerDV3(
        world_model.encoder.module,
        world_model.rssm,
        actor.module,
        actions_dim,
        cfg.algo.player.expl_amount,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        cfg.algo.world_model.recurrent_model.recurrent_state_size,
        fabric.device,
        discrete_size=cfg.algo.world_model.discrete_size,
    )

    # Test the agent
    sample_actions = True  # Whether to sample actions from the actor's distribution or not
    test_vis(
        player,
        fabric,
        cfg,
        f"{cfg.env.id}_{cfg.seed}_sample_{sample_actions}",
        sample_actions=sample_actions,
    )