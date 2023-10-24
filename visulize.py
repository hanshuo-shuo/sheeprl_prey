from functools import partial
import hydra
import os
from typing import TYPE_CHECKING, Any, Dict

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v3.agent import PlayerDV3
from sheeprl.algos.dreamer_v3.agent import PlayerDV3, build_models
from sheeprl.envs.wrappers import RestartOnException
from omegaconf import DictConfig, OmegaConf
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict, print_config
# this one is needed because is our prey env
import prey_env

import warnings
# This will ignore all UserWarnings, adjust if you want finer granularity
warnings.filterwarnings("ignore", category=UserWarning)


@torch.no_grad()
def test(
    player: "PlayerDV3",
    fabric: Fabric,
    cfg: Dict[str, Any],
    test_name: str = "",
    sample_actions: bool = False,
):
    log_dir = fabric.logger.log_dir if len(fabric.loggers) > 0 else os.getcwd()
    env: gym.Env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name != "" else ""))()

    done = False
    cumulative_rew = 0
    device = 'cpu'
    next_obs = env.reset(seed=cfg.seed)[0]
    for k in next_obs.keys():
        next_obs[k] = torch.from_numpy(next_obs[k]).view(1, *next_obs[k].shape).float()
    player.num_envs = 1
    player.init_states()
    while not done:
        # Act greedly through the environment
        preprocessed_obs = {}
        for k, v in next_obs.items():
            if k in cfg.cnn_keys.encoder:
                preprocessed_obs[k] = v[None, ...].to(device) / 255
            elif k in cfg.mlp_keys.encoder:
                preprocessed_obs[k] = v[None, ...].to(device)
        real_actions = player.get_greedy_action(
            preprocessed_obs, sample_actions, {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
        )
        if player.actor.is_continuous:
            real_actions = torch.cat(real_actions, -1).cpu().numpy()
        else:
            real_actions = np.array([real_act.cpu().argmax(dim=-1).numpy() for real_act in real_actions])

        # Single environment step
        next_obs, reward, done, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
        env.render()
        for k in next_obs.keys():
            next_obs[k] = torch.from_numpy(next_obs[k]).view(1, *next_obs[k].shape).float()
        done = done or truncated or cfg.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    if len(fabric.loggers) > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()

@hydra.main(version_base=None, config_path="sheeprl/configs", config_name="config")
def main(cfg: DictConfig):
    fabric = Fabric(accelerator="cpu")

    print(cfg)
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    # the same as the dreamer-v3 training
    rank = 0
    # fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    # Use the load method on the instance
    state = fabric.load(path="ckpt_200000_0.ckpt")

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            partial(
                RestartOnException,
                make_env(
                    cfg,
                    cfg.seed + rank * cfg.env.num_envs + i,
                    rank * cfg.env.num_envs,
                    "train",
                    vector_env_idx=i,
                ),
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space
    print(observation_space)
    print(action_space)

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = (
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )

    world_model, actor, critic, target_critic = build_models(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"],
        state["actor"],
        state["critic"],
        state["target_critic"],
    )
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
    print(fabric.device)
    envs.close()
    test(player, fabric, cfg, sample_actions=True)

if __name__=="__main__":
    main()