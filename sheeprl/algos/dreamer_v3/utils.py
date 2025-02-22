import os
from typing import TYPE_CHECKING, Any, Dict

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from torch import Tensor

from sheeprl.utils.env import make_env

if TYPE_CHECKING:
    from sheeprl.algos.dreamer_v3.agent import PlayerDV3


class Moments(torch.nn.Module):
    def __init__(
        self,
        fabric: Fabric,
        decay: float = 0.99,
        max_: float = 1e8,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._fabric = fabric
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor) -> Any:
        gathered_x = self._fabric.all_gather(x).detach()
        low = torch.quantile(gathered_x, self._percentile_low)
        high = torch.quantile(gathered_x, self._percentile_high)
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    lmbda: float = 0.95,
):
    vals = [values[-1:]]
    interm = rewards + continues * values * (1 - lmbda)
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret


@torch.no_grad()
def test(
    player: "PlayerDV3",
    fabric: Fabric,
    cfg: Dict[str, Any],
    test_name: str = "",
    sample_actions: bool = False,
):
    """Test the model on the environment with the frozen model.

    Args:
        player (PlayerDV3): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
        cfg (DictConfig): the hyper-parameters.
        test_name (str): the name of the test.
            Default to "".
        sample_actions (bool): whether or not to sample the actions.
            Default to False.
    """
    log_dir = fabric.logger.log_dir if len(fabric.loggers) > 0 else os.getcwd()
    env: gym.Env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name != "" else ""))()
    done = False
    cumulative_rew = 0
    device = fabric.device
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
        for k in next_obs.keys():
            next_obs[k] = torch.from_numpy(next_obs[k]).view(1, *next_obs[k].shape).float()
        done = done or truncated or cfg.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    if len(fabric.loggers) > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        torch.nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        torch.nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(given_scale):
    def f(m):
        if isinstance(m, torch.nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            torch.nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, torch.nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


@torch.no_grad()
def test_vis(
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
    return cumulative_rew