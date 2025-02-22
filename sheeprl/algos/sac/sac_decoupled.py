import copy
import os
import pathlib
import warnings
from datetime import timedelta
from math import prod
from typing import Any, Dict

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy
from omegaconf import OmegaConf
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.utils.data.sampler import BatchSampler
from torchmetrics import MeanMetric, SumMetric

from sheeprl.algos.sac.agent import SACActor, SACAgent, SACCritic
from sheeprl.algos.sac.sac import train
from sheeprl.algos.sac.utils import test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.env import make_env
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import dotdict


@torch.no_grad()
def player(
    fabric: Fabric, cfg: Dict[str, Any], world_collective: TorchCollective, player_trainer_collective: TorchCollective
):
    logger = fabric.logger
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        root_dir = cfg.root_dir
        run_name = cfg.run_name
        state = fabric.load(cfg.checkpoint.resume_from)
        ckpt_path = pathlib.Path(cfg.checkpoint.resume_from)
        cfg = dotdict(OmegaConf.load(ckpt_path.parent.parent.parent / ".hydra" / "config.yaml"))
        cfg.checkpoint.resume_from = str(ckpt_path)
        cfg.per_rank_batch_size = state["batch_size"] // fabric.world_size
        cfg.root_dir = root_dir
        cfg.run_name = run_name

    if len(cfg.cnn_keys.encoder) > 0:
        warnings.warn("SAC algorithm cannot allow to use images as observations, the CNN keys will be ignored")
        cfg.cnn_keys.encoder = []

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                cfg,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank * cfg.env.num_envs,
                logger.log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported for the SAC agent")
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if len(cfg.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `mlp_keys.encoder=[state]`")
    for k in cfg.mlp_keys.encoder:
        if len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the SAC agent. "
                f"Provided environment: {cfg.env.id}"
            )

    # Send (possibly updated, by the make_env method for example) cfg to the trainers
    world_collective.broadcast_object_list([cfg], src=0)

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(action_space.shape)
    obs_dim = sum([prod(observation_space[k].shape) for k in cfg.mlp_keys.encoder])
    actor = SACActor(
        observation_dim=obs_dim,
        action_dim=act_dim,
        distribution_cfg=cfg.distribution,
        hidden_size=cfg.algo.actor.hidden_size,
        action_low=action_space.low,
        action_high=action_space.high,
    ).to(device)
    flattened_parameters = torch.empty_like(
        torch.nn.utils.convert_parameters.parameters_to_vector(actor.parameters()), device=device
    )

    # Receive the first weights from the rank-1, a.k.a. the first of the trainers
    # In this way we are sure that before the first iteration everyone starts with the same parameters
    player_trainer_collective.broadcast(flattened_parameters, src=1)
    torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, actor.parameters())

    # Metrics
    aggregator = MetricAggregator(
        {"Rewards/rew_avg": MeanMetric(sync_on_compute=False), "Game/ep_len_avg": MeanMetric(sync_on_compute=False)}
    ).to(device)

    # Local data
    buffer_size = cfg.buffer.size // cfg.env.num_envs if not cfg.dry_run else 1
    rb = ReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        device=device,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(logger.log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], ReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=device)

    # Global variables
    first_info_sent = False
    start_step = state["update"] if cfg.checkpoint.resume_from else 1
    policy_step = (state["update"] - 1) * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs)
    num_updates = int(cfg.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    if cfg.checkpoint.resume_from and not cfg.buffer.checkpoint:
        learning_starts += start_step

    # Warning for log and checkpoint every
    if cfg.metric.log_every % policy_steps_per_update != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % policy_steps_per_update != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    with device:
        # Get the first environment observation and start the optimization
        o = envs.reset(seed=cfg.seed)[0]
        obs = torch.cat(
            [torch.tensor(o[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder], dim=-1
        )  # [N_envs, N_obs]

    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
            if update <= learning_starts:
                actions = envs.action_space.sample()
            else:
                # Sample an action given the observation received by the environment
                with torch.no_grad():
                    actions, _ = actor(obs)
                    actions = actions.cpu().numpy()
            next_obs, rewards, dones, truncated, infos = envs.step(actions)
            dones = np.logical_or(dones, truncated)

        if "final_info" in infos:
            for i, agent_ep_info in enumerate(infos["final_info"]):
                if agent_ep_info is not None:
                    ep_rew = agent_ep_info["episode"]["r"]
                    ep_len = agent_ep_info["episode"]["l"]
                    aggregator.update("Rewards/rew_avg", ep_rew)
                    aggregator.update("Game/ep_len_avg", ep_len)
                    fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Save the real next observation
        real_next_obs = copy.deepcopy(next_obs)
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v

        with device:
            next_obs = torch.cat(
                [torch.tensor(next_obs[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder], dim=-1
            )  # [N_envs, N_obs]
            real_next_obs = torch.cat(
                [torch.tensor(real_next_obs[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder], dim=-1
            )  # [N_envs, N_obs]
            actions = torch.tensor(actions, dtype=torch.float32).view(cfg.env.num_envs, -1)
            rewards = torch.tensor(rewards, dtype=torch.float32).view(cfg.env.num_envs, -1)  # [N_envs, 1]
            dones = torch.tensor(dones, dtype=torch.float32).view(cfg.env.num_envs, -1)

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        if not cfg.buffer.sample_next_obs:
            step_data["next_observations"] = real_next_obs
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = next_obs

        # Send data to the training agents
        if update >= learning_starts:
            # Send local info to the trainers
            if not first_info_sent:
                world_collective.broadcast_object_list(
                    [{"update": update, "last_log": last_log, "last_checkpoint": last_checkpoint}], src=0
                )
                first_info_sent = True

            # Sample data to be sent to the trainers
            training_steps = learning_starts if update == learning_starts else 1
            chunks = rb.sample(
                training_steps * cfg.algo.per_rank_gradient_steps * cfg.per_rank_batch_size * (fabric.world_size - 1),
                sample_next_obs=cfg.buffer.sample_next_obs,
            ).split(training_steps * cfg.algo.per_rank_gradient_steps * cfg.per_rank_batch_size)
            world_collective.scatter_object_list([None], [None] + chunks, src=0)

            # Wait the trainers to finish
            player_trainer_collective.broadcast(flattened_parameters, src=1)

            # Convert back the parameters
            torch.nn.utils.convert_parameters.vector_to_parameters(flattened_parameters, actor.parameters())

            # Logs trainers-only metrics
            if policy_step - last_log >= cfg.metric.log_every or cfg.dry_run:
                # Gather metrics from the trainers
                metrics = [None]
                player_trainer_collective.broadcast_object_list(metrics, src=1)

                # Log metrics
                fabric.log_dict(metrics[0], policy_step)

        # Logs player-only metrics
        if policy_step - last_log >= cfg.metric.log_every or cfg.dry_run:
            fabric.log_dict(aggregator.compute(), policy_step)
            aggregator.reset()

            # Sync timers
            timer_metrics = timer.compute()
            fabric.log(
                "Time/sps_env_interaction",
                ((policy_step - last_log) * cfg.env.action_repeat) / timer_metrics["Time/env_interaction_time"],
                policy_step,
            )
            timer.reset()

            # Reset counters
            last_log = policy_step

        # Checkpoint model
        if (
            update >= learning_starts  # otherwise the processes end up deadlocked
            and cfg.checkpoint.every > 0
            and policy_step - last_checkpoint >= cfg.checkpoint.every
        ) or cfg.dry_run:
            last_checkpoint = policy_step
            ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_player",
                fabric=fabric,
                player_trainer_collective=player_trainer_collective,
                ckpt_path=ckpt_path,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    world_collective.scatter_object_list([None], [None] + [-1] * (world_collective.world_size - 1), src=0)

    # Last Checkpoint
    ckpt_path = fabric.logger.log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
    fabric.call(
        "on_checkpoint_player",
        fabric=fabric,
        player_trainer_collective=player_trainer_collective,
        ckpt_path=ckpt_path,
        replay_buffer=rb if cfg.buffer.checkpoint else None,
    )

    envs.close()
    if fabric.is_global_zero:
        test(actor, fabric, cfg)


def trainer(
    world_collective: TorchCollective,
    player_trainer_collective: TorchCollective,
    optimization_pg: CollectibleGroup,
):
    global_rank = world_collective.rank
    group_world_size = world_collective.world_size - 1

    # Receive (possibly updated, by the make_env method for example) cfg from the player
    data = [None]
    world_collective.broadcast_object_list(data, src=0)
    cfg: Dict[str, Any] = data[0]

    # Initialize Fabric
    fabric = Fabric(
        strategy=DDPStrategy(process_group=optimization_pg),
        devices=cfg.fabric.devices,
        callbacks=[CheckpointCallback()],
    )
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)
        ckpt_path = pathlib.Path(cfg.checkpoint.resume_from)
        cfg = dotdict(OmegaConf.load(ckpt_path.parent.parent.parent / ".hydra" / "config.yaml"))
        cfg.checkpoint.resume_from = str(ckpt_path)
        cfg.per_rank_batch_size = state["batch_size"] // fabric.world_size

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env([make_env(cfg, 0, 0, None)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(envs.single_action_space.shape)
    obs_dim = sum([prod(envs.single_observation_space[k].shape) for k in cfg.mlp_keys.encoder])

    actor = SACActor(
        observation_dim=obs_dim,
        action_dim=act_dim,
        distribution_cfg=cfg.distribution,
        hidden_size=cfg.algo.actor.hidden_size,
        action_low=envs.single_action_space.low,
        action_high=envs.single_action_space.high,
    )
    critics = [
        SACCritic(observation_dim=obs_dim + act_dim, hidden_size=cfg.algo.critic.hidden_size, num_critics=1)
        for _ in range(cfg.algo.critic.n)
    ]
    target_entropy = -act_dim
    agent = SACAgent(actor, critics, target_entropy, alpha=cfg.algo.alpha.alpha, tau=cfg.algo.tau, device=fabric.device)
    if cfg.checkpoint.resume_from:
        agent.load_state_dict(state["agent"])
    agent.actor = fabric.setup_module(agent.actor)
    agent.critics = [fabric.setup_module(critic) for critic in agent.critics]

    # Optimizers
    qf_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=agent.qfs.parameters())
    actor_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer,
        params=agent.actor.parameters(),
    )
    alpha_optimizer = hydra.utils.instantiate(cfg.algo.alpha.optimizer, params=[agent.log_alpha])
    if cfg.checkpoint.resume_from:
        qf_optimizer.load_state_dict(state["qf_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        alpha_optimizer.load_state_dict(state["alpha_optimizer"])
    qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
        qf_optimizer, actor_optimizer, alpha_optimizer
    )

    # Send weights to rank-0, a.k.a. the player
    if global_rank == 1:
        player_trainer_collective.broadcast(
            torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters()), src=1
        )

    # Metrics
    aggregator = MetricAggregator(
        {
            "Loss/value_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg),
            "Loss/policy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg),
            "Loss/alpha_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg),
        }
    ).to(device)

    # Receive data from player reagrding the:
    # * update
    # * last_log
    # * last_checkpoint
    data = [None]
    world_collective.broadcast_object_list(data, src=0)
    update = data[0]["update"]
    last_log = data[0]["last_log"]
    last_checkpoint = data[0]["last_checkpoint"]

    # Start training
    train_step = 0
    last_train = 0
    policy_steps_per_update = cfg.env.num_envs
    policy_step = update * policy_steps_per_update
    while True:
        # Wait for data
        data = [None]
        world_collective.scatter_object_list(data, [None for _ in range(world_collective.world_size)], src=0)
        data = data[0]
        if not isinstance(data, TensorDictBase) and data == -1:
            # Last Checkpoint
            if global_rank == 1:
                state = {
                    "agent": agent.state_dict(),
                    "qf_optimizer": qf_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "alpha_optimizer": alpha_optimizer.state_dict(),
                    "update": update,
                    "batch_size": cfg.per_rank_batch_size * (world_collective.world_size - 1),
                    "last_log": last_log,
                    "last_checkpoint": last_checkpoint,
                }
                fabric.call("on_checkpoint_trainer", player_trainer_collective=player_trainer_collective, state=state)
            return
        data = make_tensordict(data, device=device)
        sampler = BatchSampler(range(len(data)), batch_size=cfg.per_rank_batch_size, drop_last=False)

        # Start training
        with timer(
            "Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute, process_group=optimization_pg)
        ):
            for batch_idxes in sampler:
                train(
                    fabric,
                    agent,
                    actor_optimizer,
                    qf_optimizer,
                    alpha_optimizer,
                    data[batch_idxes],
                    aggregator,
                    update,
                    cfg,
                    policy_steps_per_update,
                    group=optimization_pg,
                )
            train_step += group_world_size

        if global_rank == 1:
            player_trainer_collective.broadcast(
                torch.nn.utils.convert_parameters.parameters_to_vector(agent.actor.parameters()), src=1
            )

        if policy_step - last_log >= cfg.metric.log_every or cfg.dry_run:
            # Sync distributed metrics
            metrics = aggregator.compute()
            aggregator.reset()

            # Sync distributed timers
            timers = timer.compute()
            metrics.update({"Time/sps_train": (train_step - last_train) / timers["Time/train_time"]})
            timer.reset()

            if global_rank == 1:
                player_trainer_collective.broadcast_object_list(
                    [metrics], src=1
                )  # Broadcast metrics: fake send with object list between rank-0 and rank-1

            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint model on rank-0: send it everything
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or cfg.dry_run:
            last_checkpoint = policy_step
            if global_rank == 1:
                state = {
                    "agent": agent.state_dict(),
                    "qf_optimizer": qf_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "alpha_optimizer": alpha_optimizer.state_dict(),
                    "update": update,
                    "batch_size": cfg.per_rank_batch_size * (world_collective.world_size - 1),
                    "last_log": last_log,
                    "last_checkpoint": last_checkpoint,
                }
                fabric.call("on_checkpoint_trainer", player_trainer_collective=player_trainer_collective, state=state)

        # Update counters
        update += 1
        policy_step += policy_steps_per_update


@register_algorithm(decoupled=True)
def main(fabric: Fabric, cfg: Dict[str, Any]):
    if fabric.world_size == 1:
        raise RuntimeError(
            "Please run the script with the number of devices greater than 1: "
            "`python sheeprl.py exp=sac_decoupled fabric.devices=2 ...`"
        )

    if "minedojo" in cfg.env.wrapper._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by SAC agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
        )

    world_collective = TorchCollective()
    player_trainer_collective = TorchCollective()
    world_collective.setup(
        backend="nccl" if os.environ.get("LT_ACCELERATOR", None) in ("gpu", "cuda") else "gloo",
        timeout=timedelta(days=1),
    )

    # Create a global group, assigning it to the collective: used by the player to exchange
    # collected experiences with the trainers
    world_collective.create_group(timeout=timedelta(days=1))
    global_rank = world_collective.rank

    # Create a group between rank-0 (player) and rank-1 (trainer), assigning it to the collective:
    # used by rank-1 to send metrics to be tracked by the rank-0 at the end of a training episode
    player_trainer_collective.create_group(ranks=[0, 1], timeout=timedelta(days=1))

    # Create a new group, without assigning it to the collective: in this way the trainers can
    # still communicate with the player through the global group, but they can optimize the agent
    # between themselves
    optimization_pg = world_collective.new_group(
        ranks=list(range(1, world_collective.world_size)), timeout=timedelta(days=1)
    )
    if global_rank == 0:
        player(fabric, cfg, world_collective, player_trainer_collective)
    else:
        trainer(world_collective, player_trainer_collective, optimization_pg)
