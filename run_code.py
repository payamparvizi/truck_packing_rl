#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:06:23 2026

@author: payam
"""

import jax_PACK
import random
import argparse
import datetime
import os
import numpy as np
import pprint

import gymnasium as gym
# from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import FrameStack

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from packages.tianshou.env import DummyVectorEnv
from packages.tianshou.utils.net.continuous import ActorProb, Critic
from packages.tianshou.utils.net.common import ActorCritic
from packages.tianshou.policy import PPOPolicy
from packages.tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from packages.tianshou.utils import WandbLogger
from packages.tianshou.trainer import OnpolicyTrainer

# from utils.network import TripleGridNet
from utils.networkCNN import TripleGridNet
from utils.arguments import get_args


def env_config(args, seed=0):
    env = gym.make("PACK-v0", 
                   render=False, 
                   seed=seed, 
                   grid_res=args.grid_res, 
                   num_boxes=args.num_boxes)
    
    if args.stack_num > 1:
        env = FrameStackObservation(env, args.stack_num)
    return env


def test_env(args: argparse.Namespace = get_args()) -> None:

    env = env_config(args)

    train_envs = DummyVectorEnv([
        lambda seed=args.seed + i: env_config(args, seed)
        for i in range(args.training_num)
        ])

    test_envs = DummyVectorEnv([
        lambda seed=args.seed + 10000 + i: env_config(args, seed) 
        for i in range(args.test_num)
        ])

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    print("Observation shape:", args.state_shape)
    print("Action shape:", args.action_shape)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    net_a = TripleGridNet(
        state_shape=args.state_shape,
        hidden_sizes=(args.hidden_size_1, args.hidden_size_2),
        fusion_hidden_sizes=(args.hidden_size_1, args.hidden_size_2),
        device=args.device,
        )
    
    actor = ActorProb(
        net_a,
        args.action_shape,
        unbounded=False,
        device=args.device,
    ).to(args.device)
    
    net_c = TripleGridNet(
        state_shape=args.state_shape,
        hidden_sizes=(args.hidden_size_1, args.hidden_size_2),
        fusion_hidden_sizes=(args.hidden_size_1, args.hidden_size_2),
        device=args.device,
    )
    
    critic = Critic(net_c, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: max(0.0, 1 - epoch / max_update_num)
        )
            
    def dist(*logits):
        return Independent(Normal(*logits), 1)
    
    policy: PPOPolicy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )
    
    
    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)
    
    
    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
        
    train_collector = Collector(policy=policy, env=train_envs, buffer=buffer, 
                                exploration_noise=True)
    
    test_collector = Collector(policy=policy, env=test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    logger = WandbLogger(
    save_interval= 1,
    train_interval = 2,
    test_interval = 1,
    update_interval = 2,
    
    name=log_name.replace(os.path.sep, "_"),
    config=args,
    project="results_packaging_v15f"
    )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.torch"))
    
    if not args.watch:
        # trainer
        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=True,
        ).run()
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)

    policy_file_path = os.path.join(log_path, "policy.torch")
    policy.load_state_dict(torch.load(policy_file_path))    


if __name__ == "__main__":
    test_env(get_args())
    
    
    
# import jax_PACK

# import numpy as np
# import argparse
# import gymnasium as gym
# import time

# #env = PACKEnv(render=False, seed=0)
# set_time = time.time()
# seedp = 0

# for j in range(1):
#     env = gym.make('PACK-v0',render=True, seed=seedp)
#     obs, _ = env.reset()
    
#     # print("reset:", np.sum(obs), obs.shape)
#     action = np.array([1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
#     for i in range(1000):
#         obs, reward, terminate, truncate, info = env.step(action)
#         print(j,i)
#         seedp += 1
#         time.sleep(2)
        
#         # print(f"\nStep {i}")
#         # print("  sum:", np.sum(obs))
#         # print("  reward:", reward)
#         # print("  terminate:", terminate)
#         # print("  truncate:", truncate)
#         # print("  info:", info)

# current_time = time.time()
# diff = current_time - set_time
# print(diff)  # seconds
