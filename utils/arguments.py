#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:02:32 2026

@author: payam
"""

import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    
    # Training specific information
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden_size_1", type=int, default=128)
    parser.add_argument("--hidden_size_2", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step-per-epoch", type=int, default=12000)
    parser.add_argument("--step-per-collect", type=int, default=1024)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    
    parser.add_argument("--training-num", type=int, default=4)
    parser.add_argument("--test_num", type=int, default=10)
    parser.add_argument("--max_steps_per_episode", type=int, default=120)
    
    parser.add_argument("--rew-norm", type=int, default=True)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent_coef", type=float, default=0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    
    parser.add_argument("--grid_res", type=int, default=40)
    parser.add_argument("--num_boxes", type=int, default=120)
    
    parser.add_argument("--stack_num", type=int, default=1)
    
    # Environment specific information
    parser.add_argument("--task", type=str, default='PACK-v0')
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    
    return parser.parse_args()
    