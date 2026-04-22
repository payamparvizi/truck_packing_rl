# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:39:12 2023

@author: payam
"""
from gymnasium.envs.registration import register

register(
    id='PACK-v0',
    entry_point='jax_PACK.envs:PACKEnv',
)