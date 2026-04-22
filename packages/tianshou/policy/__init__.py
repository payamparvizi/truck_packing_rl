"""Policy package."""
# isort:skip_file

from packages.tianshou.policy.base import BasePolicy
from packages.tianshou.policy.random import RandomPolicy
from packages.tianshou.policy.modelfree.dqn import DQNPolicy
from packages.tianshou.policy.modelfree.bdq import BranchingDQNPolicy
from packages.tianshou.policy.modelfree.c51 import C51Policy
from packages.tianshou.policy.modelfree.rainbow import RainbowPolicy
from packages.tianshou.policy.modelfree.qrdqn import QRDQNPolicy
from packages.tianshou.policy.modelfree.iqn import IQNPolicy
from packages.tianshou.policy.modelfree.fqf import FQFPolicy
from packages.tianshou.policy.modelfree.pg import PGPolicy
from packages.tianshou.policy.modelfree.a2c import A2CPolicy
from packages.tianshou.policy.modelfree.npg import NPGPolicy
from packages.tianshou.policy.modelfree.ddpg import DDPGPolicy
from packages.tianshou.policy.modelfree.ppo import PPOPolicy
from packages.tianshou.policy.modelfree.trpo import TRPOPolicy
from packages.tianshou.policy.modelfree.td3 import TD3Policy
from packages.tianshou.policy.modelfree.sac import SACPolicy
from packages.tianshou.policy.modelfree.redq import REDQPolicy
from packages.tianshou.policy.modelfree.discrete_sac import DiscreteSACPolicy
from packages.tianshou.policy.imitation.base import ImitationPolicy
from packages.tianshou.policy.imitation.bcq import BCQPolicy
from packages.tianshou.policy.imitation.cql import CQLPolicy
from packages.tianshou.policy.imitation.td3_bc import TD3BCPolicy
from packages.tianshou.policy.imitation.discrete_bcq import DiscreteBCQPolicy
from packages.tianshou.policy.imitation.discrete_cql import DiscreteCQLPolicy
from packages.tianshou.policy.imitation.discrete_crr import DiscreteCRRPolicy
from packages.tianshou.policy.imitation.gail import GAILPolicy
from packages.tianshou.policy.modelbased.psrl import PSRLPolicy
from packages.tianshou.policy.modelbased.icm import ICMPolicy
from packages.tianshou.policy.multiagent.mapolicy import MultiAgentPolicyManager

__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "DQNPolicy",
    "BranchingDQNPolicy",
    "C51Policy",
    "RainbowPolicy",
    "QRDQNPolicy",
    "IQNPolicy",
    "FQFPolicy",
    "PGPolicy",
    "A2CPolicy",
    "NPGPolicy",
    "DDPGPolicy",
    "PPOPolicy",
    "TRPOPolicy",
    "TD3Policy",
    "SACPolicy",
    "REDQPolicy",
    "DiscreteSACPolicy",
    "ImitationPolicy",
    "BCQPolicy",
    "CQLPolicy",
    "TD3BCPolicy",
    "DiscreteBCQPolicy",
    "DiscreteCQLPolicy",
    "DiscreteCRRPolicy",
    "GAILPolicy",
    "PSRLPolicy",
    "ICMPolicy",
    "MultiAgentPolicyManager",
]
