# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from isaaclab_tasks.manager_based.manipulation.stack.config.franka import agents as franka_agents

from . import stack_ik_rel_env_cfg_skillgen, stack_joint_pos_env_cfg

##
# Joint Position Control
##

gym.register(
    id="Isaac-Stack-Cube-ZerithH1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_env_cfg.ZerithH1CubeStackEnvCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control (SkillGen)
##

gym.register(
    id="Isaac-Stack-Cube-ZerithH1-IK-Rel-Skillgen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_env_cfg_skillgen.ZerithH1CubeStackSkillgenEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(franka_agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)
