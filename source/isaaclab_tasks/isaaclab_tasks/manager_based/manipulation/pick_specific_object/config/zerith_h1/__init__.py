# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import pick_ik_rel_env_cfg, pick_pointvla_env_cfg, pick_pointvla_eval_env_cfg

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Pick-Specific-Object-ZerithH1-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pick_ik_rel_env_cfg.ZerithH1PickSpecificObjectIKRelEnvCfg,
    },
    disable_env_checker=True,
)

##
# PointVLA data collection (30 Hz RGB head + dual wrist)
##

gym.register(
    id="Isaac-Pick-Specific-Object-ZerithH1-IK-Rel-PointVLA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pick_pointvla_env_cfg.ZerithH1PickSpecificObjectPointVLAEnvCfg,
    },
    disable_env_checker=True,
)

##
# PointVLA closed-loop eval (23-DoF absolute joint targets)
##

gym.register(
    id="Isaac-Pick-Specific-Object-ZerithH1-JointAbs-PointVLA-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pick_pointvla_eval_env_cfg.ZerithH1PickSpecificObjectPointVLAEvalEnvCfg,
    },
    disable_env_checker=True,
)
