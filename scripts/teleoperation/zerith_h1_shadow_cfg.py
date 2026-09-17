# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Zerith H1 shadow-runner constants and read-only JointState decoding."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import numpy as np

ZERITH_H1_USD_PATH = Path(r"D:\robot\sim\zerith_H1\robot_fix.usda")
LEFT_ARM_JOINTS = ("left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint")
RIGHT_ARM_JOINTS = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint")
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
REAL_JOINT_STATE_INDICES = tuple(range(5, 12)) + tuple(range(14, 21))
DEFAULT_COMMAND_TIMEOUT_S = 0.5
DEFAULT_PHYSICS_DT_S = 1.0 / 240.0
DEFAULT_LOG_DIR = Path("logs/zerith_h1_shadow")

@dataclass(frozen=True)
class ShadowTopics:
    """Topics consumed by the runner. The runner never publishes ROS messages."""
    final_left_arm: str = "/h1/left_arm/motors_cmd"
    final_right_arm: str = "/h1/right_arm/motors_cmd"
    joint_states: str = "/h1/joint_states"
    mux_input_left_arm: str = "/control/mux/in/xr/left_arm/motors_cmd"
    mux_input_right_arm: str = "/control/mux/in/xr/right_arm/motors_cmd"
    xr_pose: str = "/xr_pose"

DEFAULT_TOPICS = ShadowTopics()
SIM_TOPICS = ShadowTopics(
    final_left_arm="/sim/control/mux/in/xr/left_arm/motors_cmd",
    final_right_arm="/sim/control/mux/in/xr/right_arm/motors_cmd",
    joint_states="/sim/h1/joint_states",
    mux_input_left_arm="/sim/control/mux/in/xr/left_arm/motors_cmd",
    mux_input_right_arm="/sim/control/mux/in/xr/right_arm/motors_cmd",
    xr_pose="/sim/xr_pose",
)

class JointStateLike(Protocol):
    name: list[str]
    position: list[float]


def validate_standalone_topics(topics: ShadowTopics) -> None:
    """Reject standalone topic maps that could publish onto a real robot graph."""
    for topic in (topics.final_left_arm, topics.final_right_arm, topics.joint_states):
        if not topic.startswith("/sim/"):
            raise ValueError(f"Standalone topics must use the /sim namespace: {topic}")


def encode_sim_joint_state(arm_positions: np.ndarray) -> np.ndarray:
    """Encode arm positions in pico-teleop's 21-element JointState layout."""
    values = np.asarray(arm_positions, dtype=np.float64)
    if values.shape != (len(ARM_JOINTS),) or not np.all(np.isfinite(values)):
        raise ValueError("Expected 14 finite Zerith arm joint positions.")
    state = np.zeros(21, dtype=np.float64)
    state[5:12] = values[:7]
    state[14:21] = values[7:]
    return state

def decode_arm_command(message: JointStateLike, joint_names: tuple[str, ...], side: str) -> tuple[np.ndarray | None, str]:
    """Return seven values by canonical name, or Pico's documented positional layout."""
    names, values = list(message.name), np.asarray(message.position, dtype=np.float64)
    if values.size < len(joint_names) or not np.all(np.isfinite(values)):
        return None, "invalid"
    by_name = {name: index for index, name in enumerate(names)}
    if all(name in by_name for name in joint_names):
        return np.asarray([values[by_name[name]] for name in joint_names]), "named"
    generic_names = [f"{side}_arm_joint_{index}" for index in range(len(joint_names))]
    if not names or names == generic_names:
        return values[: len(joint_names)].copy(), "positional"
    return None, "missing_joint"

def decode_real_joint_state(message: JointStateLike) -> tuple[np.ndarray | None, str]:
    """Return the dual-arm feedback using names first, then pico-teleop's known layout."""
    names, values = list(message.name), np.asarray(message.position, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None, "invalid"
    by_name = {name: index for index, name in enumerate(names)}
    if all(name in by_name for name in ARM_JOINTS):
        return np.asarray([values[by_name[name]] for name in ARM_JOINTS]), "named"
    if values.size > max(REAL_JOINT_STATE_INDICES):
        return values[list(REAL_JOINT_STATE_INDICES)].copy(), "controller_layout"
    return None, "missing_joint"
